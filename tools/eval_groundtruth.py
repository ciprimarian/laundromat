#!/usr/bin/env python3
"""Fuzzy-match pipeline findings against a free-text ground-truth markdown.

Tolerates sealed/human ground truth: extracts entity ids, document numbers and
amounts from markdown (or plain text) and pairs them to scored findings.

Usage:
  python tools/eval_groundtruth.py path/to/ground_truth.md [dossier_path]
"""

from __future__ import annotations

import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from laundromat.contracts import Finding, Tier  # noqa: E402
from laundromat.pipeline import run  # noqa: E402

# entity-like ids (vendors 2xxxxx, customers 1xxxxx, assets, etc.)
_ENTITY_RX = re.compile(r"\b([12]\d{5}|20\d{4}|10\d{4}|2091\d{2})\b")
# document numbers seen in practice: ER/AR/SG/WE/WA/BE + digits, GJ..., AB-2024
_DOC_RX = re.compile(
    r"\b((?:ER|AR|SG|WE|WA|BE|RE|KR|DR)[\-_]?\d{4,}|GJ\d{5,}|AB[\-_]\d{2,4})\b",
    re.I,
)
_AMT_RX = re.compile(
    r"(?<![\w/])(-?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|-?\d+[.,]\d{2})(?!\w)"
)


def parse_amount(token: str) -> Decimal | None:
    s = token.strip().replace(" ", "")
    if not s:
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        head, _, tail = s.rpartition(",")
        if len(tail) <= 2:
            s = head + "." + tail
        else:
            s = s.replace(",", "")
    try:
        return abs(Decimal(s))
    except InvalidOperation:
        return None


def extract_items(text: str) -> list[dict]:
    """Split ground truth into items (headings / bullets / numbered lines)."""
    chunks: list[str] = []
    buf: list[str] = []
    for line in text.splitlines():
        if re.match(r"^\s*#{1,3}\s+\S", line) or re.match(r"^\s*[-*+]\s+\S", line) or re.match(
            r"^\s*\d+[.)]\s+\S", line
        ):
            if buf:
                chunks.append("\n".join(buf))
                buf = []
            buf.append(line)
        else:
            if buf or line.strip():
                buf.append(line)
    if buf:
        chunks.append("\n".join(buf))
    if not chunks:
        chunks = [text]

    items = []
    for i, chunk in enumerate(chunks):
        entities = list(dict.fromkeys(_ENTITY_RX.findall(chunk)))
        docs = list(dict.fromkeys(m.group(1).upper().replace("_", "-") for m in _DOC_RX.finditer(chunk)))
        amounts: list[Decimal] = []
        for m in _AMT_RX.finditer(chunk):
            a = parse_amount(m.group(1))
            if a is not None and a >= Decimal("100"):
                amounts.append(a)
        # skip pure TOC / empty
        if not entities and not docs and not amounts and len(chunk.strip()) < 20:
            continue
        items.append(
            {
                "id": i + 1,
                "text": chunk.strip()[:240],
                "entities": entities,
                "docs": docs,
                "amounts": amounts,
            }
        )
    return items


def finding_keys(f: Finding) -> tuple[set[str], set[str], list[Decimal]]:
    entities: set[str] = set()
    docs: set[str] = set()
    amounts: list[Decimal] = []
    if f.subject_kind == "entity":
        entities.add(f.subject_id)
    if f.subject_kind == "transaction":
        docs.add(f.subject_id.upper())
    for fl in f.flags:
        if fl.entity_id:
            entities.add(fl.entity_id)
        if fl.doc_no:
            docs.add(fl.doc_no.upper())
        if fl.amount is not None:
            amounts.append(abs(fl.amount))
    if f.max_amount:
        amounts.append(abs(f.max_amount))
    return entities, docs, amounts


def match_score(item: dict, f: Finding) -> tuple[float, str]:
    ents, docs, amts = finding_keys(f)
    reasons = []
    score = 0.0
    ent_hit = set(item["entities"]) & ents
    if ent_hit:
        score += 3.0
        reasons.append(f"entity={','.join(sorted(ent_hit))}")
    doc_hit = set(d.upper() for d in item["docs"]) & docs
    if doc_hit:
        score += 3.0
        reasons.append(f"doc={','.join(sorted(doc_hit))}")
    # amount within 1%
    for ga in item["amounts"]:
        for fa in amts:
            if fa == 0:
                continue
            if abs(ga - fa) <= max(Decimal("1"), fa * Decimal("0.01")):
                score += 2.0
                reasons.append(f"amount={fa}")
                break
        else:
            continue
        break
    # weak: subject id string in free text
    if score == 0 and f.subject_id and f.subject_id in item["text"]:
        score += 1.0
        reasons.append("subject_in_text")
    return score, "; ".join(reasons) if reasons else "-"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__.strip())
        print("\nerror: ground-truth path required", file=sys.stderr)
        return 2
    gt_path = Path(argv[0])
    dossier_path = argv[1] if len(argv) > 1 else "data/practice"
    if not gt_path.is_file():
        print(f"error: ground truth not found: {gt_path}", file=sys.stderr)
        return 2

    text = gt_path.read_text(encoding="utf-8", errors="replace")
    items = extract_items(text)
    print(f"ground truth: {gt_path} -> {len(items)} item(s)")
    print(f"running pipeline on {dossier_path} ...")
    _d, _flags, findings = run(dossier_path)
    # only reportable findings for recall
    findings = [f for f in findings if f.tier != Tier.DISMISSED]
    print(f"findings (non-dismissed): {len(findings)}")

    # greedy best match each GT item to at most one finding
    used: set[int] = set()
    found: list[tuple] = []
    missed: list[dict] = []
    for item in items:
        best = None
        best_score = 0.0
        best_reason = "-"
        for i, f in enumerate(findings):
            if i in used:
                continue
            sc, reason = match_score(item, f)
            if sc > best_score:
                best_score, best, best_reason = sc, i, reason
        if best is not None and best_score >= 2.0:
            used.add(best)
            f = findings[best]
            found.append((item, f, best_score, best_reason))
        else:
            missed.append(item)

    extra = [f for i, f in enumerate(findings) if i not in used]

    print()
    print(f"{'status':8s}  {'gt/finding':40s}  {'tier':8s}  match")
    print("-" * 80)
    for item, f, sc, reason in found:
        label = f"GT#{item['id']} -> {f.subject_id}"
        print(f"{'FOUND':8s}  {label[:40]:40s}  {f.tier.value:8s}  {reason} (score={sc:.0f})")
    for item in missed:
        hint = item["entities"][:2] or item["docs"][:2] or ["?"]
        print(f"{'MISSED':8s}  GT#{item['id']} {str(hint)[:30]:30s}  {'':8s}  {(item['text'][:50]).replace(chr(10),' ')}")
    for f in extra:
        print(f"{'EXTRA':8s}  {f.subject_id[:40]:40s}  {f.tier.value:8s}  families={sorted(x.value for x in f.families)}")

    print()
    print(
        f"SUMMARY: found={len(found)} missed={len(missed)} extra={len(extra)} "
        f"(gt_items={len(items)} findings={len(findings)})"
    )
    if items:
        print(f"  recall≈ {len(found)/len(items):.0%}  precision≈ {len(found)/max(1,len(found)+len(extra)):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

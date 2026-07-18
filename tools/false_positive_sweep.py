#!/usr/bin/env python3
"""Precision self-audit: innocent-explanation heuristics on HIGH/MEDIUM findings.

For each high/medium finding, test whether a benign story fits:
  - opening / closing / vortrag journal
  - storno / reversal pair (equal amount, opposite sign, close dates)
  - counterparty is a shareholder / intercompany party
  - recurring same-amount monthly pattern (rent, payroll, leasing)

Usage:
  python tools/false_positive_sweep.py [dossier_path]
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from laundromat.contracts import Dossier, Finding, Posting, Tier  # noqa: E402
from laundromat.pipeline import run  # noqa: E402

# Bare "Vortrag" is a period code on many normal invoices in GDPdU exports;
# only true carry-forward language counts.
_OPENING_RX = re.compile(
    r"saldenvortrag|opening\s+balance|brought\s+forward|"
    r"balance\s+brought\s+forward|closing\s+balance|eb-wert|eroeffnungsbilanz",
    re.I,
)
_STORNO_RX = re.compile(
    r"\bstorno\b|\breversal\b|\bcancel(?:lation)?\b|r[uü]ckbuchung|korrektur\s*storno",
    re.I,
)
_RECUR_TEXT = re.compile(
    r"miete|rent|leasing|lease|gehalt|lohn|salary|payroll|lohnlauf|"
    r"versicherung|insurance|pacht|abonnement|subscription|rate\b",
    re.I,
)


def _is_opening_text(s: str) -> bool:
    return bool(_OPENING_RX.search(s or ""))


def _postings_for_subject(dossier: Dossier, finding: Finding) -> list[Posting]:
    sid = finding.subject_id
    out: list[Posting] = []
    for p in dossier.postings:
        if finding.subject_kind == "entity" and p.entity_id == sid:
            out.append(p)
        elif finding.subject_kind == "transaction" and p.doc_no == sid:
            out.append(p)
    # also collect from flag evidence subjects
    if not out:
        for fl in finding.flags:
            if fl.entity_id:
                out.extend(p for p in dossier.postings if p.entity_id == fl.entity_id)
            if fl.doc_no:
                out.extend(p for p in dossier.postings if p.doc_no == fl.doc_no)
    # de-dupe by id
    seen: set[int] = set()
    uniq: list[Posting] = []
    for p in out:
        i = id(p)
        if i not in seen:
            seen.add(i)
            uniq.append(p)
    return uniq


def _shareholder_names(dossier: Dossier) -> set[str]:
    names: set[str] = set()
    ids: set[str] = set()
    for doc in dossier.docs_of("shareholder"):
        name = (doc.fields.get("NAME") or doc.ref or "").casefold().strip()
        if name:
            names.add(name)
        remark = doc.fields.get("BEMERKUNG") or doc.fields.get("REMARK") or ""
        for m in re.findall(r"\b(20\d{4}|10\d{4})\b", remark):
            ids.add(m)
        if doc.entity_id:
            ids.add(doc.entity_id)
    return names, ids


def check_opening_closing(dossier: Dossier, finding: Finding, posts: list[Posting]) -> str | None:
    # subject id looks like opening journal (AB-2024 style)
    sid = finding.subject_id or ""
    if re.match(r"^AB[-_]?\d{2,4}$", sid, re.I):
        return f"subject {sid!r} looks like opening-balance journal"
    for fl in finding.flags:
        if _is_opening_text(fl.title) or _is_opening_text(fl.rationale):
            return "flag text mentions opening/closing balance"
        for ev in fl.evidence:
            if _is_opening_text(ev.excerpt or ""):
                return f"evidence excerpt is opening/closing ({ev.file}:{ev.line})"
    # true carry-forward: Saldenvortrag text (BUCHUNGSTYP Vortrag alone is not enough)
    n_open = sum(1 for p in posts if _is_opening_text(p.text))
    if posts and n_open / len(posts) >= 0.5:
        return f"{n_open}/{len(posts)} postings are Saldenvortrag/opening balance"
    if n_open and finding.subject_kind == "transaction" and n_open == len(posts):
        return f"transaction is entirely opening-balance ({n_open} rows)"
    return None


def check_storno(posts: list[Posting]) -> str | None:
    """Storno/reversal only when text says so, or same account + opposite sign + storno-ish.

    Invoice + payment at opposite signs is normal AP/AR and must not hit.
    """
    if len(posts) < 2:
        return None
    # explicit storno language
    storno_posts = [p for p in posts if _STORNO_RX.search(p.text or "")]
    if storno_posts:
        p = storno_posts[0]
        return f"storno/reversal wording in text {p.text!r} ({p.doc_no})"

    by_acct_amt: dict[tuple[str, Decimal], list[Posting]] = defaultdict(list)
    for p in posts:
        by_acct_amt[(p.account, abs(p.amount))].append(p)
    for (acct, amt), group in by_acct_amt.items():
        if amt < Decimal("1000") or len(group) < 2:
            continue
        pos = [p for p in group if p.amount > 0]
        neg = [p for p in group if p.amount < 0]
        if not pos or not neg:
            continue
        for a in pos:
            for b in neg:
                delta = abs((a.booking_date - b.booking_date).days)
                if delta > 7:
                    continue
                # same doc_no often means invoice+clearing; skip different business texts
                ta, tb = (a.text or "").casefold(), (b.text or "").casefold()
                if "storno" in ta or "storno" in tb or "reversal" in ta or "reversal" in tb:
                    return (
                        f"storno pair account {acct} amount {amt:,.2f} "
                        f"{a.booking_date}/{b.booking_date}"
                    )
                # near-identical text with opposite sign = true reversal
                if ta and tb and ta == tb:
                    return (
                        f"mirror-text reversal account {acct} amount {amt:,.2f} "
                        f"{a.booking_date}/{b.booking_date}"
                    )
    return None


def check_intercompany(
    dossier: Dossier, finding: Finding, posts: list[Posting], sh_names: set[str], sh_ids: set[str]
) -> str | None:
    candidates: list[str] = []
    if finding.subject_kind == "entity":
        candidates.append(finding.subject_id)
    for fl in finding.flags:
        if fl.entity_id:
            candidates.append(fl.entity_id)
    for eid in candidates:
        if eid in sh_ids:
            return f"entity {eid} listed as shareholder/IC account"
        ent = dossier.entities.get(eid)
        if ent and ent.name and ent.name.casefold().strip() in sh_names:
            return f"entity {eid} name matches Gesellschafterliste"
        if ent and ent.name:
            en = ent.name.casefold()
            for sn in sh_names:
                if len(sn) >= 8 and (sn in en or en in sn):
                    return f"entity {eid} name ~ shareholder {sn!r}"
    # IC marker in attrs
    for p in posts[:50]:
        for v in (p.attrs or {}).values():
            if v and re.search(r"\bIC\b|intercompany|verbund|konzern", v, re.I):
                return "posting attrs mark intercompany/IC"
    return None


def check_recurring(posts: list[Posting]) -> str | None:
    if len(posts) < 4:
        return None
    # same abs amount in 3+ distinct months
    by_amt_months: dict[Decimal, set[tuple[int, int]]] = defaultdict(set)
    by_amt_text: dict[Decimal, list[str]] = defaultdict(list)
    for p in posts:
        a = abs(p.amount)
        if a < Decimal("500"):
            continue
        by_amt_months[a].add((p.booking_date.year, p.booking_date.month))
        if p.text:
            by_amt_text[a].append(p.text)
    for amt, months in by_amt_months.items():
        if len(months) < 3:
            continue
        texts = " ".join(by_amt_text.get(amt, [])[:10])
        if _RECUR_TEXT.search(texts):
            return (
                f"recurring amount {amt:,.2f} in {len(months)} months "
                f"with rent/payroll/lease-like text"
            )
        # even without keyword: very regular monthly same amount
        if len(months) >= 6:
            return f"same amount {amt:,.2f} in {len(months)} distinct months (rent-like cadence)"
    return None


def analyse(dossier: Dossier, finding: Finding, sh_names: set[str], sh_ids: set[str]) -> list[str]:
    posts = _postings_for_subject(dossier, finding)
    reasons: list[str] = []
    for check in (
        lambda: check_opening_closing(dossier, finding, posts),
        lambda: check_storno(posts),
        lambda: check_intercompany(dossier, finding, posts, sh_names, sh_ids),
        lambda: check_recurring(posts),
    ):
        r = check()
        if r:
            reasons.append(r)
    return reasons


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    path = argv[0] if argv else "data/practice"
    print(f"loading and scoring {path} ...")
    dossier, flags, findings = run(path)
    targets = [f for f in findings if f.tier in (Tier.HIGH, Tier.MEDIUM)]
    print(f"flags={len(flags)} findings={len(findings)} high/medium={len(targets)}")
    sh_names, sh_ids = _shareholder_names(dossier)

    rows: list[tuple] = []
    for f in targets:
        reasons = analyse(dossier, f, sh_names, sh_ids)
        suspect = "YES" if reasons else "no"
        reason = "; ".join(reasons) if reasons else "-"
        name = ""
        if f.subject_kind == "entity" and f.subject_id in dossier.entities:
            name = dossier.entities[f.subject_id].name
        rows.append(
            (
                f.subject_id,
                name[:28],
                f.tier.value,
                f"{f.score:.2f}",
                len(f.families),
                f"{f.max_amount:,.2f}" if f.max_amount else "-",
                suspect,
                reason,
            )
        )

    # table
    headers = ("subject", "name", "tier", "score", "fam", "amount", "innocent?", "reason")
    widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    # cap reason width
    widths[-1] = min(widths[-1], 72)
    def fmt(row):
        cells = []
        for i, w in enumerate(widths):
            s = str(row[i])
            if i == len(widths) - 1 and len(s) > w:
                s = s[: w - 1] + "…"
            cells.append(s.ljust(w) if i < 2 or i == len(widths) - 1 else s.rjust(w) if i in (3, 4) else s.ljust(w))
        return "  ".join(cells)

    print()
    print(fmt(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))

    n_yes = sum(1 for r in rows if r[6] == "YES")
    print()
    print(f"SUMMARY: {n_yes}/{len(rows)} HIGH/MEDIUM findings have an innocent-explanation hit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""LLM defense pass: argue innocence for REVIEW-tier findings.

Runs after scoring, only on Tier.REVIEW. Each finding goes to gpt-4o-mini
with its subject, every flag title/rationale/amount and the verbatim
evidence excerpts. The model must either exonerate with a reason grounded
in the quoted evidence or state why the defense fails. Exonerated ->
DISMISSED, otherwise -> MEDIUM. The reason lands in finding.defense_note
either way, so the report can show why something was dropped.

Grounding is enforced locally: an exoneration whose reason quotes no
number, date or name from the evidence is rejected and the finding is
promoted. Verdicts are disk-cached by content hash. Without an API key
the pass leaves tiers untouched and notes the skip.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from .contracts import Dossier, Finding, Tier

__all__ = ["run_defense"]

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
_CACHE_DIR = Path.home() / ".cache" / "laundromat"

_MAX_FLAGS = 25  # REVIEW findings are small; cap the payload anyway
_MAX_REFS = 6
_MAX_EXCERPT = 300

_SYSTEM = (
    "You are defense counsel in a financial audit. You receive one finding: "
    "a subject, the red flags raised against it, and verbatim excerpts from "
    "the source records. Argue the strongest honest case that this is an "
    "innocent, ordinary business event. Use only the quoted evidence; invent "
    "nothing. Exonerate only when the excerpts themselves (posting text, "
    "dates, document type, amounts) supply the innocent explanation, and "
    "your reason must address every flag; one flag the evidence cannot "
    "explain means the defense fails. Appeals to unquoted context or mere "
    "plausibility are not a defense. The excerpts cut both ways: when the "
    "posting text itself names a routine accounting event consistent with "
    "the dates and amount (an opening balance carry-forward on January 1, a "
    "monthly rent or payroll run), that is a complete defense. Otherwise "
    "say plainly why the defense fails. Reply as JSON: "
    '{"exonerate": true|false, "reason": "..."}. The reason must quote at '
    "least one concrete number, date or name copied from the evidence; a "
    "generic 'could be legitimate' is not a defense. Write the reason in "
    "German when the evidence is mostly German, otherwise in English."
)

# Generic audit vocabulary that must not count as quoting a name.
_STOP = {
    "buchung", "buchungen", "beleg", "belege", "zahlung", "zahlungen",
    "rechnung", "rechnungen", "kreditor", "kreditoren", "debitor",
    "debitoren", "lieferant", "lieferanten", "kunde", "kunden", "konto",
    "konten", "betrag", "betraege", "datum", "wareneingang", "freigabe",
    "journal", "sachkonto", "stammdaten", "vendor", "supplier", "customer",
    "invoice", "payment", "amount", "account", "posting", "entry",
    "booking", "document", "receipt", "evidence", "finding", "defense",
    "ledger", "balance", "approval",
}

_YEAR_RX = re.compile(r"^(19|20)\d{2}$")


def _digit_runs(text: str) -> set[str]:
    """Digit sequences, raw and with in-number separators collapsed.

    "25.000,00" and Decimal("25000.00") both yield "2500000"; the date
    18.03.2025 yields both its parts and "18032025". Bare years are
    dropped: "im Jahr 2025" is not a quote.
    """
    collapsed = re.sub(r"(?<=\d)[.,\s](?=\d)", "", text)
    runs = set(re.findall(r"\d{3,}", text)) | set(re.findall(r"\d{3,}", collapsed))
    return {r for r in runs if not _YEAR_RX.match(r)}


def _grounded(reason: str, corpus: str) -> bool:
    """True if the reason quotes a number, date or name from the evidence."""
    if _digit_runs(reason) & _digit_runs(corpus):
        return True
    tokens = {t.casefold() for t in re.findall(r"\w{4,}", corpus)}
    for tok in re.findall(r"\b[A-ZÄÖÜ]\w{3,}\b", reason):
        t = tok.casefold()
        if t not in _STOP and t in tokens:
            return True
    return False


def _corpus(finding: Finding, dossier: Dossier) -> str:
    parts = [finding.subject_id]
    ent = dossier.entities.get(finding.subject_id)
    if ent:
        parts.append(ent.name)
        if ent.address:
            parts.append(ent.address)
    for fl in finding.flags:
        if fl.amount is not None:
            parts.append(str(fl.amount))
        if fl.doc_no:
            parts.append(fl.doc_no)
        for ref in fl.evidence:
            parts.append(ref.cite())
            parts.append(ref.excerpt)
    return " ".join(p for p in parts if p)


def _payload(finding: Finding, dossier: Dossier) -> dict:
    subject: dict = {"id": finding.subject_id, "kind": finding.subject_kind}
    ent = dossier.entities.get(finding.subject_id)
    if ent:
        subject["name"] = ent.name
        subject["type"] = ent.type.value
        if ent.address:
            subject["address"] = ent.address
    flags = []
    for fl in finding.flags[:_MAX_FLAGS]:
        flags.append(
            {
                "lens": fl.lens_id,
                "title": fl.title,
                "rationale": fl.rationale,
                "amount": str(fl.amount) if fl.amount is not None else None,
                "evidence": [
                    {"source": ref.cite(), "excerpt": ref.excerpt[:_MAX_EXCERPT]}
                    for ref in fl.evidence[:_MAX_REFS]
                ],
            }
        )
    payload = {"subject": subject, "flags": flags}
    if len(finding.flags) > _MAX_FLAGS:
        payload["omitted_flags"] = len(finding.flags) - _MAX_FLAGS
    return payload


def _have_key() -> bool:
    try:
        if _ENV_FILE.exists():
            from dotenv import load_dotenv

            load_dotenv(_ENV_FILE)
    except Exception:
        pass
    return bool(os.environ.get("OPENAI_API_KEY"))


def _read_cache(cache: Path) -> dict | None:
    try:
        if cache.exists():
            raw = json.loads(cache.read_text("utf-8"))
            if (
                isinstance(raw, dict)
                and isinstance(raw.get("exonerate"), bool)
                and isinstance(raw.get("reason"), str)
                and raw["reason"].strip()
            ):
                return raw
    except Exception:
        pass
    return None


def _llm_verdict(payload_json: str) -> dict | None:
    try:
        from openai import OpenAI

        resp = OpenAI().chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": payload_json},
            ],
        )
        data = json.loads(resp.choices[0].message.content or "")
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    exon = data.get("exonerate")
    reason = data.get("reason")
    if not isinstance(exon, bool) or not isinstance(reason, str) or not reason.strip():
        return None
    return {"exonerate": exon, "reason": reason.strip()}


def _defend(finding: Finding, dossier: Dossier, have_key: bool) -> None:
    if not finding.flags:
        finding.defense_note = "defense pass skipped: finding has no flags"
        return
    payload_json = json.dumps(_payload(finding, dossier), ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256((_SYSTEM + payload_json).encode("utf-8")).hexdigest()
    cache = _CACHE_DIR / f"defense_{digest[:16]}.json"
    verdict = _read_cache(cache)
    if verdict is None:
        if not have_key:
            finding.defense_note = "defense pass skipped: no api key"
            return
        verdict = _llm_verdict(payload_json)
        if verdict is None:
            finding.defense_note = "defense pass skipped: llm call failed"
            return
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(verdict, ensure_ascii=False), "utf-8")
        except Exception:
            pass
    reason = verdict["reason"]
    if not verdict["exonerate"]:
        finding.tier = Tier.MEDIUM
        finding.defense_note = reason
    elif _grounded(reason, _corpus(finding, dossier)):
        finding.tier = Tier.DISMISSED
        finding.defense_note = reason
    else:
        finding.tier = Tier.MEDIUM
        finding.defense_note = (
            "Entlastung verworfen, Begruendung ohne Bezug zur Evidenz: " + reason
        )


def run_defense(findings: list[Finding], dossier: Dossier) -> None:
    """Adjudicate every Tier.REVIEW finding in place. Never raises."""
    review = [f for f in findings if f.tier is Tier.REVIEW]
    if not review:
        return
    have_key = _have_key()
    for finding in review:
        try:
            _defend(finding, dossier, have_key)
        except Exception:
            pass  # a broken defense leaves the finding in REVIEW

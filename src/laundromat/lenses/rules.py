"""Deterministic audit rule lenses."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from decimal import Decimal

from ..contracts import Dossier, Flag, JET_FLOOR, LensFamily, Posting, SourceRef, register


def _norm(value: str | None) -> str:
    text = unicodedata.normalize("NFKD", value or "").casefold()
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _is_gl(posting: Posting) -> bool:
    ledger = _norm(posting.attrs.get("ledger"))
    return not ledger or ledger in {"gl", "general ledger", "hauptbuch"}


def _is_opening_or_reversal(posting: Posting) -> bool:
    values = " ".join(
        [posting.doc_no, posting.text, *(str(value) for value in posting.attrs.values())]
    )
    text = _norm(values)
    return any(
        marker in text
        for marker in (
            "opening balance",
            "balance brought forward",
            "carry forward",
            "eroffnungsbilanz",
            "eroffnungssaldo",
            "saldenvortrag",
            "anfangsbestand",
            "storno",
            "reversal",
            "reversed",
            "cancellation",
        )
    )


def _evidence(postings: list[Posting]) -> tuple[SourceRef, ...]:
    seen: set[tuple[str, int | None, int | None, str | None]] = set()
    evidence: list[SourceRef] = []
    for posting in postings:
        source = posting.source
        key = (source.file, source.line, source.page, source.sheet)
        if key not in seen:
            seen.add(key)
            evidence.append(source)
    return tuple(evidence)


@register
class RoundAmount:
    """K6: material, conspicuously round general-ledger transactions."""

    # Practice set: 13 flags across 26,647 postings (0.049%).
    lens_id = "K6_round_amount"
    family = LensFamily.RULE

    @staticmethod
    def run(dossier: Dossier):
        groups: dict[tuple[str, object, Decimal], list[Posting]] = defaultdict(list)
        for posting in dossier.postings:
            amount = abs(posting.amount)
            if not _is_gl(posting) or posting.currency.upper() != "EUR":
                continue
            if _is_opening_or_reversal(posting) or amount <= JET_FLOOR:
                continue
            if amount % Decimal("1000"):
                continue
            key = (posting.doc_no, posting.booking_date, amount)
            groups[key].append(posting)

        for (_, _, amount), postings in groups.items():
            first = postings[0]
            yield Flag(
                lens_id=RoundAmount.lens_id,
                family=RoundAmount.family,
                title=f"Runder Betrag über JET-Grenze: {amount:.2f} {first.currency}",
                rationale=(
                    "Der Hauptbuchbetrag liegt über der JET-Grenze und ist auf volle "
                    "Tausend gerundet. Runde Beträge sind als regelbasierter Prüfhinweis "
                    "zu würdigen."
                ),
                evidence=_evidence(postings),
                entity_id=first.entity_id,
                doc_no=first.doc_no or None,
                amount=amount,
                confidence=0.45,
            )

from __future__ import annotations

import unittest
from datetime import date
from decimal import Decimal

from laundromat.contracts import Dossier, LensFamily, Posting, SourceRef
from laundromat.lenses.rules import RoundAmount


def source(line: int, excerpt: str) -> SourceRef:
    return SourceRef(file="Sachkonten/Sachkontobuchungen.txt", line=line, excerpt=excerpt)


def posting(
    amount: str,
    *,
    line: int,
    doc_no: str,
    text: str = "",
    currency: str = "EUR",
) -> Posting:
    return Posting(
        doc_no=doc_no,
        booking_date=date(2025, 6, 1),
        amount=Decimal(amount),
        account="440000",
        source=source(line, f"{doc_no};{amount};{text}"),
        text=text,
        currency=currency,
        attrs={"ledger": "GL"},
    )


class RoundAmountTests(unittest.TestCase):
    def test_flags_round_material_amount_once_for_mirrored_lines(self):
        dossier = Dossier(
            name="bad",
            postings=[
                posting("50000", line=2, doc_no="ER-1"),
                posting("-50000", line=3, doc_no="ER-1"),
            ],
        )

        flags = list(RoundAmount.run(dossier))

        self.assertEqual(len(flags), 1)
        flag = flags[0]
        self.assertEqual(flag.lens_id, "K6_round_amount")
        self.assertEqual(flag.family, LensFamily.RULE)
        self.assertEqual(flag.amount, Decimal("50000"))
        self.assertEqual(flag.doc_no, "ER-1")
        self.assertEqual(len(flag.evidence), 2)
        self.assertTrue(all(ref.line and ref.excerpt for ref in flag.evidence))

    def test_ignores_nonround_floor_opening_and_non_eur_rows(self):
        dossier = Dossier(
            name="good",
            postings=[
                posting("25000", line=2, doc_no="AT-FLOOR"),
                posting("50123.45", line=3, doc_no="NOT-ROUND"),
                posting("50000", line=4, doc_no="AB-2024", text="Opening balance"),
                posting("50000", line=5, doc_no="USD-1", currency="USD"),
            ],
        )

        self.assertEqual(list(RoundAmount.run(dossier)), [])

    def test_empty_dossier_is_safe(self):
        self.assertEqual(list(RoundAmount.run(Dossier(name="empty"))), [])


if __name__ == "__main__":
    unittest.main()

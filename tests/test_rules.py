from __future__ import annotations

import unittest
from datetime import date
from decimal import Decimal

from laundromat.contracts import Dossier, Document, Entity, EntityType, LensFamily, Posting, SourceRef
from laundromat.lenses.rules import NewVendorQuickPayment, RoundAmount, SplitPayments


def source(line: int, excerpt: str) -> SourceRef:
    return SourceRef(file="Sachkonten/Sachkontobuchungen.txt", line=line, excerpt=excerpt)


def posting(
    amount: str,
    *,
    line: int,
    doc_no: str,
    text: str = "",
    currency: str = "EUR",
    booking_date: date = date(2025, 6, 1),
    entity_id: str | None = None,
    ledger: str = "GL",
) -> Posting:
    return Posting(
        doc_no=doc_no,
        booking_date=booking_date,
        amount=Decimal(amount),
        account="440000",
        source=source(line, f"{doc_no};{amount};{text}"),
        entity_id=entity_id,
        text=text,
        currency=currency,
        attrs={"ledger": ledger},
    )


def vendor(vendor_id: str, *, line: int, created_at: date | None = None) -> Entity:
    return Entity(
        id=vendor_id,
        type=EntityType.VENDOR,
        name=f"Vendor {vendor_id}",
        source=SourceRef(
            file="Kreditoren/Lieferanten.txt",
            line=line,
            excerpt=f"{vendor_id};Vendor {vendor_id}",
        ),
        created_at=created_at,
    )


class NewVendorQuickPaymentTests(unittest.TestCase):
    def test_flags_material_payment_within_seven_days_of_first_appearance(self):
        dossier = Dossier(
            name="bad",
            entities={"OLD": vendor("OLD", line=2), "NEW": vendor("NEW", line=3)},
            postings=[
                posting(
                    "-100",
                    line=2,
                    doc_no="OLD-1",
                    text="Eingangsrechnung",
                    booking_date=date(2025, 1, 1),
                    entity_id="OLD",
                    ledger="AP",
                ),
                posting(
                    "-53550",
                    line=3,
                    doc_no="ER-NEW",
                    text="Beratungsrechnung",
                    booking_date=date(2025, 5, 19),
                    entity_id="NEW",
                    ledger="AP",
                ),
                posting(
                    "53550",
                    line=4,
                    doc_no="ER-NEW",
                    text="Zahlungsausgang Beratung",
                    booking_date=date(2025, 5, 21),
                    entity_id="NEW",
                    ledger="AP",
                ),
            ],
        )

        flags = list(NewVendorQuickPayment.run(dossier))

        self.assertEqual(len(flags), 1)
        flag = flags[0]
        self.assertEqual(flag.lens_id, "K1_new_vendor_quick_payment")
        self.assertEqual(flag.entity_id, "NEW")
        self.assertEqual(flag.doc_no, "ER-NEW")
        self.assertEqual(flag.amount, Decimal("53550"))
        self.assertEqual(len(flag.evidence), 2)
        self.assertTrue(all(ref.line and ref.excerpt for ref in flag.evidence))

    def test_ignores_slow_low_value_early_cohort_and_non_ap_payments(self):
        entities = {
            key: vendor(key, line=index)
            for index, key in enumerate(("BASE", "SLOW", "SMALL", "EARLY", "GL"), 2)
        }
        dossier = Dossier(
            name="good",
            entities=entities,
            postings=[
                posting(
                    "-100",
                    line=2,
                    doc_no="BASE-1",
                    text="Invoice",
                    booking_date=date(2025, 1, 1),
                    entity_id="BASE",
                    ledger="AP",
                ),
                posting(
                    "-40000",
                    line=3,
                    doc_no="SLOW-1",
                    text="Invoice",
                    booking_date=date(2025, 5, 1),
                    entity_id="SLOW",
                    ledger="AP",
                ),
                posting(
                    "40000",
                    line=4,
                    doc_no="SLOW-1",
                    text="Payment",
                    booking_date=date(2025, 5, 9),
                    entity_id="SLOW",
                    ledger="AP",
                ),
                posting(
                    "-20000",
                    line=5,
                    doc_no="SMALL-1",
                    text="Invoice",
                    booking_date=date(2025, 5, 1),
                    entity_id="SMALL",
                    ledger="AP",
                ),
                posting(
                    "20000",
                    line=6,
                    doc_no="SMALL-1",
                    text="Wire payment",
                    booking_date=date(2025, 5, 2),
                    entity_id="SMALL",
                    ledger="AP",
                ),
                posting(
                    "-40000",
                    line=7,
                    doc_no="EARLY-1",
                    text="Invoice",
                    booking_date=date(2025, 1, 2),
                    entity_id="EARLY",
                    ledger="AP",
                ),
                posting(
                    "40000",
                    line=8,
                    doc_no="EARLY-1",
                    text="Payment",
                    booking_date=date(2025, 1, 3),
                    entity_id="EARLY",
                    ledger="AP",
                ),
                posting(
                    "40000",
                    line=9,
                    doc_no="GL-1",
                    text="Payment",
                    booking_date=date(2025, 5, 2),
                    entity_id="GL",
                    ledger="GL",
                ),
            ],
        )

        self.assertEqual(list(NewVendorQuickPayment.run(dossier)), [])

    def test_flags_unregistered_vendor_only_when_master_is_available(self):
        suspicious = posting(
            "5000",
            line=4,
            doc_no="PAY-X",
            text="Partial payment",
            booking_date=date(2025, 6, 1),
            entity_id="MISSING",
            ledger="AP",
        )
        with_master = Dossier(
            name="with-master",
            entities={"KNOWN": vendor("KNOWN", line=2)},
            postings=[suspicious],
        )
        without_master = Dossier(name="without-master", postings=[suspicious])

        flags = list(NewVendorQuickPayment.run(with_master))

        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0].entity_id, "MISSING")
        self.assertEqual(flags[0].confidence, 0.9)
        self.assertEqual(list(NewVendorQuickPayment.run(without_master)), [])

    def test_supports_english_vendor_field_on_purchase_invoice(self):
        document = Document(
            kind="purchase_invoice",
            ref="INV-X",
            source=SourceRef(
                file="support/vendor_invoices.csv",
                line=2,
                excerpt="INV-X;MISSING;38000.00",
            ),
            doc_date=date(2025, 6, 1),
            amount=Decimal("38000"),
            fields={"VENDOR_ID": "MISSING", "INVOICE_DATE": "2025-06-01"},
        )
        dossier = Dossier(
            name="invoice",
            entities={"KNOWN": vendor("KNOWN", line=2)},
            documents=[document],
        )

        flags = list(NewVendorQuickPayment.run(dossier))

        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0].entity_id, "MISSING")
        self.assertEqual(flags[0].doc_no, "INV-X")
        self.assertEqual(flags[0].evidence, (document.source,))

    def test_empty_dossier_is_safe(self):
        self.assertEqual(list(NewVendorQuickPayment.run(Dossier(name="empty"))), [])


class SplitPaymentsTests(unittest.TestCase):
    def test_flags_one_cluster_and_cites_every_payment(self):
        dossier = Dossier(
            name="bad",
            entities={"V1": vendor("V1", line=2)},
            postings=[
                posting(
                    amount,
                    line=line,
                    doc_no="BATCH-V1",
                    text="Teilzahlung Lieferantenrechnung",
                    booking_date=date(2025, 10, 14),
                    entity_id="V1",
                    ledger="AP",
                )
                for line, amount in enumerate(("9780", "9820", "9750", "9690"), 10)
            ],
        )

        flags = list(SplitPayments.run(dossier))

        self.assertEqual(len(flags), 1)
        flag = flags[0]
        self.assertEqual(flag.lens_id, "K5_split_payments")
        self.assertEqual(flag.entity_id, "V1")
        self.assertEqual(flag.doc_no, "BATCH-V1")
        self.assertEqual(flag.amount, Decimal("39040"))
        self.assertEqual(len(flag.evidence), 4)
        self.assertTrue(all(ref.line and ref.excerpt for ref in flag.evidence))

    def test_ignores_invoice_pairs_distant_rows_limit_and_foreign_currency(self):
        dossier = Dossier(
            name="good",
            entities={"V1": vendor("V1", line=2)},
            postings=[
                posting(
                    "9500",
                    line=10,
                    doc_no="INV-1",
                    text="Purchase invoice",
                    booking_date=date(2025, 5, 1),
                    entity_id="V1",
                    ledger="AP",
                ),
                posting(
                    "9500",
                    line=11,
                    doc_no="PAY-1",
                    text="Payment",
                    booking_date=date(2025, 5, 1),
                    entity_id="V1",
                    ledger="AP",
                ),
                posting(
                    "9500",
                    line=12,
                    doc_no="PAY-2",
                    text="Payment",
                    booking_date=date(2025, 5, 10),
                    entity_id="V1",
                    ledger="AP",
                ),
                posting(
                    "10000",
                    line=13,
                    doc_no="AT-LIMIT",
                    text="Payment",
                    booking_date=date(2025, 5, 1),
                    entity_id="V1",
                    ledger="AP",
                ),
                posting(
                    "9500",
                    line=14,
                    doc_no="USD-1",
                    text="Payment",
                    currency="USD",
                    booking_date=date(2025, 5, 1),
                    entity_id="V1",
                    ledger="AP",
                ),
            ],
        )

        self.assertEqual(list(SplitPayments.run(dossier)), [])

    def test_empty_dossier_is_safe(self):
        self.assertEqual(list(SplitPayments.run(Dossier(name="empty"))), [])


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

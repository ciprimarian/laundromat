from __future__ import annotations

import unittest
from datetime import date, datetime
from decimal import Decimal

from laundromat.contracts import Dossier, Document, Entity, EntityType, LensFamily, Posting, SourceRef
from laundromat.lenses.rules import (
    Backdating,
    CutoffViolation,
    NewVendorQuickPayment,
    NoGoodsReceipt,
    OddHoursAdmin,
    RepairCapitalized,
    RoundAmount,
    SplitPayments,
)


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
    posted_at: datetime | None = None,
    user: str | None = None,
    account: str = "440000",
    attrs: dict[str, str] | None = None,
) -> Posting:
    posting_attrs = {"ledger": ledger}
    posting_attrs.update(attrs or {})
    return Posting(
        doc_no=doc_no,
        booking_date=booking_date,
        amount=Decimal(amount),
        account=account,
        source=source(line, f"{doc_no};{amount};{text}"),
        posted_at=posted_at,
        entity_id=entity_id,
        user=user,
        text=text,
        currency=currency,
        attrs=posting_attrs,
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


def account(account_id: str, name: str, *, line: int) -> Entity:
    return Entity(
        id=account_id,
        type=EntityType.ACCOUNT,
        name=name,
        source=SourceRef(
            file="Sachkonten/Sachkonten.txt",
            line=line,
            excerpt=f"{account_id};{name};Bilanz",
        ),
        attrs={"KONTENART": "Bilanz"},
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


class NoGoodsReceiptTests(unittest.TestCase):
    def goods_postings(self, vendor_id: str = "V1", doc_no: str = "INV-1") -> list[Posting]:
        return [
            posting(
                "30000",
                line=10,
                doc_no=doc_no,
                text="Zahlungsausgang",
                booking_date=date(2025, 6, 10),
                entity_id=vendor_id,
                ledger="AP",
                account=vendor_id,
            ),
            posting(
                "25210.08",
                line=11,
                doc_no=doc_no,
                text="Eingangsrechnung Material",
                booking_date=date(2025, 6, 8),
                account="MAT",
            ),
        ]

    def receipt(
        self,
        reference: str | None,
        vendor_id: str,
        amount: str,
        receipt_date: date,
    ) -> Document:
        fields = {"KREDITOR": vendor_id}
        if reference is not None:
            fields["RECHNUNGSNUMMER"] = reference
        return Document(
            kind="goods_receipt",
            ref="GR-1",
            source=SourceRef(
                file="support/goods_receipts.csv",
                line=2,
                excerpt=f"GR-1;{reference or ''};{vendor_id};{amount}",
            ),
            entity_id=vendor_id,
            doc_date=receipt_date,
            amount=Decimal(amount),
            fields=fields,
        )

    def test_exact_invoice_and_vendor_match_is_silent(self):
        dossier = Dossier(
            name="matched",
            entities={
                "V1": vendor("V1", line=2),
                "MAT": account("MAT", "Roh-, Hilfs- und Betriebsstoffe", line=3),
            },
            postings=self.goods_postings(),
            documents=[self.receipt("INV-1", "V1", "30000", date(2025, 6, 8))],
        )

        self.assertEqual(list(NoGoodsReceipt.run(dossier)), [])

    def test_flags_missing_receipt_and_cites_payment_and_goods_row(self):
        dossier = Dossier(
            name="missing",
            entities={
                "V1": vendor("V1", line=2),
                "V2": vendor("V2", line=3),
                "MAT": account("MAT", "Inventory raw materials", line=4),
            },
            postings=self.goods_postings(),
            documents=[self.receipt("OTHER", "V2", "30000", date(2025, 6, 8))],
        )

        flags = list(NoGoodsReceipt.run(dossier))

        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0].lens_id, "K2_no_goods_receipt")
        self.assertEqual(flags[0].entity_id, "V1")
        self.assertEqual(flags[0].doc_no, "INV-1")
        self.assertEqual(flags[0].amount, Decimal("30000"))
        self.assertEqual(len(flags[0].evidence), 2)

    def test_service_exclusion_overrides_material_word(self):
        dossier = Dossier(
            name="service",
            entities={
                "V1": vendor("V1", line=2),
                "SERV": account("SERV", "Material consulting service expense", line=3),
            },
            postings=[
                self.goods_postings()[0],
                posting(
                    "25210.08",
                    line=11,
                    doc_no="INV-1",
                    text="Material consulting service",
                    account="SERV",
                ),
            ],
            documents=[self.receipt("OTHER", "V1", "100", date(2025, 6, 8))],
        )

        self.assertEqual(list(NoGoodsReceipt.run(dossier)), [])

    def test_fallback_matches_same_vendor_amount_within_thirty_days(self):
        dossier = Dossier(
            name="fallback",
            entities={
                "V1": vendor("V1", line=2),
                "MAT": account("MAT", "Inventory", line=3),
            },
            postings=self.goods_postings(),
            documents=[self.receipt(None, "V1", "30000", date(2025, 5, 12))],
        )

        self.assertEqual(list(NoGoodsReceipt.run(dossier)), [])

    def test_empty_or_malformed_receipts_are_safe(self):
        malformed = Document(
            kind="goods_receipt",
            ref="GR-BAD",
            source=SourceRef(
                file="support/goods_receipts.csv",
                line=2,
                excerpt="GR-BAD;;;;",
            ),
            fields={"RECHNUNGSNUMMER": ""},
        )
        dossier = Dossier(
            name="malformed",
            postings=self.goods_postings(),
            documents=[malformed],
        )

        self.assertEqual(list(NoGoodsReceipt.run(dossier)), [])
        self.assertEqual(list(NoGoodsReceipt.run(Dossier(name="empty"))), [])


class RepairCapitalizedTests(unittest.TestCase):
    def test_flags_direct_repair_text_on_prefixed_asset_account(self):
        dossier = Dossier(
            name="direct",
            entities={"FA": account("FA", "Fixed asset machinery", line=2)},
            postings=[
                posting(
                    "-27500",
                    line=10,
                    doc_no="INV-DIRECT",
                    text="Machine repair and servicing",
                    account="FA-001",
                )
            ],
        )

        flags = list(RepairCapitalized.run(dossier))

        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0].doc_no, "INV-DIRECT")
        self.assertEqual(flags[0].amount, Decimal("27500"))
        self.assertEqual(flags[0].evidence, (dossier.postings[0].source,))

    def test_groups_asset_addition_and_german_repair_sibling_across_ledgers(self):
        dossier = Dossier(
            name="bad",
            entities={"A100": account("A100", "Maschinen und maschinelle Anlagen", line=2)},
            postings=[
                posting(
                    "28000",
                    line=10,
                    doc_no="ER-1",
                    text="Acquisition",
                    ledger="FA",
                    account="A100-0001",
                    attrs={"account_base": "A100", "BUCHUNGSART": "Zugang"},
                ),
                posting(
                    "5320",
                    line=11,
                    doc_no="ER-1",
                    text="Reparatur Konfektioniermaschine Linie 2",
                    ledger="GL",
                    account="VAT",
                ),
            ],
        )

        flags = list(RepairCapitalized.run(dossier))

        self.assertEqual(len(flags), 1)
        flag = flags[0]
        self.assertEqual(flag.lens_id, "K3_repair_capitalized")
        self.assertEqual(flag.doc_no, "ER-1")
        self.assertEqual(flag.amount, Decimal("28000"))
        self.assertEqual(len(flag.evidence), 2)
        self.assertTrue(all(ref.line and ref.excerpt for ref in flag.evidence))

    def test_supports_english_replacement_and_overhaul_terms(self):
        dossier = Dossier(
            name="english",
            entities={"FA": account("FA", "Property, plant and equipment", line=2)},
            postings=[
                posting(
                    "41000",
                    line=10,
                    doc_no="INV-1",
                    text="Asset addition",
                    account="FA-9",
                    attrs={"account_base": "FA"},
                ),
                posting(
                    "41000",
                    line=11,
                    doc_no="INV-1",
                    text="Hydraulic unit replacement and overhaul",
                    account="PAYABLE",
                ),
            ],
        )

        flags = list(RepairCapitalized.run(dossier))

        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0].doc_no, "INV-1")

    def test_ignores_expense_investment_inventory_depreciation_and_opening(self):
        entities = {
            "FA": account("FA", "Fixed asset machinery", line=2),
            "INV": account("INV", "Inventory and merchandise", line=3),
        }
        dossier = Dossier(
            name="good",
            entities=entities,
            postings=[
                posting(
                    "30000",
                    line=10,
                    doc_no="EXPENSE",
                    text="Repair expense",
                    account="EXP",
                ),
                posting(
                    "30000",
                    line=11,
                    doc_no="INVEST",
                    text="Acquisition",
                    account="FA-1",
                    attrs={"account_base": "FA"},
                ),
                posting(
                    "30000",
                    line=12,
                    doc_no="INVEST",
                    text="New production line investment",
                    account="PAYABLE",
                ),
                posting(
                    "30000",
                    line=13,
                    doc_no="INVENTORY",
                    text="Acquisition",
                    account="INV-1",
                    attrs={"account_base": "INV"},
                ),
                posting(
                    "30000",
                    line=14,
                    doc_no="INVENTORY",
                    text="Warehouse service",
                    account="PAYABLE",
                ),
                posting(
                    "30000",
                    line=15,
                    doc_no="DEPR",
                    text="Depreciation",
                    account="FA-2",
                    attrs={"account_base": "FA"},
                ),
                posting(
                    "30000",
                    line=16,
                    doc_no="DEPR",
                    text="Machine maintenance",
                    account="PAYABLE",
                ),
                posting(
                    "30000",
                    line=17,
                    doc_no="AB-2024",
                    text="Opening balance acquisition",
                    account="FA-3",
                    attrs={"account_base": "FA"},
                ),
                posting(
                    "30000",
                    line=18,
                    doc_no="AB-2024",
                    text="Repair",
                    account="PAYABLE",
                ),
            ],
        )

        self.assertEqual(list(RepairCapitalized.run(dossier)), [])

    def test_empty_dossier_is_safe(self):
        self.assertEqual(list(RepairCapitalized.run(Dossier(name="empty"))), [])


class CutoffViolationTests(unittest.TestCase):
    def test_derives_year_and_flags_english_purchase_invoice_shift(self):
        invoice = Document(
            kind="purchase_invoice",
            ref="INV-2032-1",
            source=SourceRef(
                file="support/vendor_invoices.csv",
                line=2,
                excerpt="INV-2032-1;2032-01-12;2031-12-20;V1;22000",
            ),
            entity_id="V1",
            doc_date=date(2032, 1, 12),
            amount=Decimal("22000"),
            fields={"INVOICE_DATE": "2032-01-12", "SERVICE_DATE": "2031-12-20"},
        )
        dossier = Dossier(
            name="shift",
            postings=[
                posting(
                    "100",
                    line=2,
                    doc_no="BASE",
                    booking_date=date(2031, 6, 1),
                )
            ],
            documents=[invoice],
        )

        flags = list(CutoffViolation.run(dossier))

        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0].doc_no, "INV-2032-1")
        self.assertEqual(flags[0].entity_id, "V1")
        self.assertEqual(flags[0].amount, Decimal("22000"))
        self.assertEqual(flags[0].evidence, (invoice.source,))

    def test_flags_gl_document_date_straddle(self):
        crossing = posting(
            "31000",
            line=3,
            doc_no="YEAR-END",
            booking_date=date(2032, 1, 4),
            attrs={"BELEGDATUM": "28.12.2031"},
        )
        dossier = Dossier(
            name="gl-shift",
            postings=[
                posting(
                    "100",
                    line=2,
                    doc_no="BASE",
                    booking_date=date(2031, 6, 1),
                ),
                crossing,
            ],
        )

        flags = list(CutoffViolation.run(dossier))

        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0].doc_no, "YEAR-END")
        self.assertEqual(flags[0].evidence, (crossing.source,))

    def test_ignores_accrued_same_year_same_year_invoice_and_next_period_cash(self):
        accrued = Document(
            kind="purchase_invoice",
            ref="ACCRUED-1",
            source=SourceRef(file="support/invoices.csv", line=2, excerpt="ACCRUED-1"),
            doc_date=date(2032, 1, 5),
            fields={"LEISTUNGSDATUM": "20.12.2031"},
        )
        same_year = Document(
            kind="purchase_invoice",
            ref="SAME-1",
            source=SourceRef(file="support/invoices.csv", line=3, excerpt="SAME-1"),
            doc_date=date(2031, 12, 22),
            fields={"LEISTUNGSDATUM": "20.12.2031"},
        )
        cash = Document(
            kind="next_period_posting",
            ref="CASH-1",
            source=SourceRef(file="support/next_period.csv", line=2, excerpt="CASH-1;payment"),
            doc_date=date(2032, 1, 6),
            fields={"BUCHUNGSTEXT": "Payment receipt / settlement"},
        )
        document_date_only = Document(
            kind="purchase_invoice",
            ref="NO-SERVICE-DATE",
            source=SourceRef(file="support/invoices.csv", line=4, excerpt="NO-SERVICE-DATE"),
            doc_date=date(2032, 1, 5),
            fields={"BELEGDATUM": "20.12.2031"},
        )
        dossier = Dossier(
            name="good",
            postings=[
                posting(
                    "100",
                    line=2,
                    doc_no="BASE",
                    booking_date=date(2031, 1, 1),
                ),
                posting(
                    "50000",
                    line=3,
                    doc_no="ACCRUED-1",
                    booking_date=date(2031, 12, 31),
                ),
                posting(
                    "50000",
                    line=4,
                    doc_no="ACCRUED-1",
                    booking_date=date(2032, 1, 5),
                    attrs={"BELEGDATUM": "20.12.2031"},
                ),
            ],
            documents=[accrued, same_year, cash, document_date_only],
        )

        self.assertEqual(list(CutoffViolation.run(dossier)), [])

    def test_empty_and_malformed_dossiers_are_safe(self):
        malformed = Document(
            kind="purchase_invoice",
            ref="BAD-DATE",
            source=SourceRef(file="support/invoices.csv", line=2, excerpt="BAD-DATE;oops"),
            fields={"INVOICE_DATE": "not-a-date", "SERVICE_DATE": "also-bad"},
        )
        self.assertEqual(list(CutoffViolation.run(Dossier(name="empty"))), [])
        self.assertEqual(
            list(
                CutoffViolation.run(
                    Dossier(
                        name="malformed",
                        postings=[
                            posting(
                                "100",
                                line=2,
                                doc_no="BASE",
                                booking_date=date(2031, 1, 1),
                            )
                        ],
                        documents=[malformed],
                    )
                )
            ),
            [],
        )


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


class OddHoursAdminTests(unittest.TestCase):
    def test_flags_material_admin_transaction_once_for_balanced_lines(self):
        entered = datetime(2025, 6, 2, 12, 30)
        dossier = Dossier(
            name="admin",
            postings=[
                posting(
                    "500000",
                    line=10,
                    doc_no="ADMIN-1",
                    text="Fertigung Umlagerung",
                    posted_at=entered,
                    user="Admin",
                ),
                posting(
                    "-500000",
                    line=11,
                    doc_no="ADMIN-1",
                    text="Fertigung Umlagerung",
                    posted_at=entered,
                    user="Admin",
                ),
                posting(
                    "500000",
                    line=12,
                    doc_no="USER-1",
                    posted_at=entered,
                    user="MV-U01",
                ),
                posting(
                    "100",
                    line=13,
                    doc_no="ADMIN-ROUTINE",
                    posted_at=entered,
                    user="Admin",
                ),
            ],
        )

        flags = list(OddHoursAdmin.run(dossier))

        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0].lens_id, "K7_odd_hours_admin")
        self.assertEqual(flags[0].doc_no, "ADMIN-1")
        self.assertEqual(flags[0].amount, Decimal("500000"))
        self.assertEqual(len(flags[0].evidence), 2)
        self.assertEqual(flags[0].confidence, 0.3)

    def test_derives_outward_rounded_business_hours(self):
        workday = date(2025, 6, 2)
        baseline = [
            posting(
                "100",
                line=100 + index,
                doc_no=f"BASE-{index}",
                booking_date=workday,
                posted_at=datetime(2025, 6, 2, 8 if index < 100 else 18),
                user="MV-U01",
            )
            for index in range(200)
        ]
        night = posting(
            "30000",
            line=400,
            doc_no="NIGHT-1",
            booking_date=workday,
            posted_at=datetime(2025, 6, 2, 23, 30),
            user="MV-U02",
        )

        flags = list(OddHoursAdmin.run(Dossier(name="night", postings=baseline + [night])))

        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0].doc_no, "NIGHT-1")
        self.assertEqual(flags[0].evidence, (night.source,))

    def test_suppresses_common_weekends_and_opening_at_night(self):
        weekday = [
            posting(
                "100",
                line=500 + index,
                doc_no=f"WEEKDAY-{index}",
                booking_date=date(2025, 6, 2),
                posted_at=datetime(2025, 6, 2, 12),
                user="MV-U01",
            )
            for index in range(20)
        ]
        weekend = [
            posting(
                "30000",
                line=600 + index,
                doc_no=f"WEEKEND-{index}",
                booking_date=date(2025, 6, 7),
                posted_at=datetime(2025, 6, 7, 12),
                user="MV-U01",
            )
            for index in range(20)
        ]
        opening = posting(
            "500000",
            line=700,
            doc_no="AB-2024",
            text="Opening balance",
            booking_date=date(2025, 1, 1),
            posted_at=datetime(2025, 1, 1, 23, 30),
            user="Admin",
        )

        flags = list(
            OddHoursAdmin.run(
                Dossier(name="weekends", postings=weekday + weekend + [opening])
            )
        )

        self.assertEqual(flags, [])

    def test_empty_dossier_is_safe(self):
        self.assertEqual(list(OddHoursAdmin.run(Dossier(name="empty"))), [])


class BackdatingTests(unittest.TestCase):
    def test_uses_strict_empirical_tail_and_groups_balanced_lines(self):
        booking = date(2025, 1, 1)
        baseline = [
            posting(
                "100",
                line=1000 + index,
                doc_no=f"NORMAL-{index}",
                booking_date=booking,
                posted_at=datetime(2025, 1, 1, 10),
                user="MV-U01",
            )
            for index in range(1000)
        ]
        boundary = [
            posting(
                "30000",
                line=2100,
                doc_no="LAG-7",
                booking_date=booking,
                posted_at=datetime(2025, 1, 8, 10),
                user="MV-U02",
            ),
            posting(
                "30000",
                line=2101,
                doc_no="LAG-8",
                booking_date=booking,
                posted_at=datetime(2025, 1, 9, 10),
                user="MV-U02",
            ),
        ]
        bad = [
            posting(
                "86500",
                line=2102,
                doc_no="LAG-15",
                booking_date=booking,
                posted_at=datetime(2025, 1, 16, 10),
                user="MV-U02",
            ),
            posting(
                "-86500",
                line=2103,
                doc_no="LAG-15",
                booking_date=booking,
                posted_at=datetime(2025, 1, 16, 10),
                user="MV-U02",
            ),
        ]

        flags = list(
            Backdating.run(Dossier(name="tail", postings=baseline + boundary + bad))
        )

        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0].lens_id, "backdating_entry_lag")
        self.assertEqual(flags[0].doc_no, "LAG-15")
        self.assertEqual(flags[0].amount, Decimal("86500"))
        self.assertEqual(len(flags[0].evidence), 2)

    def test_requires_explicit_lock_date(self):
        booked = date(2025, 1, 31)
        entered = datetime(2025, 2, 2, 9)
        locked = posting(
            "30000",
            line=2200,
            doc_no="LOCKED",
            booking_date=booked,
            posted_at=entered,
            attrs={"PERIOD_LOCK_DATE": "2025-01-31"},
        )
        status_only = posting(
            "30000",
            line=2201,
            doc_no="STATUS-ONLY",
            booking_date=booked,
            posted_at=entered,
            attrs={"FESTSCHREIBUNG": "Ja"},
        )

        flags = list(Backdating.run(Dossier(name="locks", postings=[locked, status_only])))

        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0].doc_no, "LOCKED")
        self.assertIn("Sperrdatum", flags[0].rationale)

    def test_ignores_opening_entries_and_low_value_tail(self):
        dossier = Dossier(
            name="safe",
            postings=[
                posting(
                    "500000",
                    line=2300,
                    doc_no="AB-2024",
                    text="Saldenvortrag",
                    booking_date=date(2025, 1, 1),
                    posted_at=datetime(2025, 1, 20, 9),
                ),
                posting(
                    "1000",
                    line=2301,
                    doc_no="LOW",
                    booking_date=date(2025, 1, 1),
                    posted_at=datetime(2025, 1, 20, 9),
                ),
            ],
        )

        self.assertEqual(list(Backdating.run(dossier)), [])

    def test_empty_dossier_is_safe(self):
        self.assertEqual(list(Backdating.run(Dossier(name="empty"))), [])


if __name__ == "__main__":
    unittest.main()

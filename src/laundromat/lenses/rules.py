"""Deterministic audit rule lenses."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal

from ..contracts import (
    Dossier,
    Document,
    EntityType,
    Flag,
    JET_FLOOR,
    LensFamily,
    Posting,
    SourceRef,
    register,
)


def _norm(value: str | None) -> str:
    text = unicodedata.normalize("NFKD", value or "").casefold()
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _is_gl(posting: Posting) -> bool:
    ledger = _norm(posting.attrs.get("ledger"))
    return not ledger or ledger in {"gl", "general ledger", "hauptbuch"}


def _is_ap(posting: Posting) -> bool:
    ledger = _norm(posting.attrs.get("ledger"))
    return not ledger or ledger in {
        "ap",
        "accounts payable",
        "kreditoren",
        "kreditorenbuch",
    }


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


def _source_evidence(*sources: SourceRef) -> tuple[SourceRef, ...]:
    seen: set[tuple[str, int | None, int | None, str | None]] = set()
    evidence: list[SourceRef] = []
    for source in sources:
        key = (source.file, source.line, source.page, source.sheet)
        if key not in seen:
            seen.add(key)
            evidence.append(source)
    return tuple(evidence)


def _norm_id(value: str | None) -> str:
    return re.sub(r"\s+", "", (value or "").strip().casefold())


def _field(document: Document, *aliases: str) -> str | None:
    fields = {_norm(key): str(value).strip() for key, value in document.fields.items()}
    for alias in aliases:
        value = fields.get(_norm(alias))
        if value:
            return value
    return None


def _parse_date(value: str | date | datetime | None) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = (value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        pass
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


_PAYMENT_MARKERS = (
    "zahlungsausgang",
    "teilzahlung",
    "uberweisung",
    "ueberweisung",
    "ausgleich",
    "bezahlt",
    "lastschrift",
    "payment",
    "partial payment",
    "wire transfer",
    "bank transfer",
    "settlement",
    "paid",
    "disbursement",
)


def _is_vendor_payment(posting: Posting) -> bool:
    if not _is_ap(posting) or posting.amount <= 0 or _is_opening_or_reversal(posting):
        return False
    text = _norm(
        " ".join(
            (
                posting.text,
                posting.attrs.get("BUCHUNGSTEXT", ""),
                posting.attrs.get("DESCRIPTION", ""),
                posting.attrs.get("MEMO", ""),
                posting.attrs.get("BUCHUNGSART", ""),
                posting.attrs.get("TRANSACTION_TYPE", ""),
            )
        )
    )
    return any(marker in text for marker in _PAYMENT_MARKERS)


def _is_purchase_invoice(document: Document) -> bool:
    kind = _norm(document.kind)
    if kind in {"purchase invoice", "vendor invoice", "creditor invoice"}:
        return True
    if kind != "invoice":
        return False
    return bool(
        _field(
            document,
            "KREDITOR",
            "LIEFERANT",
            "VENDOR",
            "SUPPLIER",
            "CREDITOR",
        )
    )


def _document_vendor_id(document: Document) -> str | None:
    return document.entity_id or _field(
        document,
        "KREDITOR",
        "KREDITORENNUMMER",
        "LIEFERANT",
        "LIEFERANTENKONTONUMMER",
        "VENDOR",
        "VENDOR_ID",
        "SUPPLIER",
        "SUPPLIER_ID",
        "CREDITOR",
        "CREDITOR_ID",
    )


def _document_date(document: Document) -> date | None:
    return document.doc_date or _parse_date(
        _field(
            document,
            "FAKTURADATUM",
            "RECHNUNGSDATUM",
            "BELEGDATUM",
            "INVOICE_DATE",
            "DOCUMENT_DATE",
            "DOC_DATE",
        )
    )


@register
class NewVendorQuickPayment:
    """K1: a new or unregistered vendor paid unusually quickly."""

    # Practice set: 1 flag across 26,647 postings (0.004%).
    lens_id = "K1_new_vendor_quick_payment"
    family = LensFamily.RULE
    quick_days = 7
    observation_days = 90

    @staticmethod
    def run(dossier: Dossier):
        vendor_entities = {
            _norm_id(entity.id): entity
            for entity in dossier.entities.values()
            if entity.type is EntityType.VENDOR
        }
        master_available = bool(vendor_entities)

        postings: dict[str, list[Posting]] = defaultdict(list)
        original_ids: dict[str, str] = {}
        for posting in dossier.postings:
            if not _is_ap(posting) or not posting.entity_id:
                continue
            if _is_opening_or_reversal(posting):
                continue
            vendor_id = _norm_id(posting.entity_id)
            entity = dossier.entities.get(posting.entity_id)
            if entity is not None and entity.type is not EntityType.VENDOR:
                continue
            postings[vendor_id].append(posting)
            original_ids.setdefault(vendor_id, posting.entity_id)

        invoice_docs: dict[str, list[Document]] = defaultdict(list)
        for document in dossier.documents:
            if not _is_purchase_invoice(document):
                continue
            raw_id = _document_vendor_id(document)
            if not raw_id:
                continue
            vendor_id = _norm_id(raw_id)
            entity = dossier.entities.get(raw_id)
            if entity is not None and entity.type is not EntityType.VENDOR:
                continue
            invoice_docs[vendor_id].append(document)
            original_ids.setdefault(vendor_id, raw_id)

        flagged: set[str] = set()
        if master_available:
            observed_ids = set(postings) | set(invoice_docs)
            for vendor_id in sorted(observed_ids - set(vendor_entities)):
                payment_rows = [row for row in postings[vendor_id] if _is_vendor_payment(row)]
                if payment_rows:
                    payment = min(payment_rows, key=lambda row: (row.booking_date, row.source.line or 0))
                    source = payment.source
                    doc_no = payment.doc_no or None
                    amount = abs(payment.amount)
                    rationale = (
                        f"Die Zahlung an Kreditor {original_ids[vendor_id]} erscheint im "
                        "Kreditorenbuch, der Kreditor fehlt jedoch im Kreditorenstamm."
                    )
                elif invoice_docs[vendor_id]:
                    document = min(
                        invoice_docs[vendor_id],
                        key=lambda item: (_document_date(item) or date.max, item.source.line or 0),
                    )
                    source = document.source
                    doc_no = document.ref or None
                    amount = abs(document.amount) if document.amount is not None else None
                    rationale = (
                        f"Kreditor {original_ids[vendor_id]} erscheint im Rechnungsjournal, "
                        "fehlt jedoch im Kreditorenstamm."
                    )
                else:
                    continue
                flagged.add(vendor_id)
                yield Flag(
                    lens_id=NewVendorQuickPayment.lens_id,
                    family=NewVendorQuickPayment.family,
                    title=f"Nicht registrierter Kreditor {original_ids[vendor_id]}",
                    rationale=rationale,
                    evidence=(source,),
                    entity_id=original_ids[vendor_id],
                    doc_no=doc_no,
                    amount=amount,
                    confidence=0.9,
                )

        observation_dates = [
            posting.booking_date
            for posting in dossier.postings
            if not _is_opening_or_reversal(posting)
        ]
        observation_start = min(observation_dates, default=None)

        for vendor_id, entity in sorted(vendor_entities.items()):
            if vendor_id in flagged:
                continue
            vendor_postings = postings.get(vendor_id, [])
            payments = [
                posting
                for posting in vendor_postings
                if _is_vendor_payment(posting)
                and _norm(posting.currency) == "eur"
                and abs(posting.amount) >= JET_FLOOR
            ]
            if not payments:
                continue

            if entity.created_at is not None:
                first_date = entity.created_at
                first_source = entity.source
            else:
                appearances: list[tuple[date, SourceRef]] = [
                    (posting.booking_date, posting.source)
                    for posting in vendor_postings
                    if not _is_vendor_payment(posting)
                ]
                appearances.extend(
                    (doc_date, document.source)
                    for document in invoice_docs.get(vendor_id, [])
                    if (doc_date := _document_date(document)) is not None
                )
                if not appearances:
                    continue
                first_date, first_source = min(
                    appearances,
                    key=lambda item: (item[0], item[1].line or 0),
                )
                if (
                    observation_start is None
                    or (first_date - observation_start).days
                    < NewVendorQuickPayment.observation_days
                ):
                    continue

            qualifying = [
                payment
                for payment in payments
                if 0
                <= (payment.booking_date - first_date).days
                <= NewVendorQuickPayment.quick_days
            ]
            if not qualifying:
                continue
            payment = min(qualifying, key=lambda row: (row.booking_date, row.source.line or 0))
            lag = (payment.booking_date - first_date).days
            amount = abs(payment.amount)
            yield Flag(
                lens_id=NewVendorQuickPayment.lens_id,
                family=NewVendorQuickPayment.family,
                title=f"Neuer Kreditor {entity.id} nach {lag} Tagen bezahlt",
                rationale=(
                    f"Kreditor {entity.id} erscheint erstmals am {first_date.isoformat()} "
                    f"und erhält bereits {lag} Tage später eine Zahlung über "
                    f"{amount:.2f} {payment.currency}."
                ),
                evidence=_source_evidence(first_source, payment.source),
                entity_id=entity.id,
                doc_no=payment.doc_no or None,
                amount=amount,
                confidence=0.78,
            )


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

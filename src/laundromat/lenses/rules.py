"""Deterministic audit rule lenses."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime
from decimal import Decimal

from ..contracts import (
    APPROVAL_LIMIT,
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
from ..ingest.accounts import AccountClass, classify_account


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


def _posting_field(posting: Posting, *aliases: str) -> str | None:
    fields = {_norm(key): str(value).strip() for key, value in posting.attrs.items()}
    for alias in aliases:
        value = fields.get(_norm(alias))
        if value:
            return value
    return None


def _reference(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", _norm(value))


def _document_reference(document: Document) -> str:
    return _reference(
        _field(
            document,
            "RECHNUNGSNUMMER",
            "INVOICE_NUMBER",
            "INVOICE_NO",
            "BELEGNUMMER",
            "DOCUMENT_NUMBER",
            "DOC_NO",
        )
        or document.ref
    )


def _fiscal_year(dossier: Dossier) -> int | None:
    gl_years = [posting.booking_date.year for posting in dossier.postings if _is_gl(posting)]
    years = gl_years or [posting.booking_date.year for posting in dossier.postings]
    if not years:
        return None
    counts = Counter(years)
    return min(counts, key=lambda year: (-counts[year], year))


def _performance_date(document: Document) -> date | None:
    return _parse_date(
        _field(
            document,
            "LEISTUNGSDATUM",
            "SERVICE_DATE",
            "PERFORMANCE_DATE",
            "DELIVERY_DATE",
            "LIEFERDATUM",
            "BELEGDATUM",
            "DOCUMENT_DATE",
        )
    )


def _looks_like_payment(document: Document) -> bool:
    text = _norm(" ".join(str(value) for value in document.fields.values()))
    return any(marker in text for marker in _PAYMENT_MARKERS)


_FIXED_ASSET_MARKERS = (
    "anlagevermogen",
    "sachanlage",
    "grundstuck",
    "gebaude",
    "maschine",
    "maschinelle anlage",
    "anlagen im bau",
    "betriebsausstattung",
    "geschaftsausstattung",
    "edv hardware",
    "fixed asset",
    "property plant equipment",
    "property",
    "plant",
    "equipment",
    "machinery",
    "building",
    "vehicle",
    "construction in progress",
)

_ADDITION_MARKERS = (
    "zugang",
    "anlagezugang",
    "anschaffung",
    "aktivierung",
    "erwerb",
    "addition",
    "acquisition",
    "capitalization",
    "capitalisation",
)

_DISPOSAL_MARKERS = (
    "abgang",
    "verausserung",
    "abschreibung",
    "afa",
    "disposal",
    "depreciation",
    "amortization",
    "amortisation",
    "retirement",
    "write off",
)

_REPAIR_RE = re.compile(
    r"\b(?:reparatur|wartung|instandhaltung|instandsetzung|ersatzteile?|service|"
    r"uberholung|ueberholung|generaluberholung|generalueberholung|austausch|repairs?|"
    r"maintenance|servicing|refurbish(?:ment|ed|ing)?|overhauls?|spare parts?|"
    r"replacements?)\b"
)


def _posting_text(posting: Posting) -> str:
    return _norm(
        " ".join(
            [posting.text, *(str(value) for value in posting.attrs.values() if value)]
        )
    )


def _fixed_asset_account(posting: Posting, dossier: Dossier) -> bool:
    account_id = posting.attrs.get("account_base") or posting.account
    entity = dossier.entities.get(account_id)
    if entity is None:
        candidates = [
            candidate
            for candidate in dossier.entities.values()
            if candidate.type in {EntityType.ACCOUNT, EntityType.ASSET}
            and posting.account.startswith(candidate.id)
            and posting.account[len(candidate.id) : len(candidate.id) + 1]
            in {"-", "/", ".", " "}
        ]
        entity = max(candidates, key=lambda candidate: len(candidate.id), default=None)
    if entity is None:
        return False
    if entity.type is EntityType.ASSET:
        return True
    if entity.type is not EntityType.ACCOUNT:
        return False
    account_text = _norm(
        " ".join([entity.name, *(str(value) for value in entity.attrs.values() if value)])
    )
    if not any(marker in account_text for marker in _FIXED_ASSET_MARKERS):
        return False
    return classify_account(entity.id, entity.name, entity.attrs) is AccountClass.ASSET


def _fixed_asset_addition(posting: Posting, dossier: Dossier) -> bool:
    if posting.amount == 0 or _is_opening_or_reversal(posting):
        return False
    text = _posting_text(posting)
    if any(marker in text for marker in _DISPOSAL_MARKERS):
        return False
    return _fixed_asset_account(posting, dossier) and (
        any(marker in text for marker in _ADDITION_MARKERS)
        or bool(_REPAIR_RE.search(text))
    )


def _repair_narrative(posting: Posting) -> bool:
    return bool(_REPAIR_RE.search(_posting_text(posting)))


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
class RepairCapitalized:
    """K3: repair work recorded as a fixed-asset addition."""

    # Practice set: 6 flags across 26,647 postings (0.023%).
    lens_id = "K3_repair_capitalized"
    family = LensFamily.RULE

    @staticmethod
    def run(dossier: Dossier):
        groups: dict[str, list[Posting]] = defaultdict(list)
        for posting in dossier.postings:
            reference = posting.doc_no.strip().casefold()
            if reference:
                groups[reference].append(posting)

        for postings in groups.values():
            additions = [
                posting
                for posting in postings
                if _fixed_asset_addition(posting, dossier)
            ]
            narratives = [
                posting
                for posting in postings
                if not _is_opening_or_reversal(posting) and _repair_narrative(posting)
            ]
            if not additions or not narratives:
                continue

            first = additions[0]
            amount = max(abs(posting.amount) for posting in additions)
            vendor_ids = {
                posting.entity_id
                for posting in postings
                if posting.entity_id
                and (entity := dossier.entities.get(posting.entity_id)) is not None
                and entity.type is EntityType.VENDOR
            }
            yield Flag(
                lens_id=RepairCapitalized.lens_id,
                family=RepairCapitalized.family,
                title=f"Reparatur als Anlagezugang gebucht: {first.doc_no}",
                rationale=(
                    f"Beleg {first.doc_no} enthält einen Anlagezugang über {amount:.2f} "
                    f"{first.currency}, während die Belegtexte Reparatur, Wartung oder "
                    "Austausch beschreiben."
                ),
                evidence=_evidence(additions + narratives),
                entity_id=next(iter(vendor_ids)) if len(vendor_ids) == 1 else None,
                doc_no=first.doc_no,
                amount=amount,
                confidence=0.82,
            )


@register
class CutoffViolation:
    """K4: documents and postings straddling the derived fiscal boundary."""

    # Practice set: 8 flags across 26,647 postings (0.030%).
    lens_id = "K4_cutoff_violation"
    family = LensFamily.RULE

    @staticmethod
    def run(dossier: Dossier):
        fiscal_year = _fiscal_year(dossier)
        if fiscal_year is None:
            return

        fiscal_references: set[str] = set()
        for posting in dossier.postings:
            if posting.booking_date.year != fiscal_year or _is_opening_or_reversal(posting):
                continue
            for value in (
                posting.doc_no,
                _posting_field(
                    posting,
                    "BELEGNUMMER",
                    "RECHNUNGSNUMMER",
                    "INVOICE_NUMBER",
                    "DOCUMENT_NUMBER",
                ),
            ):
                if reference := _reference(value):
                    fiscal_references.add(reference)

        flagged_references: set[str] = set()
        for document in dossier.documents:
            kind = _norm(document.kind)
            if not _is_purchase_invoice(document) and kind != "next period posting":
                continue
            if kind == "next period posting" and _looks_like_payment(document):
                continue

            invoice_date = (
                _parse_date(
                    _field(
                        document,
                        "FAKTURADATUM",
                        "RECHNUNGSDATUM",
                        "INVOICE_DATE",
                        "BOOKING_DATE",
                        "BUCHUNGSDATUM",
                    )
                )
                or document.doc_date
            )
            performance_date = _performance_date(document)
            if invoice_date is None or performance_date is None:
                continue
            if (
                performance_date.year != fiscal_year
                or invoice_date.year != fiscal_year + 1
                or performance_date > invoice_date
                or (invoice_date - performance_date).days > 90
            ):
                continue

            reference = _document_reference(document)
            if reference and reference in fiscal_references:
                continue
            flagged_references.add(reference)
            amount = abs(document.amount) if document.amount is not None else None
            yield Flag(
                lens_id=CutoffViolation.lens_id,
                family=CutoffViolation.family,
                title=f"Periodenverschiebung bei Beleg {document.ref}",
                rationale=(
                    f"Das Leistungsdatum {performance_date.isoformat()} liegt im Geschäftsjahr "
                    f"{fiscal_year}, die Rechnung wurde jedoch erst am "
                    f"{invoice_date.isoformat()} im Folgejahr erfasst. Eine passende "
                    "Buchung oder Abgrenzung im Geschäftsjahr ist nicht vorhanden."
                ),
                evidence=(document.source,),
                entity_id=_document_vendor_id(document),
                doc_no=document.ref or None,
                amount=amount,
                confidence=0.9,
            )

        for posting in dossier.postings:
            if not _is_gl(posting) or _is_opening_or_reversal(posting):
                continue
            document_date = _parse_date(
                _posting_field(
                    posting,
                    "BELEGDATUM",
                    "DOCUMENT_DATE",
                    "DOC_DATE",
                    "INVOICE_DATE",
                    "LEISTUNGSDATUM",
                    "SERVICE_DATE",
                )
            )
            if document_date is None:
                continue
            years = {posting.booking_date.year, document_date.year}
            if (
                abs(posting.booking_date.year - document_date.year) != 1
                or fiscal_year not in years
                or abs((posting.booking_date - document_date).days) > 45
            ):
                continue
            reference = _reference(posting.doc_no)
            if reference and reference in flagged_references:
                continue
            flagged_references.add(reference)
            amount = abs(posting.amount)
            yield Flag(
                lens_id=CutoffViolation.lens_id,
                family=CutoffViolation.family,
                title=f"Buchung und Beleg in verschiedenen Perioden: {posting.doc_no}",
                rationale=(
                    f"Belegdatum {document_date.isoformat()} und Buchungsdatum "
                    f"{posting.booking_date.isoformat()} liegen auf unterschiedlichen Seiten "
                    "der Geschäftsjahresgrenze."
                ),
                evidence=(posting.source,),
                entity_id=posting.entity_id,
                doc_no=posting.doc_no or None,
                amount=amount,
                confidence=0.86,
            )


@register
class SplitPayments:
    """K5: clustered vendor payments structured below an approval limit."""

    # Practice set: 1 flag across 26,647 postings (0.004%).
    lens_id = "K5_split_payments"
    family = LensFamily.RULE
    window_days = 3
    lower_fraction = Decimal("0.80")

    @staticmethod
    def run(dossier: Dossier):
        lower = APPROVAL_LIMIT * SplitPayments.lower_fraction
        grouped: dict[tuple[str, str], list[Posting]] = defaultdict(list)

        for posting in dossier.postings:
            if not posting.entity_id or not _is_vendor_payment(posting):
                continue
            entity = dossier.entities.get(posting.entity_id)
            if entity is not None and entity.type is not EntityType.VENDOR:
                continue
            amount = abs(posting.amount)
            if amount < lower or amount >= APPROVAL_LIMIT:
                continue
            currency = _norm(posting.currency)
            if currency != "eur":
                continue
            grouped[(_norm_id(posting.entity_id), currency)].append(posting)

        for payments in grouped.values():
            payments.sort(
                key=lambda posting: (
                    posting.booking_date,
                    posting.source.file,
                    posting.source.line or 0,
                )
            )
            start = 0
            while start < len(payments):
                end = start + 1
                while (
                    end < len(payments)
                    and (payments[end].booking_date - payments[start].booking_date).days
                    <= SplitPayments.window_days
                ):
                    end += 1

                cluster = payments[start:end]
                total = sum((abs(posting.amount) for posting in cluster), Decimal(0))
                if len(cluster) < 2 or total <= APPROVAL_LIMIT:
                    start += 1
                    continue

                first = cluster[0]
                references = {posting.doc_no for posting in cluster if posting.doc_no}
                shared_reference = (
                    next(iter(references))
                    if len(references) == 1 and all(posting.doc_no for posting in cluster)
                    else None
                )
                yield Flag(
                    lens_id=SplitPayments.lens_id,
                    family=SplitPayments.family,
                    title=(
                        f"{len(cluster)} Teilzahlungen unter der Freigabegrenze an "
                        f"Kreditor {first.entity_id}"
                    ),
                    rationale=(
                        f"Innerhalb von {(cluster[-1].booking_date - first.booking_date).days} "
                        f"Tagen wurden {len(cluster)} Zahlungen über zusammen {total:.2f} "
                        f"{first.currency} gebucht. Jede Einzelzahlung liegt zwischen 80 Prozent "
                        "und der Freigabegrenze, gemeinsam überschreiten sie die Grenze."
                    ),
                    evidence=_evidence(cluster),
                    entity_id=first.entity_id,
                    doc_no=shared_reference,
                    amount=total,
                    confidence=0.82,
                )
                start = end


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

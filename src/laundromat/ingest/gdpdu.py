"""GDPdU table reader driven by index.xml.

The XML declares column names and order only. Encoding, delimiters, decimal
and date formats fall back to the DTD defaults (latin-1, ";", '"', decimal
comma, DD.MM.YYYY) unless the XML overrides them. Tables are classified by
column signature, never by file name, so a foreign chart of accounts or an
English export still maps to the canonical types.
"""

from __future__ import annotations

import csv
import unicodedata
from dataclasses import replace
import xml.etree.ElementTree as ET
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from pathlib import Path

from ..contracts import Dossier, Entity, EntityType, Posting, SourceRef

_EXCERPT_LEN = 200

_DATE_FORMATS = ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d.%m.%y")
_TIME_FORMATS = ("%H:%M:%S", "%H:%M", "%H.%M.%S")


def _norm(s: str) -> str:
    """Casefold plus umlaut transliteration so DE/EN header sets match."""
    s = unicodedata.normalize("NFC", s or "").casefold().strip()
    for a, b in (("\xe4", "ae"), ("\xf6", "oe"), ("\xfc", "ue"), ("\xdf", "ss")):
        s = s.replace(a, b)
    return s


def _find(cols: list[str], *keys: str) -> str | None:
    """First column whose normalized name contains a key; keys are ordered by priority."""
    normed = [(_norm(c), c) for c in cols]
    for key in keys:
        for n, c in normed:
            if key in n:
                return c
    return None


def parse_amount(s: str | None) -> Decimal | None:
    s = (s or "").strip().replace("\xa0", "").replace(" ", "")
    if not s:
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        head, _, tail = s.rpartition(",")
        if len(tail) <= 2 and s.count(",") == 1:
            s = head + "." + tail
        else:
            s = s.replace(",", "")
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def parse_date(s: str | None) -> date | None:
    s = (s or "").strip()
    if not s:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def parse_time(s: str | None) -> time | None:
    s = (s or "").strip()
    if not s:
        return None
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(s, fmt).time()
        except ValueError:
            continue
    return None


class TableSpec:
    def __init__(self, url: str, name: str, columns: list[str], delim: str, quote: str, skip_to: int):
        self.url = url
        self.name = name
        self.columns = columns
        self.delim = delim
        self.quote = quote
        self.skip_to = skip_to  # 1-based first data line, from Range/From


def read_index(index_path: Path) -> list[TableSpec]:
    tree = ET.parse(index_path)
    tables = []
    for t in tree.getroot().iter("Table"):
        url = t.findtext("URL")
        if not url:
            continue
        name = t.findtext("Name") or url
        vl = t.find("VariableLength")
        if vl is None:
            raise ValueError(f"{name}: fixed-length tables not supported")
        cols = [
            c.findtext("Name") or ""
            for c in vl
            if c.tag in ("VariablePrimaryKey", "VariableColumn")
        ]
        delim = vl.findtext("ColumnDelimiter") or ";"
        quote = vl.findtext("TextEncapsulator") or '"'
        skip_to = 1
        rng = t.find("Range")
        if rng is not None:
            try:
                skip_to = int(rng.findtext("From") or 1)
            except ValueError:
                pass
        tables.append(TableSpec(url, name, cols, delim, quote, skip_to))
    return tables


def _iter_rows(path: Path, spec: TableSpec):
    """Yield (line_no, raw_line, fields). Quoted embedded newlines are joined."""
    with open(path, "rb") as fb:
        head = fb.read(3)
    enc = "utf-8-sig" if head == b"\xef\xbb\xbf" else "latin-1"
    pending, start = "", 0
    with open(path, encoding=enc, errors="replace", newline="") as f:
        for i, line in enumerate(f, 1):
            raw = line.rstrip("\r\n")
            if pending:
                pending += "\n" + raw
            else:
                pending, start = raw, i
            if pending.count(spec.quote) % 2 == 1:
                continue  # open quote, row continues on next line
            if start >= spec.skip_to and pending.strip():
                fields = next(csv.reader([pending], delimiter=spec.delim, quotechar=spec.quote))
                if len(fields) < len(spec.columns):
                    fields += [""] * (len(spec.columns) - len(fields))
                yield start, pending, fields
            pending = ""


# ---------------------------------------------------------------- classification

_VENDOR_KEYS = ("lieferant", "kreditor", "vendor", "supplier", "creditor")
_CUSTOMER_KEYS = ("kunde", "debitor", "customer", "debtor", "client")
_ASSET_KEYS = ("anlage", "asset")
_GL_KEYS = ("sachkonto", "hauptbuch", "general ledger", "gl account", "ledger")
_AMOUNT_KEYS = ("buchungsbetrag", "transaction amount", "amount", "betrag", "value")
_BOOKDATE_KEYS = ("buchungsdatum", "posting date", "booking date", "transaction date",
                  "wertstellung", "value date", "valuta")
_NAME_KEYS = ("name", "bezeichnung", "description")


def _classify(spec: TableSpec) -> str:
    cols = spec.columns
    amount = _find(cols, *_AMOUNT_KEYS)
    bookdate = _find(cols, *_BOOKDATE_KEYS, "datum", "date")
    vendor = _find(cols, *_VENDOR_KEYS)
    customer = _find(cols, *_CUSTOMER_KEYS)
    asset = _find(cols, *_ASSET_KEYS)
    if amount and bookdate:
        if vendor:
            return "ap_postings"
        if customer:
            return "ar_postings"
        if asset:
            return "fa_postings"
        return "gl_postings"
    if vendor and _find(cols, *_NAME_KEYS):
        return "vendors"
    if customer and _find(cols, *_NAME_KEYS):
        return "customers"
    if asset:
        return "assets"
    if _find(cols, *_GL_KEYS, "kontenart", "account"):
        return "accounts"
    return "unknown"


# ---------------------------------------------------------------- row mapping

def _postings_from_table(path: Path, rel: str, spec: TableSpec, kind: str, dossier: Dossier) -> int:
    cols = spec.columns
    c_amount = _find(cols, *_AMOUNT_KEYS)
    c_currency = _find(cols, "waehrung", "currency", "curr")
    c_bookdate = _find(cols, *_BOOKDATE_KEYS) or _find(cols, "datum", "date")
    c_docno = _find(cols, "belegnummer", "belegnr", "document no", "doc no", "voucher", "invoice no")
    c_bookno = _find(cols, "buchungsnummer", "journal no", "entry no")
    c_docref = _find(cols, "dokument")
    c_docdate = _find(cols, "belegdatum", "document date", "doc date", "invoice date")
    c_text = _find(cols, "buchungstext", "text", "narrative", "verwendungszweck", "memo", "description", "bezeichnung")
    c_user = _find(cols, "benutzerkennung", "benutzer", "erfasser", "created by", "entered by", "user")
    c_entrydate = _find(cols, "erfassungsdatum", "entry date", "created", "capture date")
    c_entrytime = _find(cols, "erfassungszeit", "entry time", "uhrzeit", "time")
    c_counter = _find(cols, "gegenkonto", "counter account", "offset account", "contra")
    c_account = _find(cols, "sachkontonummer", "sachkonto", "hauptbuchkonto", "gl account",
                      "ledger account", "account no", "account number", "kontonummer", "konto", "account")
    c_entity = {
        "ap_postings": _find(cols, *_VENDOR_KEYS),
        "ar_postings": _find(cols, *_CUSTOMER_KEYS),
        "fa_postings": _find(cols, *_ASSET_KEYS),
        "gl_postings": None,
    }[kind]
    ledger = {"gl_postings": "GL", "ap_postings": "AP", "ar_postings": "AR", "fa_postings": "FA"}[kind]

    loaded = dropped = 0
    for line_no, raw, fields in _iter_rows(path, spec):
        row = dict(zip(cols, fields))
        amount = parse_amount(row.get(c_amount)) if c_amount else None
        booking = parse_date(row.get(c_bookdate)) if c_bookdate else None
        if booking is None and c_docdate:
            booking = parse_date(row.get(c_docdate))
        if amount is None or booking is None:
            dropped += 1
            continue
        entity_id = row.get(c_entity, "").strip() if c_entity else None
        account = row.get(c_account, "").strip() if c_account else ""
        if not account and entity_id:
            account = entity_id  # subledger account number
        posted_at = None
        if c_entrydate:
            entry_d = parse_date(row.get(c_entrydate))
            if entry_d:
                entry_t = parse_time(row.get(c_entrytime)) if c_entrytime else None
                posted_at = datetime.combine(entry_d, entry_t or time(0, 0))
        attrs = dict(row)
        attrs["ledger"] = ledger
        attrs["table"] = spec.name
        doc_no = ""
        for c in (c_docno, c_bookno, c_docref):
            if c and row.get(c, "").strip():
                doc_no = row[c].strip()
                break
        dossier.postings.append(Posting(
            doc_no=doc_no,
            booking_date=booking,
            amount=amount,
            account=account,
            source=SourceRef(file=rel, line=line_no, excerpt=raw[:_EXCERPT_LEN]),
            posted_at=posted_at,
            counter_account=(row.get(c_counter, "").strip() or None) if c_counter else None,
            entity_id=entity_id or None,
            user=(row.get(c_user, "").strip() or None) if c_user else None,
            text=row.get(c_text, "").strip() if c_text else "",
            currency=(row.get(c_currency, "").strip() or "EUR") if c_currency else "EUR",
            attrs=attrs,
        ))
        loaded += 1
    if dropped:
        dossier.unparsed.append((rel, f"{dropped} rows dropped: no parseable amount or date"))
    return loaded


_ENTITY_TYPES = {
    "vendors": EntityType.VENDOR,
    "customers": EntityType.CUSTOMER,
    "assets": EntityType.ASSET,
    "accounts": EntityType.ACCOUNT,
}


def _entities_from_table(path: Path, rel: str, spec: TableSpec, kind: str, dossier: Dossier) -> int:
    cols = spec.columns
    etype = _ENTITY_TYPES[kind]
    id_keys = {
        "vendors": ("lieferantenkontonummer", *(k + "kontonummer" for k in _VENDOR_KEYS),
                    "vendor no", "supplier no", "creditor no", "kontonummer", "nummer", "number", "konto", "id"),
        "customers": ("kundenkontonummer", "customer no", "debtor no", "kontonummer", "nummer", "number", "konto", "id"),
        "assets": ("anlagennummer", "asset no", "nummer", "number", "id"),
        "accounts": ("sachkontonummer", "kontonummer", "account no", "account number", "nummer", "number", "account", "id"),
    }[kind]
    c_id = _find(cols, *id_keys) or cols[0]
    c_name = _find(cols, "sachkontoname", *(t + "name" for t in ("lieferanten", "kunden")),
                   "bezeichnung", "name", "description")
    c_street = _find(cols, "strasse", "street", "address")
    c_zip = _find(cols, "plz", "postal", "zip")
    c_city = _find(cols, "ort", "city", "town")
    c_country = _find(cols, "staat", "country", "land")
    c_iban = _find(cols, "iban", "bankkonto", "bank account")

    loaded = 0
    for line_no, raw, fields in _iter_rows(path, spec):
        row = dict(zip(cols, fields))
        eid = row.get(c_id, "").strip()
        if not eid:
            continue
        addr_parts = [row.get(c, "").strip() for c in (c_street, c_zip, c_city, c_country) if c]
        address = ", ".join(p for p in addr_parts if p) or None
        ent = Entity(
            id=eid,
            type=etype,
            name=(row.get(c_name, "").strip() if c_name else "") or eid,
            source=SourceRef(file=rel, line=line_no, excerpt=raw[:_EXCERPT_LEN]),
            address=address,
            iban=(row.get(c_iban, "").strip() or None) if c_iban else None,
            attrs=dict(row),
        )
        if eid in dossier.entities:
            eid = f"{etype.value}:{eid}"
            if eid in dossier.entities:
                continue
            ent = Entity(id=eid, type=ent.type, name=ent.name, source=ent.source,
                         address=ent.address, iban=ent.iban, attrs=ent.attrs)
        dossier.entities[eid] = ent
        loaded += 1
    return loaded


def _link_subaccounts(dossier: Dossier) -> None:
    """Resolve ACCOUNT-SUBACCOUNT notation (e.g. 332000-100097).

    The prefix rolls up to the GL account, the suffix often names the
    entity the row belongs to. Both are discovered, never assumed.
    """
    accounts = {e.id for e in dossier.entities.values() if e.type == EntityType.ACCOUNT}
    ids = set(dossier.entities)
    for i, p in enumerate(dossier.postings):
        if "-" not in p.account:
            continue
        base, _, suffix = p.account.partition("-")
        if base not in accounts:
            continue
        attrs = dict(p.attrs)
        attrs["account_base"] = base
        entity_id = p.entity_id or (suffix if suffix in ids else None)
        dossier.postings[i] = replace(p, entity_id=entity_id, attrs=attrs)


def load_gdpdu(root: Path, dossier: Dossier) -> None:
    """Read every GDPdU table set (a directory with an index.xml) under root."""
    for index_path in sorted(root.rglob("index.xml")):
        try:
            specs = read_index(index_path)
        except Exception as e:
            dossier.unparsed.append((str(index_path.relative_to(root)), f"index parse failed: {e}"))
            continue
        for spec in specs:
            data_path = index_path.parent / spec.url
            rel = str(data_path.relative_to(root))
            if not data_path.exists():
                dossier.unparsed.append((rel, "file listed in index.xml but missing"))
                continue
            kind = _classify(spec)
            try:
                if kind.endswith("_postings"):
                    _postings_from_table(data_path, rel, spec, kind, dossier)
                elif kind in _ENTITY_TYPES:
                    _entities_from_table(data_path, rel, spec, kind, dossier)
                else:
                    dossier.unparsed.append((rel, f"unrecognized table signature: {spec.columns[:6]}"))
            except Exception as e:
                dossier.unparsed.append((rel, f"read failed: {e}"))
    _link_subaccounts(dossier)

"""Account classification: chart-of-accounts numbers/names -> semantic classes.

Three layers, cheapest first:
  1. attrs type hints (SACHKONTOTYP / KONTENART / whatever the export calls them),
  2. bilingual DE+EN keyword rules on the account name,
  3. one batched OpenAI call for the leftovers, cached on disk.

Lenses ask for classes, never account numbers, so this module is the only
place that knows what an account name means.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from enum import Enum
from pathlib import Path

from ..contracts import Dossier, EntityType

__all__ = ["AccountClass", "classify_account", "classify_accounts"]


class AccountClass(str, Enum):
    ASSET = "asset"
    EXPENSE = "expense"
    REVENUE = "revenue"
    PAYABLE = "payable"
    RECEIVABLE = "receivable"
    BANK = "bank"
    EQUITY = "equity"
    TAX = "tax"
    OTHER = "other"


_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"
_CACHE_DIR = Path.home() / ".cache" / "laundromat"

_UMLAUTS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})


def _norm(s: str) -> str:
    return (s or "").lower().translate(_UMLAUTS)


# Statement-side hints found in attrs values. They never decide a class on
# their own; they constrain which name rules may fire.
_PL_RX = re.compile(
    r"\bguv\b|gewinn.{0,3}und.{0,3}verlust|profit\s*(and|&)\s*loss|\bp\s*&\s*l\b"
    r"|\bpnl\b|income statement|erfolgsrechnung|ergebnisrechnung"
)
_BS_RX = re.compile(r"\bbilanz\b|balance sheet|\bbalance\b|\baktiva\b|\bpassiva\b|\bbs\b")

# P&L accounts can only be flow classes; balance-sheet accounts only stock
# classes. TAX lives on both sides (Umsatzsteuer vs Steueraufwand).
_ALLOWED = {
    "pl": {AccountClass.EXPENSE, AccountClass.REVENUE, AccountClass.TAX},
    "bs": {
        AccountClass.ASSET,
        AccountClass.PAYABLE,
        AccountClass.RECEIVABLE,
        AccountClass.BANK,
        AccountClass.EQUITY,
        AccountClass.TAX,
    },
}

# Decisive attr-value hints, checked after side phrases are stripped so
# "income statement" cannot read as revenue.
_DIRECT_RULES = [
    (AccountClass.TAX, r"steuer|\btax\b|\bvat\b"),
    (AccountClass.EQUITY, r"eigenkapital|equity|\bcapital\b"),
    (AccountClass.BANK, r"bank|\bcash\b|kasse|liquide"),
    (AccountClass.PAYABLE, r"verbindlich|payable|liabilit|kreditor"),
    (AccountClass.RECEIVABLE, r"forderung|receivable|debitor"),
    (AccountClass.EXPENSE, r"aufwand|aufwendung|expense|kosten|\bcosts?\b"),
    (AccountClass.REVENUE, r"ertrag|ertraeg|erloes|umsatz|revenue|\bincome\b|\bsales\b"),
    (AccountClass.ASSET, r"anlagevermoegen|sachanlage|\bassets?\b|aktivkonto"),
]
_DIRECT = [(cls, re.compile(rx)) for cls, rx in _DIRECT_RULES]

# Ordered name rules, first compatible match wins. Strong markers (compound
# suffixes like -aufwand/-ertrag, statutory terms) before weak vocabulary,
# so "Zinsaufwand" is expense and "Zinsertrag" revenue. Dual listings
# (fuhrpark, abschreibung) resolve via the statement-side filter.
_NAME_RULE_SPECS = [
    (AccountClass.RECEIVABLE, r"^debitorisch"),
    (AccountClass.PAYABLE, r"^kreditorisch"),
    (AccountClass.TAX, r"steuer|\btax(es)?\b|\bvat\b|\bust\b|mwst|mehrwertsteuer|withholding"),
    (AccountClass.ASSET, r"aktive\s+rechnungsabgrenzung|\barap\b"),
    (
        AccountClass.PAYABLE,
        r"verbindlichkeit|payable|liabilit|rueckstellung|accrual|accrued"
        r"|\bprovisions\b|provisions?\s+for|passive\s+rechnungsabgrenzung|\bprap\b|\bdeferred\b",
    ),
    (AccountClass.RECEIVABLE, r"forderung|receivable"),
    (
        AccountClass.EQUITY,
        r"eigenkapital|gezeichnet|stammkapital|grundkapital|ruecklage|gewinnvortrag"
        r"|verlustvortrag|jahresueberschuss|bilanzgewinn|\bequity\b|share capital|retained earnings?",
    ),
    (AccountClass.EXPENSE, r"aufwand|aufwendung|expense|kosten|\bcosts?\b|\bcharges?\b"),
    (AccountClass.REVENUE, r"ertrag|ertraeg|erloes|umsatz|revenue|turnover|\bsales\b|\bincome\b"),
    (
        AccountClass.EXPENSE,
        r"gebuehr|bank fees|abschreibung|depreciation|amorti[sz]ation|miete|leasing|energie"
        r"|versicherung|beitrag|beitraeg|werbe|werbung|reise|beratung|instandhaltung"
        r"|reparatur|wartung|lohn|loehne|gehalt|gehaelter|salar|\bwages?\b|payroll"
        r"|sozial|abgabe|fuhrpark|\bkfz\b|porto|telefon|bedarf|umlage|honorar"
        r"|marketing|advertis|travel|\brent\b|utilit|insurance|maintenance|repair"
        r"|refurbish|overhaul|consult|\blegal\b|\baudit\b|einsatz",
    ),
    (
        AccountClass.BANK,
        r"bank|kasse|\bcash\b|giro|kontokorrent|tagesgeld|festgeld|geldtransit"
        r"|in transit|safeguard|client money|client funds|treuhand|\bfloat\b"
        r"|settlement|\be.?money\b|wallet",
    ),
    (AccountClass.PAYABLE, r"kreditor|lieferant|supplier|vendor|merchant|payout|auszahlung"),
    (AccountClass.RECEIVABLE, r"debitor|customer|chargeback|rueckbelastung"),
    (AccountClass.REVENUE, r"\bfees?\b|interchange|provision|kommission|commission"),
    (AccountClass.EQUITY, r"\bkapital\b|\bcapital\b|\breserves?\b"),
    (
        AccountClass.ASSET,
        r"grundstueck|gebaeude|maschine|anlage|ausstattung|hardware|software|edv"
        r"|lizenz|fahrzeug|fuhrpark|vorrat|vorraete|rohstoff|betriebsstoff|erzeugnis"
        r"|ware\b|vermoegen|beteiligung|goodwill|firmenwert|immateriell|intangible"
        r"|building|equipment|\bproperty\b|machinery|\bplant\b|vehicle|inventory"
        r"|\bstock\b|prepaid|fixed asset|\bassets?\b|\bland\b|abschreibung"
        r"|depreciation|amorti[sz]ation",
    ),
]
_NAME_RULES = [(cls, re.compile(rx)) for cls, rx in _NAME_RULE_SPECS]


def _side_hint(values: list[str]) -> str | None:
    pl = any(_PL_RX.search(v) for v in values)
    bs = any(_BS_RX.search(v) for v in values)
    if pl and not bs:
        return "pl"
    if bs and not pl:
        return "bs"
    return None


def _direct_hint(values: list[str]) -> AccountClass | None:
    for v in values:
        if len(v) > 60:  # long free text is not a type field
            continue
        stripped = _BS_RX.sub(" ", _PL_RX.sub(" ", v))
        for cls, rx in _DIRECT:
            if rx.search(stripped):
                return cls
    return None


def classify_account(number: str, name: str, attrs: dict[str, str] | None = None) -> AccountClass:
    """Rule-only classification of one account. Never raises, never networks."""
    values = [_norm(v) for v in attrs.values() if v] if attrs else []
    direct = _direct_hint(values)
    if direct is not None:
        return direct
    allowed = _ALLOWED.get(_side_hint(values) or "")
    norm = _norm(name)
    for cls, rx in _NAME_RULES:
        if rx.search(norm) and (allowed is None or cls in allowed):
            return cls
    return AccountClass.OTHER


_LLM_SYSTEM = (
    "Classify general ledger accounts for a financial audit. Account names are "
    "German or English; the company may be a payments provider. For each account "
    "pick exactly one class: asset, expense, revenue, payable, receivable, bank, "
    "equity, tax, other. Reply with a JSON object mapping each account number to "
    "its class string."
)


def _llm_call(pairs: list[tuple[str, str]]) -> dict | None:
    try:
        if _ENV_FILE.exists():
            from dotenv import load_dotenv

            load_dotenv(_ENV_FILE)
    except Exception:
        pass
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI

        resp = OpenAI().chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            temperature=0,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"accounts": [{"number": n, "name": m} for n, m in pairs]},
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        data = json.loads(resp.choices[0].message.content or "")
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _llm_classify(pairs: list[tuple[str, str]]) -> dict[str, AccountClass]:
    """One batched call for all unresolved accounts, disk-cached by content hash.

    Any failure (no key, no network, bad JSON, missing package) returns {} and
    the callers keep AccountClass.OTHER.
    """
    digest = hashlib.sha256(json.dumps(pairs, ensure_ascii=False).encode("utf-8")).hexdigest()
    cache = _CACHE_DIR / f"accounts_{digest[:16]}.json"
    raw: dict | None = None
    try:
        if cache.exists():
            raw = json.loads(cache.read_text("utf-8"))
            if not isinstance(raw, dict):
                raw = None
    except Exception:
        raw = None
    if raw is None:
        raw = _llm_call(pairs)
        if raw is None:
            return {}
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(raw, ensure_ascii=False), "utf-8")
        except Exception:
            pass
    if len(raw) == 1 and isinstance(next(iter(raw.values())), dict):
        raw = next(iter(raw.values()))  # model wrapped the mapping in one key
    numbers = {n for n, _ in pairs}
    out: dict[str, AccountClass] = {}
    for num, val in raw.items():
        if num in numbers and isinstance(val, str):
            try:
                out[num] = AccountClass(val.strip().lower())
            except ValueError:
                pass
    return out


def classify_accounts(dossier: Dossier) -> dict[str, AccountClass]:
    """Classify every ACCOUNT entity in the dossier. Never raises."""
    out: dict[str, AccountClass] = {}
    unresolved: list[tuple[str, str]] = []
    for ent in dossier.entities.values():
        if ent.type is not EntityType.ACCOUNT:
            continue
        cls = classify_account(ent.id, ent.name, ent.attrs)
        out[ent.id] = cls
        if cls is AccountClass.OTHER:
            unresolved.append((ent.id, ent.name))
    if unresolved:
        for num, cls in _llm_classify(unresolved).items():
            out[num] = cls
    return out

"""Semantic lenses: the only family that reads what a posting text means.

Three checks, each its own registered lens:
  K3_capitalized_repairs     repair/maintenance wording sitting in fixed assets
  SEM_vague_large_amount     vague or empty text on GL postings >= JET_FLOOR
  SEM_text_account_mismatch  posting text contradicts the account, LLM judged

Cheap bilingual keyword prefilter first, then one batched gpt-4o-mini call
per ~100 survivors (JSON out), disk-cached under ~/.cache/laundromat/ keyed
by content hash so reruns are free. Without key or network the keyword
checks emit at reduced confidence and the LLM-only check emits nothing.
CORTEA_LLM_CAP limits the number of LLM batches per check (smoke tests).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from decimal import Decimal
from pathlib import Path
from typing import Iterable

from ..contracts import JET_FLOOR, Dossier, EntityType, Flag, LensFamily, Posting, register
from ..ingest.accounts import AccountClass, classify_accounts

_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"
_CACHE_DIR = Path.home() / ".cache" / "laundromat"
_MODEL = "gpt-4o-mini"
_BATCH = 100

_UMLAUTS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})


def _norm(s: str) -> str:
    return (s or "").lower().translate(_UMLAUTS)


_REPAIR_RX = re.compile(
    r"reparatur|instandhaltung|instandsetz|wartung|austausch|ueberhol|ersatzteil"
    r"|\brepairs?\b|mainten|refurbish|overhaul|replacement|servicing|spare part"
)
_VAGUE_RX = re.compile(
    r"diverses?\b|sonstiges?\b|verschiedenes|korrektur|umbuchung|ausgleich|sammelbuchung"
    r"|\bmisc\b|miscellaneous|sundry|adjustments?\b|corrections?\b|\btransfers?\b|reclass\w*"
)
_OPENING_RX = re.compile(r"saldenvortrag|^vortrag|eroeffnung|opening balance|brought forward|\bb/f\b")

# Tokens that carry no information; a vague text may consist only of vague
# words, these fillers, digits and punctuation.
_FILLER = {
    "allg", "allgemein", "general", "generell", "intern", "internal", "div",
    "diverse", "buchung", "booking", "posting", "konto", "account", "monat",
    "month", "periode", "period", "gemaess", "laut", "per", "vom", "zum",
    "und", "and", "fuer", "for", "the",
}


def _is_vague(text: str) -> bool:
    t = _norm(text).strip()
    if not t:
        return True
    if not _VAGUE_RX.search(t):
        return False
    rest = re.sub(r"[\W\d_]+", " ", _VAGUE_RX.sub(" ", t))
    return not [w for w in rest.split() if len(w) > 2 and w not in _FILLER]


def _fmt(a: Decimal) -> str:
    return f"{abs(a):,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _classes(dossier: Dossier) -> dict[str, AccountClass]:
    try:
        return classify_accounts(dossier)
    except Exception:
        return {}


def _base(p: Posting) -> str:
    return p.attrs.get("account_base") or p.account.split("-")[0]


def _acct_class(p: Posting, classes: dict[str, AccountClass]) -> AccountClass | None:
    return classes.get(p.account) or classes.get(_base(p))


def _acct_name(dossier: Dossier, account: str) -> str:
    for key in (account, account.split("-")[0]):
        ent = dossier.entities.get(key)
        if ent is not None and ent.type is EntityType.ACCOUNT:
            return ent.name
    return ""


def _is_opening(p: Posting) -> bool:
    return bool(
        _OPENING_RX.search(_norm(p.text)) or _OPENING_RX.search(_norm(p.attrs.get("BUCHUNGSTYP", "")))
    )


def _is_gl(p: Posting) -> bool:
    return p.attrs.get("ledger", "GL") == "GL"


# ---------------------------------------------------------------- LLM plumbing

_DEAD = False  # latched after a failed call so a dead network costs one timeout


def _llm(system: str, payload: str) -> dict | None:
    global _DEAD
    if _DEAD:
        return None
    try:
        if _ENV_FILE.exists():
            from dotenv import load_dotenv

            load_dotenv(_ENV_FILE)
    except Exception:
        pass
    if not os.environ.get("OPENAI_API_KEY"):
        _DEAD = True
        return None
    try:
        from openai import OpenAI

        resp = OpenAI(timeout=90).chat.completions.create(
            model=_MODEL,
            response_format={"type": "json_object"},
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": payload},
            ],
        )
    except Exception:
        _DEAD = True
        return None
    try:
        data = json.loads(resp.choices[0].message.content or "")
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _judge(check: str, system: str, items: list[dict]) -> dict[str, dict]:
    """Batched, disk-cached LLM verdicts: item id -> result row. {} on failure."""
    out: dict[str, dict] = {}
    batches = [items[i : i + _BATCH] for i in range(0, len(items), _BATCH)]
    cap = os.environ.get("CORTEA_LLM_CAP", "")
    if cap.isdigit():
        batches = batches[: int(cap)]
    for batch in batches:
        payload = json.dumps({"items": batch}, ensure_ascii=False)
        digest = hashlib.sha256(f"{_MODEL}|{check}|{system}|{payload}".encode()).hexdigest()
        cache = _CACHE_DIR / f"sem_{digest[:16]}.json"
        raw = None
        try:
            if cache.exists():
                raw = json.loads(cache.read_text("utf-8"))
        except Exception:
            raw = None
        if not isinstance(raw, dict):
            raw = _llm(system, payload)
            if raw is None:
                continue
            try:
                _CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(raw, ensure_ascii=False), "utf-8")
            except Exception:
                pass
        rows = raw.get("items")
        if not isinstance(rows, list):  # model returned a plain id -> row mapping
            rows = [dict(v, id=k) for k, v in raw.items() if isinstance(v, dict)]
        for row in rows:
            if isinstance(row, dict) and "id" in row:
                out[str(row["id"])] = row
    return out


def _clip(s: str, n: int = 200) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


# -------------------------------------------------------------------- lens K3


@register
class CapitalizedRepairs:
    """K3: repair or maintenance wording capitalized as fixed assets."""

    lens_id = "K3_capitalized_repairs"
    family = LensFamily.SEMANTIC

    _SYSTEM = (
        "Du unterstuetzt eine Jahresabschlusspruefung. Jedes Item ist eine Buchung auf "
        "einem Anlagenkonto oder ein Posten des Anlagenspiegels; der Text enthaelt "
        "Reparatur- oder Wartungsvokabular (deutsch oder englisch). Entscheide je Item: "
        "beschreibt der Text aktivierungsfaehige Anschaffungs- oder Herstellungskosten "
        "(capex: Neuanschaffung, Erweiterung, wesentliche Verbesserung ueber den "
        "urspruenglichen Zustand hinaus) oder nicht aktivierungsfaehigen Erhaltungsaufwand "
        "(expense: Reparatur, Wartung, Instandsetzung, gleichwertiger Teiletausch)? "
        "Antworte nur 'expense', wenn der Text klar auf Erhaltungsaufwand hindeutet. "
        'JSON: {"items": [{"id": "...", "verdict": "expense|capex|unclear", '
        '"reason": "kurz, deutsch"}]}'
    )

    @classmethod
    def run(cls, dossier: Dossier) -> Iterable[Flag]:
        try:
            return list(cls._flags(dossier))
        except Exception:
            return []

    @classmethod
    def _flags(cls, dossier: Dossier) -> Iterable[Flag]:
        classes = _classes(dossier)
        posts = [
            p
            for p in dossier.postings
            if _REPAIR_RX.search(_norm(p.text)) and _acct_class(p, classes) is AccountClass.ASSET
        ]
        assets = [
            e
            for e in dossier.entities.values()
            if e.type is EntityType.ASSET and _REPAIR_RX.search(_norm(e.name))
        ]
        if not posts and not assets:
            return
        posts.sort(key=lambda p: (p.source.file, p.source.line or 0))
        assets.sort(key=lambda e: e.id)
        items: list[dict] = []
        subjects: list[tuple[str, object]] = []
        for p in posts:
            items.append(
                {
                    "id": str(len(items)),
                    "kind": "posting",
                    "account": p.account,
                    "account_name": _clip(_acct_name(dossier, p.account), 80),
                    "text": _clip(p.text),
                    "amount": str(p.amount),
                    "currency": p.currency,
                }
            )
            subjects.append(("posting", p))
        for e in assets:
            grp = e.attrs.get("ANLAGENGRUPPE", "")
            items.append(
                {
                    "id": str(len(items)),
                    "kind": "asset",
                    "name": _clip(e.name),
                    "asset_group_account": grp,
                    "account_name": _clip(_acct_name(dossier, grp), 80) if grp else "",
                }
            )
            subjects.append(("asset", e))
        verdicts = _judge("k3", cls._SYSTEM, items)
        for i, (kind, obj) in enumerate(subjects):
            v = verdicts.get(str(i))
            if v is not None and str(v.get("verdict", "")).strip().lower() != "expense":
                continue
            reason = str(v.get("reason", "")).strip() if v else ""
            tail = (
                f" Einschaetzung: {reason}"
                if reason
                else " Schluesselwort-Treffer ohne LLM-Bestaetigung."
            )
            if kind == "posting":
                p = obj
                name = _acct_name(dossier, p.account)
                yield Flag(
                    lens_id=cls.lens_id,
                    family=cls.family,
                    title=f"Reparaturtext auf Anlagenkonto {p.account} aktiviert",
                    rationale=(
                        f"Buchungstext '{_clip(p.text, 120)}' deutet auf Erhaltungsaufwand hin, "
                        f"gebucht auf Anlagenkonto {p.account}"
                        + (f" ({name})" if name else "")
                        + f" ueber {_fmt(p.amount)} {p.currency}." + tail
                    ),
                    evidence=(p.source,),
                    entity_id=p.entity_id,
                    doc_no=p.doc_no or None,
                    amount=abs(p.amount),
                    confidence=0.7 if v else 0.3,
                )
            else:
                e = obj
                grp = e.attrs.get("ANLAGENGRUPPE", "")
                gname = _acct_name(dossier, grp) if grp else ""
                yield Flag(
                    lens_id=cls.lens_id,
                    family=cls.family,
                    title=f"Anlagegut '{_clip(e.name, 60)}' als Reparatur bezeichnet",
                    rationale=(
                        f"Der Anlagenspiegel fuehrt '{_clip(e.name, 120)}' als Vermoegensgegenstand"
                        + (f" (Anlagengruppe {grp}{', ' + gname if gname else ''})" if grp else "")
                        + ". Die Bezeichnung deutet auf nicht aktivierungsfaehigen "
                        "Erhaltungsaufwand hin." + tail
                    ),
                    evidence=(e.source,),
                    entity_id=e.id,
                    confidence=0.7 if v else 0.3,
                )


# ---------------------------------------------------------- lens vague texts


@register
class VagueLargeText:
    """Vague or empty posting text on GL amounts at or above JET_FLOOR."""

    lens_id = "SEM_vague_large_amount"
    family = LensFamily.SEMANTIC

    _SYSTEM = (
        "Du unterstuetzt eine Jahresabschlusspruefung. Jedes Item ist eine Hauptbuch-"
        "Buchung ueber der Pruefungsgrenze, deren Buchungstext leer oder unspezifisch ist "
        "(z.B. Diverses, Korrektur, Umbuchung, misc, sundry, adjustment). Bewerte je Item, "
        "wie verdaechtig die Buchung fuer einen Pruefer ist (suspicion 0 bis 1); "
        "beruecksichtige Betrag, Konto und Text. Erkennbare Routinevorgaenge niedrig "
        'bewerten. JSON: {"items": [{"id": "...", "suspicion": 0.0, '
        '"reason": "kurz, deutsch"}]}'
    )

    @classmethod
    def run(cls, dossier: Dossier) -> Iterable[Flag]:
        try:
            return list(cls._flags(dossier))
        except Exception:
            return []

    @classmethod
    def _flags(cls, dossier: Dossier) -> Iterable[Flag]:
        classes = _classes(dossier)
        posts = [
            p
            for p in dossier.postings
            if _is_gl(p) and abs(p.amount) >= JET_FLOOR and _is_vague(p.text) and not _is_opening(p)
        ]
        if not posts:
            return
        posts.sort(key=lambda p: (-abs(p.amount), p.source.file, p.source.line or 0))
        items = [
            {
                "id": str(i),
                "text": _clip(p.text),
                "amount": str(p.amount),
                "currency": p.currency,
                "account": p.account,
                "account_name": _clip(_acct_name(dossier, p.account), 80),
                "account_class": (_acct_class(p, classes) or AccountClass.OTHER).value,
            }
            for i, p in enumerate(posts)
        ]
        verdicts = _judge("vague", cls._SYSTEM, items)
        for i, p in enumerate(posts):
            v = verdicts.get(str(i))
            if v is not None:
                try:
                    s = max(0.0, min(1.0, float(v.get("suspicion", 0))))
                except (TypeError, ValueError):
                    s = 0.0
                if s < 0.5:
                    continue
                conf = round(min(0.7, 0.3 + 0.4 * s), 2)
                reason = str(v.get("reason", "")).strip()
            else:
                conf, reason = 0.3, ""
            shown = _clip(p.text, 80).strip() or "(leer)"
            name = _acct_name(dossier, p.account)
            yield Flag(
                lens_id=cls.lens_id,
                family=cls.family,
                title=f"Unspezifischer Buchungstext '{shown}' ueber {_fmt(p.amount)} {p.currency}",
                rationale=(
                    f"Buchung ueber {_fmt(p.amount)} {p.currency} auf Konto {p.account}"
                    + (f" ({name})" if name else "")
                    + f" traegt nur den Text '{shown}'. Oberhalb der JET-Grenze ist ein "
                    "aussagekraeftiger Buchungstext zu erwarten."
                    + (f" Einschaetzung: {reason}" if reason else "")
                ),
                evidence=(p.source,),
                entity_id=p.entity_id,
                doc_no=p.doc_no or None,
                amount=abs(p.amount),
                confidence=conf,
            )


# ------------------------------------------------- lens text vs account clash


@register
class TextAccountMismatch:
    """Posting text contradicts the account, on GL amounts at or above JET_FLOOR.

    LLM-only: contradiction cannot be established by keywords, so without an
    API key this lens emits nothing. Judged once per distinct (account, text)
    pair, not per posting.
    """

    lens_id = "SEM_text_account_mismatch"
    family = LensFamily.SEMANTIC

    _SYSTEM = (
        "Du unterstuetzt eine Jahresabschlusspruefung. Jedes Item ist eine eindeutige "
        "Kombination aus Hauptbuchkonto (mit Name und Klasse) und Buchungstext aus "
        "Buchungen ueber der Pruefungsgrenze. Entscheide, ob der Text dem Konto klar "
        "widerspricht (z.B. Text beschreibt Umsatz, Konto ist Aufwand; Text nennt "
        "Zinsen, Konto ist Versicherung). Die allermeisten Kombinationen sind stimmig; "
        "Prozesstexte wie 'Eingangsrechnung', 'Zahlungseingang', 'invoice', 'payment' "
        "und Saldenvortraege sind normal. Melde nur klare Widersprueche. "
        'JSON: {"items": [{"id": "...", "contradiction": true|false, '
        '"reason": "kurz, deutsch"}]}'
    )

    @classmethod
    def run(cls, dossier: Dossier) -> Iterable[Flag]:
        try:
            return list(cls._flags(dossier))
        except Exception:
            return []

    @classmethod
    def _flags(cls, dossier: Dossier) -> Iterable[Flag]:
        classes = _classes(dossier)
        pairs: dict[tuple[str, str], list[Posting]] = {}
        for p in dossier.postings:
            if not _is_gl(p) or abs(p.amount) < JET_FLOOR or _is_opening(p):
                continue
            text = _norm(p.text).strip()
            if not text:
                continue
            pairs.setdefault((_base(p), text), []).append(p)
        if not pairs:
            return
        keyed = []
        for (acct, text), plist in pairs.items():
            name = _acct_name(dossier, acct)
            acls = classes.get(acct)
            if not name and acls is None:
                continue  # nothing to contradict against
            plist.sort(key=lambda p: -abs(p.amount))
            keyed.append((acct, name, acls, text, plist))
        keyed.sort(key=lambda k: -abs(k[4][0].amount))
        items = [
            {
                "id": str(i),
                "account": acct,
                "account_name": _clip(name, 80),
                "account_class": (acls or AccountClass.OTHER).value,
                "text": _clip(plist[0].text),
                "example_amount": str(plist[0].amount),
                "posting_count": len(plist),
            }
            for i, (acct, name, acls, text, plist) in enumerate(keyed)
        ]
        verdicts = _judge("mismatch", cls._SYSTEM, items)
        for i, (acct, name, acls, text, plist) in enumerate(keyed):
            v = verdicts.get(str(i))
            if v is None:
                continue
            c = v.get("contradiction")
            if not (c is True or str(c).strip().lower() == "true"):
                continue
            reason = str(v.get("reason", "")).strip()
            top = plist[0]
            yield Flag(
                lens_id=cls.lens_id,
                family=cls.family,
                title=f"Buchungstext passt nicht zu Konto {acct}",
                rationale=(
                    f"Text '{_clip(top.text, 120)}' auf Konto {acct}"
                    + (f" ({name}" + (f", Klasse {acls.value})" if acls else ")") if name else "")
                    + f" in {len(plist)} Buchung(en) ueber der JET-Grenze widerspricht dem "
                    "Kontoinhalt." + (f" Einschaetzung: {reason}" if reason else "")
                ),
                evidence=tuple(p.source for p in plist[:3]),
                entity_id=top.entity_id,
                doc_no=top.doc_no or None,
                amount=abs(top.amount),
                confidence=0.6,
            )

"""External lens: Tavily web checks on high-volume creditors.

For each vendor whose AP payment volume clears a materiality-derived floor
(top ~15), plus vendors that appear in postings or purchase documents but
are missing from the vendor master, run one web search (name + city +
country). Flag only clear signals: zero hits, hits that never mention the
vendor, or hits tying the vendor to mail-drop / letterbox language.

Weak evidence by design: confidence is capped at 0.4 and the lens exists to
corroborate rule/graph/temporal hits, never to carry a finding alone. Every
search is cached on disk; at most 25 API calls per run (CORTEA_TAVILY_CAP
lowers that for smoke tests). No API key means the lens emits nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Iterable

from ..contracts import (
    MATERIALITY,
    Dossier,
    EntityType,
    Flag,
    LensFamily,
    SourceRef,
    register,
)

_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"
_CACHE_DIR = Path.home() / ".cache" / "laundromat"
_HARD_CAP = 25
_TOP_N = 15
_MIN_VOLUME = MATERIALITY / 4  # only vendors with real money at stake
_MAX_RESULTS = 8

_UMLAUTS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})

# Legal forms stripped before matching: "Nord Transport GmbH" -> "nord transport".
_LEGAL_RX = re.compile(
    r"\b(gmbh|mbh|ag|kg|kgaa|ohg|gbr|ug|se|e\.?\s?k\.?|e\.?\s?v\.?|co|cie|inc"
    r"|incorporated|ltd|limited|llc|llp|corp|corporation|plc|sarl|s\.?a\.?r\.?l"
    r"|bv|b\.?v\.?|nv|oy|ab|sa|s\.?a\.?|srl|spa|s\.?p\.?a\.?|gesellschaft"
    r"|haftungsbeschraenkt)\b\.?"
)
_PUNCT_RX = re.compile(r"[^a-z0-9 ]+")

# Bilingual mail-drop / letterbox vocabulary. Only counts inside a search
# result that also mentions the vendor name, never alone.
_MAILDROP_RX = re.compile(
    r"virtual office|virtuelles buero|briefkasten|letterbox|mail ?drop"
    r"|mail forwarding|mailbox service|registered agent|registered office provider"
    r"|scheinfirma|scheindomizil|domizilgesellschaft|mantelgesellschaft|shell company"
)

# ISO country codes as they appear in vendor masters -> searchable names.
_COUNTRY = {
    "DE": "Germany", "DEU": "Germany", "AT": "Austria", "AUT": "Austria",
    "CH": "Switzerland", "CHE": "Switzerland", "GB": "United Kingdom",
    "GBR": "United Kingdom", "UK": "United Kingdom", "US": "USA", "USA": "USA",
    "FR": "France", "FRA": "France", "NL": "Netherlands", "NLD": "Netherlands",
    "BE": "Belgium", "BEL": "Belgium", "IT": "Italy", "ITA": "Italy",
    "ES": "Spain", "ESP": "Spain", "PL": "Poland", "POL": "Poland",
    "CZ": "Czechia", "CZE": "Czechia", "DK": "Denmark", "DNK": "Denmark",
    "SE": "Sweden", "SWE": "Sweden", "NO": "Norway", "NOR": "Norway",
    "FI": "Finland", "FIN": "Finland", "IE": "Ireland", "IRL": "Ireland",
    "LU": "Luxembourg", "LUX": "Luxembourg", "PT": "Portugal", "PRT": "Portugal",
    "LT": "Lithuania", "LTU": "Lithuania", "EE": "Estonia", "EST": "Estonia",
    "LV": "Latvia", "LVA": "Latvia", "HU": "Hungary", "HUN": "Hungary",
    "RO": "Romania", "ROU": "Romania",
}

_VENDOR_ID_KEYS = ("KREDITOR", "VENDOR", "SUPPLIER", "LIEFERANT")
_DEBTOR_KEYS = ("DEBITOR", "KUNDE", "CUSTOMER", "CLIENT", "DEBTOR")
_VENDOR_NAME_KEYS = ("KREDITORNAME", "VENDORNAME", "VENDOR_NAME", "SUPPLIERNAME",
                     "SUPPLIER_NAME", "LIEFERANTENNAME", "NAME")
_CITY_KEYS = ("ORT", "CITY", "STADT", "TOWN")
_COUNTRY_KEYS = ("STAAT", "COUNTRY", "LAND")


def _norm(s: str) -> str:
    return (s or "").casefold().translate(_UMLAUTS)


def _eur(amount: Decimal) -> str:
    return f"{amount:,.2f}".replace(",", "\0").replace(".", ",").replace("\0", ".")


def _core_name(name: str) -> str:
    """Vendor name minus legal form and punctuation, normalized."""
    s = _PUNCT_RX.sub(" ", _LEGAL_RX.sub(" ", _norm(name)))
    return " ".join(s.split())


def _name_in_text(core: str, text: str) -> bool:
    text = _norm(text)
    if core in text:
        return True
    try:
        from rapidfuzz import fuzz

        return fuzz.partial_ratio(core, text) >= 85
    except Exception:
        return False


def _attr_value(
    attrs: dict[str, str], key_parts: tuple[str, ...], exclude: tuple[str, ...] = ()
) -> str:
    for k, v in attrs.items():
        ku = k.upper()
        if not v or any(part in ku for part in exclude):
            continue
        if any(part in ku for part in key_parts):
            return v.strip()
    return ""


def _api_key() -> str | None:
    try:
        if _ENV_FILE.exists():
            from dotenv import load_dotenv

            load_dotenv(_ENV_FILE)
    except Exception:
        pass
    return os.environ.get("TAVILY_API_KEY") or None


def _call_cap() -> int:
    try:
        return max(0, min(_HARD_CAP, int(os.environ.get("CORTEA_TAVILY_CAP", _HARD_CAP))))
    except (TypeError, ValueError):
        return _HARD_CAP


class _Budget:
    def __init__(self, cap: int) -> None:
        self.left = cap


def _search(query: str, key: str, budget: _Budget) -> list[dict] | None:
    """Cached Tavily search. None means unavailable (error or budget), not empty."""
    digest = hashlib.sha256(_norm(query).encode("utf-8")).hexdigest()
    cache = _CACHE_DIR / f"tavily_{digest[:16]}.json"
    try:
        if cache.exists():
            data = json.loads(cache.read_text("utf-8"))
            if isinstance(data, dict) and isinstance(data.get("results"), list):
                return data["results"]
    except Exception:
        pass
    if budget.left <= 0:
        return None
    try:
        from tavily import TavilyClient

        budget.left -= 1
        resp = TavilyClient(api_key=key).search(
            query, search_depth="basic", max_results=_MAX_RESULTS
        )
        results = resp.get("results") if isinstance(resp, dict) else None
        if not isinstance(results, list):
            return None
        slim = [
            {
                "title": str(r.get("title") or ""),
                "url": str(r.get("url") or ""),
                "content": str(r.get("content") or "")[:400],
            }
            for r in results
            if isinstance(r, dict)
        ]
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"query": query, "results": slim}, ensure_ascii=False), "utf-8")
        except Exception:
            pass
        return slim
    except Exception:
        return None


def _ap_volumes(dossier: Dossier) -> dict[str, Decimal]:
    """Absolute AP volume per vendor id. Falls back to any posting bound to a
    vendor entity when the ingest did not mark ledgers."""
    vol: dict[str, Decimal] = defaultdict(Decimal)
    marked = [p for p in dossier.postings if p.attrs.get("ledger") == "AP"]
    if marked:
        for p in marked:
            if p.entity_id:
                vol[p.entity_id] += abs(p.amount)
        return vol
    for p in dossier.postings:
        ent = dossier.entities.get(p.entity_id or "")
        if ent is not None and ent.type is EntityType.VENDOR:
            vol[p.entity_id] += abs(p.amount)
    return vol


def _orphan_vendors(dossier: Dossier, volumes: dict[str, Decimal]) -> list[tuple[str, str, SourceRef, Decimal]]:
    """(id, name, source, amount) for vendors used in postings or purchase
    documents but absent from the vendor master."""
    out: dict[str, tuple[str, str, SourceRef, Decimal]] = {}
    for doc in dossier.documents:
        if doc.kind not in ("purchase_invoice", "goods_receipt", "open_item"):
            continue
        vid = _attr_value(doc.fields, _VENDOR_ID_KEYS, exclude=("NAME",))
        if not vid and doc.kind != "open_item":
            vid = (doc.entity_id or "").strip()  # vendor-specific kinds only
        if not vid or vid in dossier.entities or vid in out:
            continue
        name = _attr_value(doc.fields, _VENDOR_NAME_KEYS, exclude=_DEBTOR_KEYS)
        if name:
            out[vid] = (vid, name, doc.source, doc.amount or Decimal(0))
    # Orphans that only appear in postings carry no name anywhere, so there is
    # nothing to search for; they are left to the rule/graph lenses.
    return list(out.values())


class VendorWebPresence:
    lens_id = "X1_vendor_web_presence"
    family = LensFamily.EXTERNAL

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        try:
            yield from self._run(dossier)
        except Exception:
            return

    def _run(self, dossier: Dossier) -> Iterable[Flag]:
        key = _api_key()
        if not key:
            return
        volumes = _ap_volumes(dossier)
        if not volumes and not dossier.entities:
            return

        # Candidates: orphans first (strongest prior), then top vendors by volume.
        candidates: list[tuple[str, str, str, str, SourceRef, Decimal]] = []
        seen: set[str] = set()
        for vid, name, src, amount in _orphan_vendors(dossier, volumes):
            candidates.append((vid, name, "", "", src, volumes.get(vid, amount)))
            seen.add(vid)
        ranked = sorted(volumes.items(), key=lambda kv: -kv[1])
        for vid, vol in ranked[:_TOP_N]:
            if vid in seen or vol < _MIN_VOLUME:
                continue
            ent = dossier.entities.get(vid)
            if ent is None or ent.type is not EntityType.VENDOR or not ent.name.strip():
                continue
            city = _attr_value(ent.attrs, _CITY_KEYS)
            country = _attr_value(ent.attrs, _COUNTRY_KEYS)
            candidates.append((vid, ent.name.strip(), city, country, ent.source, vol))
            seen.add(vid)

        budget = _Budget(_call_cap())
        for vid, name, city, country, master_ref, amount in candidates:
            core = _core_name(name)
            if len(core) < 5 or core.isdigit():
                continue  # name too generic to judge absence reliably
            country_full = _COUNTRY.get(country.upper(), country)
            query = " ".join(x for x in (name, city, country_full) if x)
            results = _search(query, key, budget)
            if results is None:
                continue
            flag = self._judge(vid, name, core, query, results, master_ref, amount)
            if flag is not None:
                yield flag

    def _judge(
        self,
        vid: str,
        name: str,
        core: str,
        query: str,
        results: list[dict],
        master_ref: SourceRef,
        amount: Decimal,
    ) -> Flag | None:
        mentioning = [
            r for r in results
            if _name_in_text(core, f"{r['title']} {r['url']} {r['content']}")
        ]
        if mentioning:
            for r in mentioning:
                text = f"{r['title']} {r['content']}"
                m = _MAILDROP_RX.search(_norm(text))
                if m:
                    return Flag(
                        lens_id=self.lens_id,
                        family=self.family,
                        title=f"Kreditor {name}: Web-Treffer deutet auf Briefkastenadresse",
                        rationale=(
                            f'Tavily-Suche nach "{query}": Treffer "{r["title"]}" nennt den '
                            f'Kreditor zusammen mit dem Begriff "{m.group(0)}". Moegliches '
                            f"Scheindomizil. Nur als Korroboration anderer Befunde verwertbar."
                        ),
                        evidence=(
                            SourceRef(file=r["url"], excerpt=text[:200]),
                            master_ref,
                        ),
                        entity_id=vid,
                        amount=amount,
                        confidence=0.4,
                    )
            return None  # web presence confirmed, nothing suspicious
        if not results:
            return Flag(
                lens_id=self.lens_id,
                family=self.family,
                title=f"Kreditor {name}: keine Webpraesenz auffindbar",
                rationale=(
                    f'Tavily-Suche nach "{query}" lieferte null Treffer. Fuer einen Kreditor '
                    f"mit Zahlungsvolumen von {_eur(amount)} EUR ist das auffaellig. Schwaches "
                    f"Signal, nur als Korroboration anderer Befunde verwertbar."
                ),
                evidence=(master_ref,),
                entity_id=vid,
                amount=amount,
                confidence=0.35,
            )
        if len(results) < 3:
            return None  # thin result set, not judgeable
        top = results[0]
        return Flag(
            lens_id=self.lens_id,
            family=self.family,
            title=f"Kreditor {name}: Suchtreffer ohne Bezug zum Firmennamen",
            rationale=(
                f'Tavily-Suche nach "{query}": keiner der {len(results)} Treffer enthaelt '
                f'den Firmennamen "{name}". Bester Treffer war "{top["title"]}" und betrifft '
                f"etwas anderes. Keine belastbare Webpraesenz fuer einen Kreditor mit "
                f"Zahlungsvolumen von {_eur(amount)} EUR. Schwaches Signal, nur als "
                f"Korroboration anderer Befunde verwertbar."
            ),
            evidence=(
                SourceRef(file=top["url"], excerpt=f"{top['title']} {top['content']}"[:200]),
                master_ref,
            ),
            entity_id=vid,
            amount=amount,
            confidence=0.3,
        )


register(VendorWebPresence())

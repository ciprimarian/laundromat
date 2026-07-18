"""Cognee-backed graph lenses (family GRAPH), complementing lenses/graph.py.

Deterministic core runs offline and emits the flags. Cognee (partner tech)
builds the same relationships as a knowledge graph and, when reachable, adds
a corroboration note to the rationale. Any cognee failure, a missing
OPENAI_API_KEY, or CORTEA_SKIP_COGNEE=1 degrades silently to the
deterministic result.
"""

from __future__ import annotations

import os
import re
import threading
from collections import defaultdict
from decimal import Decimal
from typing import Iterable

from rapidfuzz import fuzz

from ..contracts import (
    Dossier,
    Entity,
    EntityType,
    Flag,
    LensFamily,
    Posting,
    SourceRef,
    register,
)

_TRUTHY = {"x", "ja", "yes", "y", "true", "1"}
_FALSY = {"nein", "no", "false", "0"}
_VAT_RE = re.compile(r"[A-Z]{2}[0-9A-Z]{5,}")
_LEGAL_FORMS = {
    "gmbh", "mbh", "ag", "kg", "kgaa", "ohg", "gbr", "ug", "se", "ek", "e", "k",
    "co", "cie", "ltd", "llc", "inc", "plc", "corp", "bv", "nv", "sa", "sarl",
    "sas", "company", "limited", "holding", "group", "gruppe",
}
_STREET_STOP = {"str", "st", "rd", "ave", "weg", "platz", "allee", "gasse", "ring"}
_TYPE_DE = {"vendor": "Kreditor", "customer": "Debitor"}

_COGNEE_TIMEOUT = float(os.environ.get("CORTEA_COGNEE_TIMEOUT", "85"))
_COGNEE_DATASET = "laundromat_graph"
_COGNEE_NOTE = " Graphanalyse (Cognee) bestaetigt die Verbindung."


def _norm_name(s: str) -> str:
    s = s.casefold().replace("ß", "ss")
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _norm_street(s: str) -> str:
    s = s.casefold().replace("ß", "ss")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"strasse\b|str\b", "str", s)
    s = re.sub(r"\bstreet\b", "st", s)
    s = re.sub(r"\broad\b", "rd", s)
    s = re.sub(r"\bavenue\b", "ave", s)
    return re.sub(r"\s+", " ", s).strip()


def _attr(e: Entity, needles: tuple[str, ...], skip: tuple[str, ...] = ()) -> str:
    for k, v in e.attrs.items():
        ku = k.casefold()
        if any(n in ku for n in skip):
            continue
        if any(n in ku for n in needles) and v and v.strip():
            return v.strip()
    return ""


def _street(e: Entity) -> str:
    return _attr(e, ("strasse", "straße", "street", "address", "addr"),
                 skip=("mail", "web"))


def _zip(e: Entity) -> str:
    return _attr(e, ("plz", "zip", "postal", "postcode"))


def _dfield(fields: dict[str, str], needles: tuple[str, ...]) -> str:
    for k, v in fields.items():
        ku = k.casefold()
        if any(n in ku for n in needles) and v and v.strip():
            return v.strip()
    return ""


def _name_tokens(name: str) -> set[str]:
    return {t for t in _norm_name(name).split()
            if len(t) >= 4 and t not in _LEGAL_FORMS and not t.isdigit()}


def _dedupe_postings(postings: list[Posting]) -> list[Posting]:
    """GL and subledger carry the same economics; keep one row per document."""
    seen: set[tuple[str, str, str]] = set()
    out: list[Posting] = []
    for p in postings:
        key = (p.doc_no, str(p.booking_date), str(p.amount))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


# --------------------------------------------------------------------------
# Deterministic core: computed once per dossier, shared by all graph lenses.
# --------------------------------------------------------------------------


class _Core:
    def __init__(self, dossier: Dossier) -> None:
        self.addr_pairs: list[tuple[Entity, Entity, str, str]] = []
        self.vat_groups: list[tuple[str, list[Entity]]] = []
        self.name_pairs: list[tuple[Entity, Entity, float, str]] = []
        self.self_approvals: list[dict] = []
        self.right_violations: list[dict] = []
        self.orphans: list[dict] = []
        self.master_issues: list[dict] = []
        self.related: list[dict] = []
        self.corroborated: set[str] = set()

        for step in (self._shared_addresses, self._duplicate_vat,
                     self._near_duplicate_names, self._sod, self._related_party):
            try:
                step(dossier)
            except Exception:
                continue
        try:
            self._cognee(dossier)
        except Exception:
            pass

    # -- shared address ----------------------------------------------------

    def _shared_addresses(self, dossier: Dossier) -> None:
        groups: dict[tuple[str, str], list[Entity]] = defaultdict(list)
        for e in dossier.entities.values():
            if e.type not in (EntityType.VENDOR, EntityType.CUSTOMER):
                continue
            street, zipc = _street(e), _zip(e)
            if street and zipc:
                groups[(_norm_street(street), zipc.replace(" ", "").casefold())].append(e)
        for (street, zipc), es in groups.items():
            if len(es) < 2 or len(es) > 6:
                continue  # >6 shared entries is a data artefact, not a finding
            for i in range(len(es)):
                for j in range(i + 1, len(es)):
                    a, b = es[i], es[j]
                    ratio = fuzz.token_sort_ratio(_norm_name(a.name), _norm_name(b.name))
                    if ratio >= 90:
                        continue  # near-identical names are the near-duplicate-name case
                    self.addr_pairs.append((a, b, street, zipc))

    # -- duplicate VAT id --------------------------------------------------

    def _duplicate_vat(self, dossier: Dossier) -> None:
        by_vat: dict[str, dict[str, Entity]] = defaultdict(dict)
        for e in dossier.entities.values():
            if e.type not in (EntityType.VENDOR, EntityType.CUSTOMER):
                continue
            for k, v in e.attrs.items():
                ku = k.upper()
                if not ("USTID" in ku or "VAT" in ku or "TAX" in ku):
                    continue
                vat = (v or "").upper().replace(" ", "").replace(".", "")
                if _VAT_RE.fullmatch(vat):
                    by_vat[vat][e.id] = e
        for vat, by_id in by_vat.items():
            es = list(by_id.values())
            if len(es) < 2:
                continue
            if (len(es) == 2 and es[0].type != es[1].type
                    and fuzz.token_sort_ratio(_norm_name(es[0].name),
                                              _norm_name(es[1].name)) >= 95):
                continue  # same firm kept as both vendor and customer: normal
            self.vat_groups.append((vat, es))

    # -- near-duplicate vendor names ---------------------------------------

    def _near_duplicate_names(self, dossier: Dossier) -> None:
        vendors = [e for e in dossier.entities.values() if e.type == EntityType.VENDOR]
        for i in range(len(vendors)):
            for j in range(i + 1, len(vendors)):
                a, b = vendors[i], vendors[j]
                ratio = fuzz.token_sort_ratio(_norm_name(a.name), _norm_name(b.name))
                if ratio < 90:
                    continue
                signal = ""
                za, zb = _zip(a), _zip(b)
                if za and za == zb:
                    signal = f"gleiche PLZ {za}"
                else:
                    ta = {t for t in _norm_street(_street(a)).split()
                          if len(t) >= 4 and t not in _STREET_STOP and not t.isdigit()}
                    tb = {t for t in _norm_street(_street(b)).split()
                          if len(t) >= 4 and t not in _STREET_STOP and not t.isdigit()}
                    common = ta & tb
                    if common:
                        signal = f"gleicher Strassenname '{sorted(common)[0]}'"
                    elif (a.id.isdigit() and b.id.isdigit()
                          and abs(int(a.id) - int(b.id)) == 1):
                        signal = f"direkt aufeinanderfolgende Kontonummern {a.id}/{b.id}"
                if signal:
                    self.name_pairs.append((a, b, ratio, signal))

    # -- segregation of duties ---------------------------------------------

    def _rights(self, dossier: Dossier) -> dict[str, dict]:
        rights: dict[str, dict] = {}
        for doc in dossier.docs_of("permission"):
            user = (_dfield(doc.fields, ("benutzer", "user")) or doc.ref or "").strip()
            if not user:
                continue
            r = {"post": False, "approve": False, "doc": doc}
            for k, v in doc.fields.items():
                ku, val = k.casefold(), (v or "").strip().casefold()
                if val not in _TRUTHY:
                    continue
                if "buchen" in ku or "post" in ku or ku == "entry":
                    r["post"] = True
                if ("freigeb" in ku or "genehmig" in ku or "approv" in ku
                        or "release" in ku):
                    r["approve"] = True
            rights[user] = r
        return rights

    def _sod(self, dossier: Dossier) -> None:
        rights = self._rights(dossier)
        creator_keys = ("ersteller", "erfasser", "creator", "created by",
                        "entered by", "prepared by")
        approver_keys = ("freigeber", "genehmiger", "approver", "approved by",
                         "released by")
        for doc in dossier.docs_of("approval"):
            er = _dfield(doc.fields, creator_keys)
            fr = _dfield(doc.fields, approver_keys)
            if er and fr and er == fr:
                self.self_approvals.append({"user": er, "doc": doc})
            if fr and fr in rights and not rights[fr]["approve"]:
                self.right_violations.append(
                    {"user": fr, "doc": doc, "kind": "approve",
                     "perm": rights[fr]["doc"]})
            if er and er in rights and not rights[er]["post"]:
                self.right_violations.append(
                    {"user": er, "doc": doc, "kind": "post",
                     "perm": rights[er]["doc"]})

        for doc in dossier.docs_of("master_change"):
            ch = _dfield(doc.fields, ("geaendert", "geändert", "changed by",
                                      "modified by"))
            ap = _dfield(doc.fields, ("genehmigt_von", "genehmigt von",
                                      "approved_by", "approved by"))
            if ch and ap and ch == ap:
                self.master_issues.append({"user": ch, "doc": doc})

        if not rights or len(rights) < 3:
            return  # matrix missing or too thin to call anyone an orphan
        by_user: dict[str, list[Posting]] = defaultdict(list)
        for p in dossier.postings:
            if p.user and p.attrs.get("ledger", "GL") == "GL":
                by_user[p.user.strip()].append(p)
        orphan_users = [u for u in by_user if u not in rights]
        if not by_user or len(orphan_users) / len(by_user) > 0.4:
            return  # matrix likely incomplete; flagging would be noise
        for u in orphan_users:
            rows = by_user[u]
            if len(rows) < 5:
                continue
            self.orphans.append(
                {"user": u, "rows": rows,
                 "perm_src": next(iter(rights.values()))["doc"].source})

    # -- related parties ----------------------------------------------------

    def _related_party(self, dossier: Dossier) -> None:
        shareholders = dossier.docs_of("shareholder")
        if not shareholders:
            return
        partners = [e for e in dossier.entities.values()
                    if e.type in (EntityType.VENDOR, EntityType.CUSTOMER)]
        matched: dict[str, dict] = {}
        for doc in shareholders:
            sh_name = _dfield(doc.fields, ("name",)) or doc.ref or ""
            remark = _dfield(doc.fields, ("bemerk", "remark", "note", "comment"))
            base = re.sub(r"\([^)]*\)", " ", sh_name)
            base = re.sub(r",.*$", " ", base).strip()
            sh_tokens = _name_tokens(base)
            for e in partners:
                ratio = fuzz.token_sort_ratio(_norm_name(base), _norm_name(e.name))
                if ratio >= 85 and (sh_tokens & _name_tokens(e.name)):
                    if e.id not in matched:
                        matched[e.id] = {"entity": e, "doc": doc,
                                         "reason": f"Namensabgleich {ratio:.0f}%",
                                         "conf": 0.6}
            for acc in re.findall(r"\b\d{4,}\b", remark):
                e = dossier.entities.get(acc)
                if e and e.type in (EntityType.VENDOR, EntityType.CUSTOMER):
                    matched[acc] = {"entity": e, "doc": doc,
                                    "reason": "Personenkonto laut Gesellschafterliste",
                                    "conf": 0.7}
        for m in matched.values():
            pays = _dedupe_postings(
                [p for p in dossier.postings if p.entity_id == m["entity"].id])
            if pays:
                m["postings"] = pays
                self.related.append(m)

    # -- cognee corroboration ----------------------------------------------

    def _candidates(self) -> tuple[list[str], list[str]]:
        """(corpus lines for the graph, keys to look for in graph output)."""
        lines: list[str] = []
        keys: list[str] = []

        def ent(e: Entity) -> None:
            keys.extend([e.id, e.name])
            addr = e.address or f"{_street(e)}, {_zip(e)}"
            lines.append(f"{e.type.value} account {e.id} '{e.name}' "
                         f"is registered at address {addr}.")

        for a, b, street, zipc in self.addr_pairs:
            ent(a)
            ent(b)
            lines.append(f"Accounts {a.id} and {b.id} share the address "
                         f"{street} {zipc}.")
        for vat, es in self.vat_groups:
            for e in es:
                ent(e)
            lines.append(f"The VAT id {vat} is used by "
                         + " and ".join(f"{e.id} '{e.name}'" for e in es) + ".")
        for a, b, ratio, signal in self.name_pairs:
            ent(a)
            ent(b)
            lines.append(f"Vendor names '{a.name}' ({a.id}) and '{b.name}' "
                         f"({b.id}) are nearly identical.")
        for item in self.self_approvals + self.master_issues:
            keys.append(item["user"])
            lines.append(f"User {item['user']} created and approved the same "
                         f"record {item['doc'].ref} without a second person.")
        for item in self.orphans:
            keys.append(item["user"])
            lines.append(f"User {item['user']} posted {len(item['rows'])} journal "
                         f"lines but is missing from the permission matrix.")
        for m in self.related:
            e = m["entity"]
            ent(e)
            sh = m["doc"].fields.get("NAME") or m["doc"].ref
            lines.append(f"Shareholder '{sh}' is related to {e.type.value} "
                         f"account {e.id} '{e.name}', which received payments.")
        return lines, keys

    def _cognee(self, dossier: Dossier) -> None:
        if os.environ.get("CORTEA_SKIP_COGNEE") == "1":
            return
        try:
            from dotenv import find_dotenv, load_dotenv
            load_dotenv(find_dotenv(usecwd=True))
        except Exception:
            pass
        if not os.environ.get("OPENAI_API_KEY"):
            return
        # cognee reads its own env name, not OPENAI_API_KEY
        os.environ.setdefault("LLM_API_KEY", os.environ["OPENAI_API_KEY"])
        lines, keys = self._candidates()
        if not lines:
            return

        found: list[set[str]] = []

        def work() -> None:
            try:
                import asyncio
                import cognee

                cache = os.path.expanduser("~/.cache/laundromat/cognee")
                os.makedirs(cache, exist_ok=True)
                try:
                    cognee.config.data_root_directory(os.path.join(cache, "data"))
                    cognee.config.system_root_directory(os.path.join(cache, "system"))
                except Exception:
                    pass

                def maybe_await(x):
                    if asyncio.iscoroutine(x):
                        return asyncio.new_event_loop().run_until_complete(x)
                    return x

                maybe_await(cognee.add("\n".join(lines),
                                       dataset_name=_COGNEE_DATASET))
                maybe_await(cognee.cognify(datasets=[_COGNEE_DATASET]))
                results = maybe_await(cognee.search(
                    query_text=("Which accounts, companies or users in this graph "
                                "are connected to each other through shared "
                                "addresses, shared VAT ids, near-identical names, "
                                "self-approval or shareholder relationships? "
                                "List their ids and names."),
                    datasets=[_COGNEE_DATASET]))
                blob = " ".join(str(r) for r in results).casefold()
                found.append({k for k in keys if k and k.casefold() in blob})
            except Exception:
                pass

        t = threading.Thread(target=work, daemon=True)
        t.start()
        t.join(_COGNEE_TIMEOUT)
        if found:
            self.corroborated = found[0]


_CORES: dict[int, _Core] = {}


def _core(dossier: Dossier) -> _Core:
    key = id(dossier)
    if key not in _CORES:
        _CORES[key] = _Core(dossier)
    return _CORES[key]


def _note(core: _Core, *keys: str) -> str:
    return _COGNEE_NOTE if any(k in core.corroborated for k in keys if k) else ""


def _label(e: Entity) -> str:
    return f"{_TYPE_DE.get(e.type.value, e.type.value)} {e.id}"


# --------------------------------------------------------------------------
# Lenses. Thin formatting layers over the shared core.
# --------------------------------------------------------------------------


@register
class SharedAddress:
    lens_id = "GC_shared_address"
    family = LensFamily.GRAPH

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        flags: list[Flag] = []
        core = _core(dossier)
        for a, b, street, zipc in core.addr_pairs:
            flags.append(Flag(
                lens_id=self.lens_id,
                family=self.family,
                title=f"Gemeinsame Anschrift: {_label(a)} und {_label(b)}",
                rationale=(f"{_label(a)} '{a.name}' und {_label(b)} '{b.name}' sind "
                           f"unter derselben Anschrift '{street}, {zipc}' erfasst. "
                           "Formal unabhaengige Geschaeftspartner mit identischer "
                           "Anschrift deuten auf eine verdeckte Verflechtung hin."
                           + _note(core, a.id, b.id)),
                evidence=(a.source, b.source),
                entity_id=a.id,
                confidence=0.65,
            ))
        return flags


@register
class DuplicateVat:
    lens_id = "GC_duplicate_vat"
    family = LensFamily.GRAPH

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        flags: list[Flag] = []
        core = _core(dossier)
        for vat, es in core.vat_groups:
            names = ", ".join(f"{_label(e)} '{e.name}'" for e in es)
            flags.append(Flag(
                lens_id=self.lens_id,
                family=self.family,
                title=f"USt-IdNr. {vat} bei {len(es)} verschiedenen Konten",
                rationale=(f"Die USt-IdNr. {vat} ist bei {names} hinterlegt. Eine "
                           "USt-IdNr. identifiziert genau ein Unternehmen; mehrere "
                           "Stammsaetze mit derselben Nummer sprechen fuer dieselbe "
                           "wirtschaftliche Partei hinter mehreren Konten."
                           + _note(core, *[e.id for e in es])),
                evidence=tuple(e.source for e in es[:6]),
                entity_id=es[0].id,
                confidence=0.8,
            ))
        return flags


@register
class NearDuplicateName:
    lens_id = "GC_near_duplicate_name"
    family = LensFamily.GRAPH

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        flags: list[Flag] = []
        core = _core(dossier)
        for a, b, ratio, signal in core.name_pairs:
            flags.append(Flag(
                lens_id=self.lens_id,
                family=self.family,
                title=f"Nahezu identische Kreditorennamen: {a.id} / {b.id}",
                rationale=(f"'{a.name}' ({a.id}) und '{b.name}' ({b.id}) stimmen zu "
                           f"{ratio:.0f}% ueberein, zusaetzlich {signal}. Doppelt "
                           "angelegte Kreditoren ermoeglichen Doppelzahlungen und "
                           "verschleierte Zahlstroeme."
                           + _note(core, a.id, b.id)),
                evidence=(a.source, b.source),
                entity_id=a.id,
                confidence=0.7,
            ))
        return flags


@register
class SegregationOfDuties:
    lens_id = "GC_sod"
    family = LensFamily.GRAPH

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        flags: list[Flag] = []
        core = _core(dossier)
        for item in core.self_approvals:
            doc = item["doc"]
            flags.append(Flag(
                lens_id=self.lens_id,
                family=self.family,
                title=f"Selbstfreigabe durch {item['user']} (Journal {doc.ref})",
                rationale=(f"Benutzer {item['user']} hat das Journal {doc.ref} "
                           "selbst erfasst und selbst freigegeben. Das "
                           "Vier-Augen-Prinzip ist ausser Kraft gesetzt."
                           + _note(core, item["user"])),
                evidence=(doc.source,),
                doc_no=doc.ref,
                confidence=0.8,
            ))
        for item in core.master_issues:
            doc = item["doc"]
            what = doc.fields.get("FELD") or doc.fields.get("NAME") or doc.ref
            flags.append(Flag(
                lens_id=self.lens_id,
                family=self.family,
                title=f"Stammdatenaenderung ohne Vier-Augen-Prinzip ({doc.ref})",
                rationale=(f"Benutzer {item['user']} hat die Stammdatenaenderung "
                           f"'{what}' an Konto {doc.ref} selbst erfasst und selbst "
                           "genehmigt."
                           + _note(core, item["user"])),
                evidence=(doc.source,),
                entity_id=doc.entity_id,
                confidence=0.75,
            ))
        for item in core.right_violations:
            doc, user = item["doc"], item["user"]
            action = ("freigegeben, besitzt laut Berechtigungsmatrix aber kein "
                      "Freigaberecht" if item["kind"] == "approve" else
                      "erfasst, besitzt laut Berechtigungsmatrix aber kein "
                      "Buchungsrecht")
            flags.append(Flag(
                lens_id=self.lens_id,
                family=self.family,
                title=f"Rechteverletzung: {user} (Journal {doc.ref})",
                rationale=f"Benutzer {user} hat das Journal {doc.ref} {action}."
                          + _note(core, user),
                evidence=(doc.source, item["perm"].source),
                doc_no=doc.ref,
                confidence=0.75,
            ))
        for item in core.orphans:
            rows = item["rows"]
            total = sum(abs(p.amount) for p in rows)
            flags.append(Flag(
                lens_id=self.lens_id,
                family=self.family,
                title=f"Benutzer {item['user']} fehlt in der Berechtigungsmatrix",
                rationale=(f"Benutzer {item['user']} hat {len(rows)} Journalzeilen "
                           f"(Summe {total:,.2f}) gebucht, ist in der "
                           "Berechtigungsauswertung aber nicht enthalten. Buchungen "
                           "ohne dokumentierte Berechtigung entziehen sich der "
                           "Kontrolle."
                           + _note(core, item["user"])),
                evidence=tuple([item["perm_src"]] + [p.source for p in rows[:4]]),
                amount=total,
                confidence=0.6,
            ))
        return flags


@register
class RelatedParty:
    lens_id = "GC_related_party"
    family = LensFamily.GRAPH

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        flags: list[Flag] = []
        core = _core(dossier)
        for m in core.related:
            e, doc, pays = m["entity"], m["doc"], m["postings"]
            total = sum(abs(p.amount) for p in pays)
            sh_name = doc.fields.get("NAME") or doc.ref
            flags.append(Flag(
                lens_id=self.lens_id,
                family=self.family,
                title=f"Zahlungen an nahestehende Partei: {_label(e)}",
                rationale=(f"{_label(e)} '{e.name}' ist laut Gesellschafterliste "
                           f"mit '{sh_name}' verbunden ({m['reason']}). Ueber das "
                           f"Konto liefen {len(pays)} Buchungen mit einem Volumen "
                           f"von {total:,.2f}. Geschaefte mit nahestehenden "
                           "Parteien sind auf Fremdueblichkeit zu pruefen."
                           + _note(core, e.id, e.name)),
                evidence=tuple([doc.source, e.source] + [p.source for p in pays[:4]]),
                entity_id=e.id,
                amount=total,
                confidence=m["conf"],
            ))
        return flags

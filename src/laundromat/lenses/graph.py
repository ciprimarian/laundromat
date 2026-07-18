"""Graph lenses: pure-python entity/user joins (no cognee required).

Practice-set flag rates (calibrated post-scoring merge, 26647 postings):
  G_self_approval          1  0.00%  master self-approve (209101)
  G_orphan_user            0  0.00%  posters absent from permission matrix
  G_rights_violation       4  0.02%  MV-U11 stammdaten without right
  G_shared_identity        0  0.00%  vendor↔customer address/VAT
  G_near_vendor            1  0.00%  Titan Verpackung GmbH/KG
  G_shareholder_link       1  0.00%  Muster Beteiligungs → 209113
  # all << 2% of rows; no further threshold cuts

Family GRAPH. Confidence 0.35-0.7. Pure joins only; cognee deferred.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from decimal import Decimal
from typing import Iterable

from rapidfuzz import fuzz

from ..contracts import (
    Dossier,
    Document,
    Entity,
    EntityType,
    Flag,
    LensFamily,
    SourceRef,
    register,
)

_LEGAL = re.compile(
    r"\b(gmbh|ag|se|kg|ohg|gbr|e\.?\s*k\.?|inc|ltd|llc|corp|co|plc|sarl|bv|nv|ab|oy|srl|spa)\b",
    re.I,
)


def _field(doc: Document, *names: str) -> str:
    for n in names:
        if n in doc.fields and doc.fields[n]:
            return doc.fields[n]
        for k, v in doc.fields.items():
            if k.lower() == n.lower() and v:
                return v
    return ""


def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").casefold()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        s = s.replace(a, b)
    return s


def _norm_addr(s: str) -> str:
    s = _norm_text(s)
    s = re.sub(r"\bstr\.?\b", "strasse", s)
    s = re.sub(r"\bstrasse\b", "strasse", s)
    return re.sub(r"[^a-z0-9]+", "", s)


def _core_name(name: str) -> str:
    s = _norm_text(name)
    s = _LEGAL.sub(" ", s)
    return re.sub(r"[^a-z0-9]+", "", s)


def _vat_id(entity: Entity) -> str:
    for k, v in entity.attrs.items():
        ku = k.upper()
        if any(x in ku for x in ("USTID", "VATID", "VAT_ID", "UID", "STEUERNR")):
            return re.sub(r"[\s.-]+", "", (v or "").upper())
    return ""


def _plz(address: str | None) -> str:
    if not address:
        return ""
    m = re.search(r"\b(\d{5})\b", address)
    return m.group(1) if m else ""


# --------------------------------------------------------------------------
# Self-approval
# --------------------------------------------------------------------------


class SelfApproval:
    """ERSTELLER==FREIGEBER or GEAENDERT_VON==GENEHMIGT_VON."""

    lens_id = "G_self_approval"
    family = LensFamily.GRAPH

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        for doc in dossier.docs_of("approval"):
            a = _field(doc, "ERSTELLER", "CREATOR", "CREATED_BY")
            b = _field(doc, "FREIGEBER", "APPROVER", "APPROVED_BY")
            if a and b and a == b:
                title = f"Selbstfreigabe Journal: {doc.ref}"
                rationale = f"ERSTELLER == FREIGEBER ({a}), Journal/Ref={doc.ref}."
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=(doc.source,),
                    confidence=0.65,
                )

        for doc in dossier.docs_of("master_change"):
            a = _field(doc, "GEAENDERT_VON", "CHANGED_BY", "USER")
            b = _field(doc, "GENEHMIGT_VON", "APPROVED_BY")
            if a and b and a == b:
                eid = doc.entity_id or _field(doc, "KONTO", "ACCOUNT")
                title = f"Selbstfreigabe Stammdaten: {eid}"
                rationale = (
                    f"GEAENDERT_VON == GENEHMIGT_VON ({a}), "
                    f"Feld={_field(doc, 'FELD', 'FIELD')!r}, Konto={eid}."
                )
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=(doc.source,),
                    entity_id=eid or None,
                    confidence=0.7,
                )


# --------------------------------------------------------------------------
# Orphan users
# --------------------------------------------------------------------------


class OrphanUser:
    """Users who post but are absent from the permission matrix."""

    lens_id = "G_orphan_user"
    family = LensFamily.GRAPH

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        perms = dossier.docs_of("permission")
        if not perms:
            return

        known: set[str] = set()
        perm_by: dict[str, Document] = {}
        for p in perms:
            uid = p.ref or _field(p, "Benutzer", "User", "USER", "BENUTZER")
            if uid:
                known.add(uid)
                perm_by[uid] = p

        counts: dict[str, list] = defaultdict(list)
        for post in dossier.postings:
            if post.user:
                counts[post.user].append(post)

        for user, rows in counts.items():
            if user in known:
                continue
            if len(rows) < 5:
                continue
            title = f"Orphan-Benutzer: {user}"
            rationale = (
                f"Benutzer {user} hat {len(rows)} Buchungen, fehlt in "
                f"Berechtigungsauswertung (n_perm={len(known)})."
            )
            yield Flag(
                lens_id=self.lens_id,
                family=self.family,
                title=title,
                rationale=rationale,
                evidence=(rows[0].source,),
                confidence=0.55,
            )


# --------------------------------------------------------------------------
# Rights violations
# --------------------------------------------------------------------------


class RightsViolation:
    """Actions that require a right the user does not have in the matrix."""

    lens_id = "G_rights_violation"
    family = LensFamily.GRAPH

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        perms = dossier.docs_of("permission")
        if not perms:
            return

        by_user: dict[str, Document] = {}
        for p in perms:
            uid = p.ref or _field(p, "Benutzer", "User", "USER")
            if uid:
                by_user[uid] = p

        def has_right(uid: str, *field_names: str) -> bool | None:
            """True/False if user known; None if unknown (orphan handled elsewhere)."""
            doc = by_user.get(uid)
            if not doc:
                return None
            for fn in field_names:
                raw = _field(doc, fn)
                if not raw:
                    # try fuzzy key match
                    for k, v in doc.fields.items():
                        if fn.casefold() in k.casefold() and v:
                            raw = v
                            break
                val = (raw or "").strip().lower()
                if val in {"x", "ja", "yes", "true", "1", "y"}:
                    return True
            return False

        # journal approvals without freigabe right
        for a in dossier.docs_of("approval"):
            fre = _field(a, "FREIGEBER", "APPROVER", "APPROVED_BY")
            if not fre:
                continue
            ok = has_right(fre, "Journal freigeben", "Freigeben", "Approve", "APPROVE")
            if ok is False:
                title = f"Freigabe ohne Recht: {fre}"
                rationale = (
                    f"Benutzer {fre} freigibt Journal {a.ref}, "
                    f"aber 'Journal freigeben' ist nicht gesetzt."
                )
                perm = by_user[fre]
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=(a.source, perm.source),
                    confidence=0.6,
                )

        # master data changes without stammdaten right
        seen_master: set[tuple[str, str]] = set()
        for m in dossier.docs_of("master_change"):
            u = _field(m, "GEAENDERT_VON", "CHANGED_BY", "USER")
            if not u:
                continue
            ok = has_right(
                u,
                "Stammdaten/Kreditor anlegen",
                "Stammdaten",
                "Master Data",
                "Kreditor anlegen",
            )
            if ok is False:
                key = (u, _field(m, "FELD", "FIELD") or m.ref)
                if key in seen_master:
                    continue
                seen_master.add(key)
                eid = m.entity_id or _field(m, "KONTO")
                title = f"Stammdaten ohne Recht: {u}"
                rationale = (
                    f"Benutzer {u} aendert Stammdaten Konto={eid} "
                    f"Feld={_field(m, 'FELD')!r}, Recht nicht gesetzt."
                )
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=(m.source, by_user[u].source),
                    entity_id=eid or None,
                    confidence=0.55,
                )

        # posting without Buchen right (only if user is in matrix with empty Buchen)
        for uid, pdoc in by_user.items():
            ok = has_right(uid, "Buchen", "Post", "Book", "BUCHEN")
            if ok is not False:
                continue
            rows = [p for p in dossier.postings if p.user == uid]
            if len(rows) < 10:
                continue
            title = f"Buchen ohne Recht: {uid}"
            rationale = (
                f"Benutzer {uid}: {len(rows)} Buchungen, 'Buchen' nicht gesetzt "
                f"in Berechtigungsmatrix."
            )
            yield Flag(
                lens_id=self.lens_id,
                family=self.family,
                title=title,
                rationale=rationale,
                evidence=(rows[0].source, pdoc.source),
                confidence=0.5,
            )


# --------------------------------------------------------------------------
# Shared identity vendor ↔ customer
# --------------------------------------------------------------------------


class SharedIdentity:
    """Same normalized address or VAT id on a vendor and a customer."""

    lens_id = "G_shared_identity"
    family = LensFamily.GRAPH

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        vendors = [e for e in dossier.entities.values() if e.type == EntityType.VENDOR]
        customers = [e for e in dossier.entities.values() if e.type == EntityType.CUSTOMER]
        if not vendors or not customers:
            return

        by_addr: dict[str, list[Entity]] = defaultdict(list)
        by_vat: dict[str, list[Entity]] = defaultdict(list)
        for v in vendors:
            a = _norm_addr(v.address or "")
            if len(a) >= 10:
                by_addr[a].append(v)
            vid = _vat_id(v)
            if len(vid) >= 8:
                by_vat[vid].append(v)

        seen: set[tuple[str, str, str]] = set()
        for c in customers:
            a = _norm_addr(c.address or "")
            if len(a) >= 10 and a in by_addr:
                for v in by_addr[a]:
                    key = ("addr", v.id, c.id)
                    if key in seen:
                        continue
                    seen.add(key)
                    title = f"Gemeinsame Adresse: {v.id} / {c.id}"
                    rationale = (
                        f"Kreditor {v.id} ({v.name}) und Debitor {c.id} ({c.name}) "
                        f"teilen normalisierte Adresse."
                    )
                    yield Flag(
                        lens_id=self.lens_id,
                        family=self.family,
                        title=title,
                        rationale=rationale,
                        evidence=(v.source, c.source),
                        entity_id=v.id,
                        confidence=0.6,
                    )

            vid = _vat_id(c)
            if len(vid) >= 8 and vid in by_vat:
                for v in by_vat[vid]:
                    key = ("vat", v.id, c.id)
                    if key in seen:
                        continue
                    seen.add(key)
                    title = f"Gemeinsame USt-Id: {v.id} / {c.id}"
                    rationale = (
                        f"Kreditor {v.id} und Debitor {c.id} teilen USt-Id {vid}."
                    )
                    yield Flag(
                        lens_id=self.lens_id,
                        family=self.family,
                        title=title,
                        rationale=rationale,
                        evidence=(v.source, c.source),
                        entity_id=v.id,
                        confidence=0.7,
                    )


# --------------------------------------------------------------------------
# Near-duplicate vendor names
# --------------------------------------------------------------------------


class NearDuplicateVendor:
    """Vendor names with rapidfuzz ratio >= 90 plus a second corroborating signal."""

    lens_id = "G_near_vendor"
    family = LensFamily.GRAPH

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        vendors = [e for e in dossier.entities.values() if e.type == EntityType.VENDOR]
        if len(vendors) < 2:
            return

        for i, a in enumerate(vendors):
            for b in vendors[i + 1 :]:
                if not a.name or not b.name:
                    continue
                ratio = fuzz.ratio(a.name.casefold(), b.name.casefold())
                if ratio < 90:
                    continue

                core_a, core_b = _core_name(a.name), _core_name(b.name)
                same_core = bool(core_a) and core_a == core_b
                token_ratio = fuzz.token_sort_ratio(a.name.casefold(), b.name.casefold())
                same_plz = bool(_plz(a.address) and _plz(a.address) == _plz(b.address))
                same_addr = bool(
                    _norm_addr(a.address or "")
                    and _norm_addr(a.address or "") == _norm_addr(b.address or "")
                )
                # second signal required
                if not (same_core or token_ratio >= 95 or same_plz or same_addr):
                    continue

                title = f"Aehnliche Kreditoren: {a.id} / {b.id}"
                rationale = (
                    f"Namen {a.name!r} vs {b.name!r}, fuzz.ratio={ratio:.0f}, "
                    f"token_sort={token_ratio:.0f}, same_core={same_core}, "
                    f"same_plz={same_plz}."
                )
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=(a.source, b.source),
                    entity_id=a.id,
                    confidence=0.45 if ratio < 95 else 0.55,
                )


# --------------------------------------------------------------------------
# Shareholder / related party
# --------------------------------------------------------------------------


class ShareholderLink:
    """Vendor/customer names matching Gesellschafterliste (related party)."""

    lens_id = "G_shareholder_link"
    family = LensFamily.GRAPH

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        holders = dossier.docs_of("shareholder")
        if not holders:
            return

        entities = [
            e
            for e in dossier.entities.values()
            if e.type in {EntityType.VENDOR, EntityType.CUSTOMER}
        ]
        if not entities:
            return

        flagged: set[str] = set()
        for sh in holders:
            name = _field(sh, "NAME", "Name", "GESELLSCHAFTER") or sh.ref
            if not name or len(name) < 4:
                continue
            # explicit kreditor hint in remark (highest confidence)
            remark = _field(sh, "BEMERKUNG", "REMARK", "NOTE", "COMMENT")
            explicit = re.findall(r"\b(20\d{4}|10\d{4})\b", remark)

            for eid in explicit:
                if eid in flagged:
                    continue
                ent = dossier.entities.get(eid)
                if not ent:
                    continue
                flagged.add(eid)
                title = f"Gesellschafter-Personenkonto: {eid}"
                rationale = (
                    f"Gesellschafter {name!r} verweist auf Konto {eid} "
                    f"({ent.name}) in Bemerkung."
                )
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=(sh.source, ent.source),
                    entity_id=eid,
                    confidence=0.7,
                )

            core_sh = _core_name(name)
            if len(core_sh) < 6:
                continue
            for ent in entities:
                if ent.id in flagged:
                    continue
                r = fuzz.partial_ratio(name.casefold(), ent.name.casefold())
                core_e = _core_name(ent.name)
                if r < 90 and core_sh != core_e:
                    continue
                if r < 85:
                    continue
                flagged.add(ent.id)
                title = f"Gesellschafter ~ {ent.type.value} {ent.id}"
                rationale = (
                    f"Gesellschafter {name!r} matched {ent.name!r} "
                    f"(partial_ratio={r:.0f}, core_eq={core_sh == core_e})."
                )
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=(sh.source, ent.source),
                    entity_id=ent.id,
                    confidence=0.55,
                )


register(SelfApproval())
register(OrphanUser())
register(RightsViolation())
register(SharedIdentity())
register(NearDuplicateVendor())
register(ShareholderLink())

"""Reconciliation lenses: documents that contradict each other.

Practice-set flag rates (calibrated post-scoring merge, 26647 postings):
  R_three_way               3  0.01%  GR without invoice (JET_FLOOR)
  R_sales_match             0  0.00%  sales invoice without goods issue
  R_credit_limit            0  0.00%  util > limit / util != saldo
  R_journal_completeness    1  0.00%  Freigabe Zeilen vs GL DOKUMENT
  R_trial_balance           2  0.01%  Saldenliste Soll/Haben vs GL
  R_subledger_tie           0  0.00%  Abstimmung claims vs recomputed
  R_fs_tie                  0  0.00%  Jahresueberschuss Entwurf vs note
  R_bank                    0  0.00%  dormant (no bank docs on practice)
  # all << 2% of rows; no further threshold cuts

Arithmetic findings: confidence 0.55-0.85. Always cite BOTH sides in evidence.
"""

from __future__ import annotations

import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from typing import Iterable

from ..contracts import (
    JET_FLOOR,
    Dossier,
    Document,
    EntityType,
    Flag,
    LensFamily,
    Posting,
    SourceRef,
    register,
)

_TOL = Decimal("1.00")
_VAT_RATES = (Decimal("1.19"), Decimal("1.07"), Decimal("1.00"))


def _field(doc: Document, *names: str) -> str:
    for n in names:
        if n in doc.fields and doc.fields[n]:
            return doc.fields[n]
        for k, v in doc.fields.items():
            if k.lower() == n.lower() and v:
                return v
    return ""


def _parse_amount(raw: str | Decimal | None) -> Decimal | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, Decimal):
        return raw
    s = str(raw).strip().replace("\xa0", "").replace(" ", "")
    if not s:
        return None
    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1]
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        head, _, tail = s.rpartition(",")
        if s.count(",") == 1 and len(tail) <= 2:
            s = head + "." + tail
        else:
            s = s.replace(",", "")
    try:
        d = Decimal(s)
    except InvalidOperation:
        return None
    return -d if neg else d


def _near(a: Decimal, b: Decimal, tol: Decimal = _TOL) -> bool:
    return abs(a - b) <= tol


def _src(doc_or_post: Document | Posting) -> SourceRef:
    return doc_or_post.source


# --------------------------------------------------------------------------
# Three-way: goods receipt ↔ invoice/payment
# --------------------------------------------------------------------------


class ThreeWayMatch:
    """Purchase: GR amount should match a GL line (net) or AP gross via VAT."""

    lens_id = "R_three_way"
    family = LensFamily.RECONCILIATION

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        grs = dossier.docs_of("goods_receipt")
        if not grs:
            return

        by_doc: dict[str, list[Posting]] = defaultdict(list)
        for p in dossier.postings:
            if p.doc_no:
                by_doc[p.doc_no].append(p)

        for g in grs:
            if g.amount is None or abs(g.amount) < JET_FLOOR:
                continue
            inv = _field(g, "RECHNUNGSNUMMER", "INVOICE", "INVOICE_NO", "RECHNUNG")
            amt = abs(g.amount)

            if not inv:
                title = f"Wareneingang ohne Rechnung: {g.ref}"
                rationale = (
                    f"WE {g.ref} Betrag {amt:,.2f} ohne RECHNUNGSNUMMER "
                    f"(Kreditor {g.entity_id}). Drei-Wege-Match unmoeglich."
                )
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=(g.source,),
                    entity_id=g.entity_id,
                    amount=amt,
                    confidence=0.7,
                )
                continue

            posts = by_doc.get(inv, [])
            if not posts:
                title = f"Wareneingang ohne Buchung: {inv}"
                rationale = (
                    f"WE {g.ref} verweist auf Rechnung {inv}, Betrag {amt:,.2f}, "
                    f"aber keine Posting mit diesem Beleg."
                )
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=(g.source,),
                    entity_id=g.entity_id,
                    doc_no=inv,
                    amount=amt,
                    confidence=0.65,
                )
                continue

            matched = False
            for p in posts:
                pa = abs(p.amount)
                if _near(pa, amt):
                    matched = True
                    break
                for rate in _VAT_RATES:
                    if _near(pa, amt * rate, Decimal("2.00")):
                        matched = True
                        break
                if matched:
                    break
            if matched:
                continue

            # material amount break only
            sample = posts[0]
            title = f"Drei-Wege-Betragsbruch: {inv}"
            rationale = (
                f"WE {g.ref}={amt:,.2f} vs Buchungen zu {inv} "
                f"(z.B. {abs(sample.amount):,.2f}, n={len(posts)}). "
                f"Kein Netto- oder Brutto-Match (USt 0/7/19%)."
            )
            yield Flag(
                lens_id=self.lens_id,
                family=self.family,
                title=title,
                rationale=rationale,
                evidence=(g.source, sample.source),
                entity_id=g.entity_id,
                doc_no=inv,
                amount=amt,
                confidence=0.75,
            )


# --------------------------------------------------------------------------
# Sales: invoice ↔ goods issue
# --------------------------------------------------------------------------


class SalesMatch:
    """Sales invoice without goods issue, or amount mismatch (fictitious revenue)."""

    lens_id = "R_sales_match"
    family = LensFamily.RECONCILIATION

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        invoices = dossier.docs_of("sales_invoice")
        issues = dossier.docs_of("goods_issue")
        if not invoices:
            return

        gi_by_inv: dict[str, Document] = {}
        for g in issues:
            inv = _field(g, "RECHNUNGSNUMMER", "INVOICE", "INVOICE_NO") or ""
            if inv:
                gi_by_inv[inv] = g

        for inv in invoices:
            if inv.amount is not None and inv.amount < 0:
                continue  # credit notes often have no GI
            amt = abs(inv.amount) if inv.amount is not None else None
            if amt is not None and amt < JET_FLOOR:
                continue

            gi = gi_by_inv.get(inv.ref)
            if gi is None:
                title = f"Umsatz ohne Warenausgang: {inv.ref}"
                rationale = (
                    f"Verkaufsrechnung {inv.ref} Betrag {amt}, Debitor {inv.entity_id}, "
                    f"kein Warenausgang mit gleicher RECHNUNGSNUMMER."
                )
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=(inv.source,),
                    entity_id=inv.entity_id,
                    doc_no=inv.ref,
                    amount=amt,
                    confidence=0.7,
                )
                continue

            if amt is not None and gi.amount is not None:
                if not _near(amt, abs(gi.amount)):
                    title = f"Faktura/WA Betragsbruch: {inv.ref}"
                    rationale = (
                        f"Rechnung {inv.ref}={amt:,.2f} vs WA {gi.ref}="
                        f"{abs(gi.amount):,.2f}."
                    )
                    yield Flag(
                        lens_id=self.lens_id,
                        family=self.family,
                        title=title,
                        rationale=rationale,
                        evidence=(inv.source, gi.source),
                        entity_id=inv.entity_id,
                        doc_no=inv.ref,
                        amount=amt,
                        confidence=0.8,
                    )


# --------------------------------------------------------------------------
# Credit limits
# --------------------------------------------------------------------------


class CreditLimitCheck:
    """Utilization above limit; utilization vs subledger balance mismatch."""

    lens_id = "R_credit_limit"
    family = LensFamily.RECONCILIATION

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        limits = dossier.docs_of("credit_limit")
        if not limits:
            return

        sl_by: dict[str, Document] = {}
        for s in dossier.docs_of("subledger_balance"):
            if s.entity_id:
                sl_by[s.entity_id] = s

        for doc in limits:
            eid = doc.entity_id or _field(doc, "DEBITOR", "KUNDE", "CUSTOMER", "KONTO")
            limit = _parse_amount(
                _field(doc, "KREDITLIMIT_EUR", "KREDITLIMIT", "CREDIT_LIMIT", "LIMIT")
            )
            util = _parse_amount(
                _field(
                    doc,
                    "AUSNUTZUNG_31_12_2025_EUR",
                    "AUSNUTZUNG",
                    "UTILIZATION",
                    "USAGE",
                    "IN_ANSPRUCH",
                )
            )
            if limit is None:
                limit = doc.amount

            if limit is not None and util is not None and util > limit + _TOL:
                title = f"Kreditlimit ueberschritten: {eid}"
                rationale = (
                    f"Debitor {eid}: Ausnutzung {util:,.2f} > Limit {limit:,.2f} "
                    f"(Delta {util - limit:,.2f})."
                )
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=(doc.source,),
                    entity_id=eid or None,
                    amount=util,
                    confidence=0.8,
                )

            if eid and util is not None and eid in sl_by:
                sl = sl_by[eid]
                bal = sl.amount
                if bal is None:
                    bal = _parse_amount(
                        _field(sl, "Saldo 31.12.2025", "Saldo", "BALANCE", "SALDO")
                    )
                if bal is not None and not _near(abs(util), abs(bal), Decimal("5")):
                    title = f"Ausnutzung != Personenkontosaldo: {eid}"
                    rationale = (
                        f"Debitor {eid}: Kreditlimit-Ausnutzung {util:,.2f} vs "
                        f"Saldo Personenkonto {bal:,.2f}."
                    )
                    yield Flag(
                        lens_id=self.lens_id,
                        family=self.family,
                        title=title,
                        rationale=rationale,
                        evidence=(doc.source, sl.source),
                        entity_id=eid,
                        amount=util,
                        confidence=0.7,
                    )


# --------------------------------------------------------------------------
# Journal completeness vs Freigabe-Log
# --------------------------------------------------------------------------


class JournalCompleteness:
    """Freigabe-Log line counts vs GL lines per JOURNALNAME/DOKUMENT; open status."""

    lens_id = "R_journal_completeness"
    family = LensFamily.RECONCILIATION

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        approvals = dossier.docs_of("approval")
        if not approvals:
            return

        by_journal: dict[str, list[Posting]] = defaultdict(list)
        for p in dossier.postings:
            if p.attrs.get("ledger") and p.attrs.get("ledger") != "GL":
                continue
            j = (
                p.attrs.get("DOKUMENT")
                or p.attrs.get("JOURNALNAME")
                or p.attrs.get("BUCHUNGSNUMMER")
                or ""
            ).strip()
            if j:
                by_journal[j].append(p)

        for a in approvals:
            status = _field(a, "FREIGABESTATUS", "STATUS", "APPROVAL_STATUS").lower()
            if status and status not in {
                "freigegeben",
                "approved",
                "ok",
                "ja",
                "genehmigt",
                "",
            }:
                if any(x in status for x in ("offen", "open", "pending", "abgelehnt", "reject")):
                    title = f"Journal unfreigegeben: {a.ref}"
                    rationale = f"FREIGABESTATUS={status!r}, Journal={_field(a, 'JOURNALNAME')!r}."
                    yield Flag(
                        lens_id=self.lens_id,
                        family=self.family,
                        title=title,
                        rationale=rationale,
                        evidence=(a.source,),
                        confidence=0.65,
                    )

            jn = _field(a, "JOURNALNAME", "JOURNAL", "NAME")
            claimed_s = _field(a, "ANZAHL_ZEILEN", "LINE_COUNT", "ZEILEN", "LINES")
            try:
                claimed = int(float(claimed_s)) if claimed_s else None
            except ValueError:
                claimed = None
            if not jn or claimed is None:
                continue
            rows = by_journal.get(jn, [])
            actual = len(rows)
            if actual == 0:
                continue
            # only large breaks (practice freigabe often claims 2 for multi-line journals)
            if actual > max(claimed * 5, claimed + 20):
                title = f"Freigabe Zeilenzahl falsch: {jn}"
                rationale = (
                    f"Journal {jn}: Freigabe-Log ANZAHL_ZEILEN={claimed}, "
                    f"GL-Zeilen zu DOKUMENT={actual}."
                )
                ev = (a.source,)
                if rows:
                    ev = (a.source, rows[0].source)
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=ev,
                    confidence=0.6,
                )


# --------------------------------------------------------------------------
# Trial balance vs GL
# --------------------------------------------------------------------------


class TrialBalanceVsGL:
    """Saldenliste Soll/Haben vs sum of GL postings per account."""

    lens_id = "R_trial_balance"
    family = LensFamily.RECONCILIATION

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        tbs = dossier.docs_of("trial_balance")
        if not tbs:
            return

        gl_rows: dict[str, list[Posting]] = defaultdict(list)
        for p in dossier.postings:
            if p.attrs.get("ledger") == "GL" or (
                p.attrs.get("ledger") is None and p.account
            ):
                # prefer pure GL; skip AP/AR subledgers
                if p.attrs.get("ledger") in {"AP", "AR", "FA"}:
                    continue
                gl_rows[p.account].append(p)

        if not gl_rows:
            return

        for t in tbs:
            acct = t.ref or _field(t, "Konto", "Account", "SACHKONTONUMMER")
            if not acct:
                continue
            rows = gl_rows.get(acct, [])
            if not rows:
                # control accounts often only on subaccount form — skip silently
                continue

            soll = _parse_amount(_field(t, "Soll 2025", "Soll", "DEBIT", "Sollumsatz")) or Decimal(
                0
            )
            haben = _parse_amount(
                _field(t, "Haben 2025", "Haben", "CREDIT", "Habenumsatz")
            ) or Decimal(0)
            s_pos = sum((p.amount for p in rows if p.amount > 0), Decimal(0))
            s_neg = sum((-p.amount for p in rows if p.amount < 0), Decimal(0))

            # material break only
            d_soll = abs(s_pos - soll)
            d_haben = abs(s_neg - haben)
            if d_soll <= Decimal("100") and d_haben <= Decimal("100"):
                continue
            if d_soll < JET_FLOOR and d_haben < JET_FLOOR:
                continue

            title = f"Saldenliste != GL: Konto {acct}"
            rationale = (
                f"Konto {acct}: Saldenliste Soll={soll:,.2f}/Haben={haben:,.2f}, "
                f"GL pos={s_pos:,.2f}/neg={s_neg:,.2f}, "
                f"Delta Soll={d_soll:,.2f} Haben={d_haben:,.2f}, n={len(rows)}."
            )
            yield Flag(
                lens_id=self.lens_id,
                family=self.family,
                title=title,
                rationale=rationale,
                evidence=(t.source, rows[0].source),
                amount=max(d_soll, d_haben),
                confidence=0.75,
            )


# --------------------------------------------------------------------------
# Sub-ledger / company recon statement ties
# --------------------------------------------------------------------------


class SubledgerTie:
    """Recompute AR/AP subledger totals vs Abstimmung claims."""

    lens_id = "R_subledger_tie"
    family = LensFamily.RECONCILIATION

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        sl = dossier.docs_of("subledger_balance")
        recon = dossier.docs_of("reconciliation_statement")
        if not sl and not recon:
            return

        ar_total = Decimal(0)
        ap_total = Decimal(0)
        ar_src = ap_src = None
        for s in sl:
            eid = s.entity_id or _field(s, "Konto", "Account")
            bal = s.amount
            if bal is None:
                bal = _parse_amount(_field(s, "Saldo 31.12.2025", "Saldo", "BALANCE"))
            if bal is None or not eid:
                continue
            if eid.startswith("1") or (
                eid in dossier.entities
                and dossier.entities[eid].type == EntityType.CUSTOMER
            ):
                ar_total += bal
                ar_src = ar_src or s
            elif eid.startswith("2") or (
                eid in dossier.entities
                and dossier.entities[eid].type == EntityType.VENDOR
            ):
                ap_total += bal
                ap_src = ap_src or s

        claims: dict[str, tuple[Decimal, Document]] = {}
        for r in recon:
            label = (_field(r, "label", "LABEL") or r.ref or "").lower()
            val = _parse_amount(_field(r, "value", "VALUE", "BETRAG"))
            if val is None:
                continue
            if "differenz debitor" in label or "differenz forderung" in label:
                claims["diff_ar"] = (val, r)
            elif "differenz kreditor" in label or "differenz verbind" in label:
                claims["diff_ap"] = (val, r)
            elif "op-liste debitor" in label or "personenkonten laut op-liste debitor" in label:
                claims["op_ar"] = (val, r)
            elif "op-liste kreditor" in label or "personenkonten laut op-liste kreditor" in label:
                claims["op_ap"] = (val, r)
            elif "hb-konto" in label and ("230" in label or "forderung" in label):
                claims["hb_ar"] = (val, r)
            elif "hb-konto" in label and ("330" in label or "verbind" in label):
                claims["hb_ap"] = (val, r)

        # company claims non-zero difference
        for key, title_de in (("diff_ar", "Debitoren"), ("diff_ap", "Kreditoren")):
            if key not in claims:
                continue
            val, doc = claims[key]
            if abs(val) > _TOL:
                title = f"Abstimmung meldet Differenz {title_de}"
                rationale = f"Abstimmung '{_field(doc, 'label')}' = {val:,.2f} (sollte 0 sein)."
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=(doc.source,),
                    amount=abs(val),
                    confidence=0.85,
                )

        # recompute vs claimed OP / HB
        if ar_src and "op_ar" in claims:
            claimed, doc = claims["op_ar"]
            if not _near(ar_total, claimed, Decimal("100")):
                title = "Debitoren-Personenkonten != Abstimmung"
                rationale = (
                    f"Summe subledger_balance Debitoren={ar_total:,.2f}, "
                    f"Abstimmung OP-Liste={claimed:,.2f}, Delta={ar_total - claimed:,.2f}."
                )
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=(ar_src.source, doc.source),
                    amount=abs(ar_total - claimed),
                    confidence=0.8,
                )

        if ap_src and "op_ap" in claims:
            claimed, doc = claims["op_ap"]
            if not _near(ap_total, claimed, Decimal("100")):
                title = "Kreditoren-Personenkonten != Abstimmung"
                rationale = (
                    f"Summe subledger_balance Kreditoren={ap_total:,.2f}, "
                    f"Abstimmung OP-Liste={claimed:,.2f}, Delta={ap_total - claimed:,.2f}."
                )
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=(ap_src.source, doc.source),
                    amount=abs(ap_total - claimed),
                    confidence=0.8,
                )

        # claimed HB vs claimed OP should match (company internal)
        if "op_ar" in claims and "hb_ar" in claims:
            a, da = claims["op_ar"]
            b, db = claims["hb_ar"]
            if not _near(a, b, Decimal("1")):
                title = "Abstimmung Debitoren OP != HB"
                rationale = f"OP-Liste={a:,.2f} vs HB={b:,.2f}."
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=(da.source, db.source),
                    amount=abs(a - b),
                    confidence=0.85,
                )


# --------------------------------------------------------------------------
# FS net income vs recon note
# --------------------------------------------------------------------------


class FinancialStatementTie:
    """Jahresueberschuss in Entwurf vs Abstimmung / simple P&L check."""

    lens_id = "R_fs_tie"
    family = LensFamily.RECONCILIATION

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        fs_docs = dossier.docs_of("financial_statements")
        recon = dossier.docs_of("reconciliation_statement")
        if not fs_docs and not recon:
            return

        fs_amt = None
        fs_src = None
        for f in fs_docs:
            text = _field(f, "text", "TEXT") or f.fields.get("text", "")
            m = re.search(
                r"Jahres[uü]berschuss[^\d]{0,40}([\d.]+,\d{2}|\d+\.\d{2})",
                text,
                re.I,
            )
            if m:
                fs_amt = _parse_amount(m.group(1))
                fs_src = f
                break
            # also EN
            m = re.search(
                r"(?:net income|annual (?:surplus|profit))[^\d]{0,40}([\d,]+(?:\.\d+)?)",
                text,
                re.I,
            )
            if m:
                fs_amt = _parse_amount(m.group(1))
                fs_src = f
                break

        recon_amt = None
        recon_src = None
        for r in recon:
            label = (_field(r, "label") or "").lower()
            if "jahres" in label and ("entwurf" in label or "ueberschuss" in label or "überschuss" in label):
                recon_amt = _parse_amount(_field(r, "value"))
                recon_src = r
                break

        if fs_amt is not None and recon_amt is not None and not _near(fs_amt, recon_amt, Decimal("1")):
            title = "Jahresueberschuss Entwurf != Abstimmung"
            rationale = (
                f"JA-Entwurf {fs_amt:,.2f} vs Abstimmung {recon_amt:,.2f}, "
                f"Delta={fs_amt - recon_amt:,.2f}."
            )
            yield Flag(
                lens_id=self.lens_id,
                family=self.family,
                title=title,
                rationale=rationale,
                evidence=(fs_src.source, recon_src.source),
                amount=abs(fs_amt - recon_amt),
                confidence=0.85,
            )


# --------------------------------------------------------------------------
# Bank ↔ ledger (dormant on practice set)
# --------------------------------------------------------------------------


class BankReconciliation:
    """Bank confirmation/statement vs bank-class GL. No-op when docs absent."""

    lens_id = "R_bank"
    family = LensFamily.RECONCILIATION

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        confs = dossier.docs_of("bank_confirmation")
        stmts = dossier.docs_of("bank_statement")
        if not confs and not stmts:
            return

        # bank postings: entity type account named bank-like, or attrs ledger bank
        bank_posts = [
            p
            for p in dossier.postings
            if "bank" in (p.account or "").lower()
            or "bank" in (p.text or "").lower()
            or (p.attrs.get("ledger") or "").upper() == "BANK"
        ]
        # also use classify if available
        try:
            from ..ingest.accounts import AccountClass, classify_account

            for p in dossier.postings:
                if p in bank_posts:
                    continue
                ent = dossier.entities.get(p.account)
                name = ent.name if ent else ""
                if classify_account(p.account, name, ent.attrs if ent else {}) == AccountClass.BANK:
                    bank_posts.append(p)
        except Exception:
            pass

        bank_bal = sum((p.amount for p in bank_posts), Decimal(0)) if bank_posts else None

        for c in confs:
            confirmed = c.amount or _parse_amount(
                _field(c, "SALDO", "BALANCE", "CONFIRMED", "BETRAG", "AMOUNT")
            )
            if confirmed is None:
                continue
            if bank_bal is not None and not _near(confirmed, bank_bal, Decimal("100")):
                title = f"Bankbestaetigung != Ledger: {c.ref}"
                rationale = (
                    f"Bestaetigter Saldo {confirmed:,.2f} vs Bank-GL Summe "
                    f"{bank_bal:,.2f}, Delta={confirmed - bank_bal:,.2f}."
                )
                ev = (c.source,)
                if bank_posts:
                    ev = (c.source, bank_posts[0].source)
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=ev,
                    amount=abs(confirmed - bank_bal),
                    confidence=0.8,
                )

        # statement lines without ledger counterpart (amount+date window)
        if not stmts or not dossier.postings:
            return
        post_by_amt: dict[Decimal, list[Posting]] = defaultdict(list)
        for p in dossier.postings:
            post_by_amt[abs(p.amount)].append(p)

        for s in stmts:
            amt = s.amount or _parse_amount(_field(s, "BETRAG", "AMOUNT", "VALUE", "UMSATZ"))
            if amt is None or abs(amt) < Decimal("1000"):
                continue
            cands = post_by_amt.get(abs(amt), [])
            if s.doc_date:
                cands = [
                    p
                    for p in cands
                    if abs((p.booking_date - s.doc_date).days) <= 5
                ]
            if not cands:
                title = f"Bankumsatz ohne Ledger: {s.ref}"
                rationale = (
                    f"Kontoauszug {s.ref} Betrag {abs(amt):,.2f} Datum {s.doc_date} "
                    f"ohne passende Buchung (Betrag+5d)."
                )
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=(s.source,),
                    amount=abs(amt),
                    confidence=0.6,
                )


register(ThreeWayMatch())
register(SalesMatch())
register(CreditLimitCheck())
register(JournalCompleteness())
register(TrialBalanceVsGL())
register(SubledgerTie())
register(FinancialStatementTie())
register(BankReconciliation())

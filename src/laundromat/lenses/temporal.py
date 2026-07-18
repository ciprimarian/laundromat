"""Temporal lenses: lag, off-hours, master-data timing, velocity, sequence, approval.

Practice-set flag rates (after excluding opening balances / Vortrag):
  T_backdating          0  0.00%  (was Saldenvortrag lag; gone)
  T_off_hours           0  0.00%
  T_master_timing       8  0.03%
  T_velocity_burst      0  0.00%  (was MV-U02 Vortrag day)
  T_sequence_gap        0  0.00%
  T_approval_timing     0  0.00%
  # AB-2024 dropped; all << 2% of rows

Baselines (working hours, lag tail, velocity median) always come from the dossier.
Never hardcode 09-18 or FY2025. Confidence 0.2-0.5 for distributional signals.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal
from statistics import median
from typing import Iterable, Sequence

from ..contracts import Dossier, Document, Flag, LensFamily, Posting, SourceRef, register

_MIN_LAG_SAMPLE = 50
_MIN_HOUR_USER = 40
_MIN_VELOCITY_DAYS = 10
_MIN_SEQUENCE_N = 20
_MAX_EVIDENCE = 8
_MASTER_PAY_WINDOW = 7  # days: change then payment
_REVERT_WINDOW = 30

_OPENING_TEXT_PREFIXES = (
    "saldenvortrag",
    "opening balance",
    "brought forward",
    "balance brought forward",
    "bfwd",
    "b/f",
)


def _is_opening_balance(p: Posting) -> bool:
    """True for carry-forward / Vortrag rows (DE+EN). Not economic activity."""
    attrs = p.attrs or {}
    for key in ("BUCHUNGSTYP", "BUCHUNGSART", "PERIODENZUGEHÖRIGKEIT", "PERIODENZUGEHOERIGKEIT"):
        raw = (attrs.get(key) or "").casefold()
        if "vortrag" in raw or "opening" in raw or "brought forward" in raw:
            return True
    text = (p.text or "").casefold().strip()
    if any(text.startswith(pref) for pref in _OPENING_TEXT_PREFIXES):
        return True
    if "saldenvortrag" in text[:40]:
        return True
    return False


def _economic_postings(dossier: Dossier) -> list[Posting]:
    return [p for p in dossier.postings if not _is_opening_balance(p)]


def _sample_evidence(postings: Sequence[Posting], limit: int = _MAX_EVIDENCE) -> tuple[SourceRef, ...]:
    if not postings:
        return (SourceRef(file="(none)", excerpt="no rows"),)
    step = max(1, len(postings) // limit)
    return tuple(p.source for p in list(postings[::step][:limit]))


def _percentile(sorted_vals: Sequence[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 1:
        return float(sorted_vals[-1])
    idx = int(p * (len(sorted_vals) - 1))
    return float(sorted_vals[idx])


def _parse_doc_date(doc: Document) -> date | None:
    if doc.doc_date:
        return doc.doc_date
    for key in ("DATUM", "DATE", "FREIGABEDATUM", "ERFASST_AM", "APPROVAL_DATE"):
        raw = doc.fields.get(key)
        if not raw:
            continue
        d = _try_parse_date(raw)
        if d:
            return d
    return None


def _try_parse_date(raw: str) -> date | None:
    raw = raw.strip()
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _try_parse_datetime(date_s: str, time_s: str | None = None) -> datetime | None:
    date_s = (date_s or "").strip()
    time_s = (time_s or "").strip()
    if not date_s:
        return None
    if time_s:
        for fmt in (
            "%d.%m.%Y %H:%M:%S",
            "%d.%m.%Y %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%d/%m/%Y %H:%M:%S",
        ):
            try:
                return datetime.strptime(f"{date_s} {time_s}", fmt)
            except ValueError:
                continue
    d = _try_parse_date(date_s)
    if d:
        return datetime(d.year, d.month, d.day)
    return None


def _field(doc: Document, *names: str) -> str:
    for n in names:
        if n in doc.fields and doc.fields[n]:
            return doc.fields[n]
        # case-insensitive
        for k, v in doc.fields.items():
            if k.lower() == n.lower() and v:
                return v
    return ""


def _truthy_yes(val: str) -> bool:
    v = (val or "").strip().lower()
    return v in {"ja", "yes", "true", "1", "y", "j", "freigegeben", "approved"}


def _hour_float(dt: datetime) -> float:
    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0


# --------------------------------------------------------------------------
# Backdating / posting lag
# --------------------------------------------------------------------------


class BackdatingLag:
    """Flag booking→posting lags in the extreme tail of the dossier's own distribution."""

    lens_id = "T_backdating"
    family = LensFamily.TEMPORAL

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        lags: list[tuple[int, Posting]] = []
        for p in _economic_postings(dossier):
            if p.posted_at is None:
                continue
            lag = (p.posted_at.date() - p.booking_date).days
            lags.append((lag, p))

        if len(lags) < _MIN_LAG_SAMPLE:
            return

        lag_vals = sorted(lag for lag, _ in lags)
        # tail cut from data; also require a meaningful absolute lag
        p99 = _percentile(lag_vals, 0.99)
        p995 = _percentile(lag_vals, 0.995)
        med = median(lag_vals)
        # cut at least a few days above median, and at the 99th percentile floor
        cut = max(float(med) + 5.0, p99, 3.0)
        # if almost everything is same-day (practice set), p99 may be 0 —
        # then any lag >= 5 is already the interesting tail
        if p99 <= 0:
            cut = max(5.0, p995 if p995 > 0 else 5.0)

        for lag, p in lags:
            if lag < cut:
                continue
            conf = 0.25
            if lag >= cut * 2:
                conf = 0.35
            if lag >= 15:
                conf = 0.4

            title = f"Rueckdatierung: Buchung {p.booking_date} erfasst +{lag}d"
            rationale = (
                f"Lag booking→posted = {lag} Tage "
                f"(Schwelle {cut:.0f}d aus p99={p99:.0f}, Median={med:.0f}). "
                f"Benutzer={p.user!r}, Beleg={p.doc_no!r}."
            )
            yield Flag(
                lens_id=self.lens_id,
                family=self.family,
                title=title,
                rationale=rationale,
                evidence=(p.source,),
                entity_id=p.entity_id,
                doc_no=p.doc_no or None,
                amount=abs(p.amount) if p.amount else None,
                confidence=conf,
            )


# --------------------------------------------------------------------------
# Off-hours / per-user deviation (never hardcode 9-18)
# --------------------------------------------------------------------------


class OffHours:
    """Flag entry times outside the dossier-derived and per-user working-hour norms.

    Does not flag Admin merely for being Admin. Per-user deviation is the signal.
    """

    lens_id = "T_off_hours"
    family = LensFamily.TEMPORAL

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        by_user: dict[str, list[tuple[datetime, Posting]]] = defaultdict(list)
        all_hours: list[float] = []
        econ = _economic_postings(dossier)

        for p in econ:
            if p.posted_at is None or not p.user:
                continue
            by_user[p.user].append((p.posted_at, p))
            all_hours.append(_hour_float(p.posted_at))

        if len(all_hours) < _MIN_HOUR_USER:
            return

        all_hours.sort()
        # core working band from the bulk of the dossier (p5-p95), not p1/p99
        # which is contaminated by the outliers we are hunting
        g_lo = _percentile(all_hours, 0.05)
        g_hi = _percentile(all_hours, 0.95)
        # weekend rate for the whole ledger
        weekend_flags = 0
        weekend_total = 0
        for p in econ:
            if p.posted_at is None:
                continue
            weekend_total += 1
            if p.posted_at.weekday() >= 5:
                weekend_flags += 1
        global_weekend_rate = weekend_flags / weekend_total if weekend_total else 0.0

        for user, items in by_user.items():
            if len(items) < _MIN_HOUR_USER:
                continue
            hours = sorted(_hour_float(dt) for dt, _ in items)
            u_lo = _percentile(hours, 0.05)
            u_hi = _percentile(hours, 0.95)
            # must clear BOTH: far outside this user's band and outside global core
            user_margin = 2.0
            global_margin = 2.0

            user_weekend = sum(1 for dt, _ in items if dt.weekday() >= 5)
            user_weekend_rate = user_weekend / len(items)

            for dt, p in items:
                h = _hour_float(dt)
                outside_user = h < (u_lo - user_margin) or h > (u_hi + user_margin)
                outside_global = h < (g_lo - global_margin) or h > (g_hi + global_margin)
                is_hour_outlier = outside_user and outside_global
                # weekend only if this user almost never posts weekends but this one is,
                # and global weekend rate is low (otherwise weekends are normal ops)
                is_weekend_outlier = (
                    dt.weekday() >= 5
                    and user_weekend_rate < 0.05
                    and global_weekend_rate < 0.10
                    and len(items) >= 80
                )

                if not is_hour_outlier and not is_weekend_outlier:
                    continue

                conf = 0.25
                if is_hour_outlier and (h < 5 or h >= 23):
                    conf = 0.4
                elif is_hour_outlier:
                    conf = 0.3

                kind = "Wochenende" if is_weekend_outlier and not is_hour_outlier else "Uhrzeit"
                title = f"Ungewoehnliche Erfassungszeit ({kind}): {user} {dt.strftime('%H:%M')}"
                rationale = (
                    f"Benutzer {user}: Erfassung {dt.isoformat(sep=' ', timespec='minutes')}, "
                    f"Stunde={h:.2f}. User-Norm p5-p95=[{u_lo:.1f},{u_hi:.1f}], "
                    f"global p5-p95=[{g_lo:.1f},{g_hi:.1f}], "
                    f"Weekend-Rate user={user_weekend_rate:.1%} global={global_weekend_rate:.1%}."
                )
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=(p.source,),
                    entity_id=p.entity_id,
                    doc_no=p.doc_no or None,
                    amount=abs(p.amount) if p.amount else None,
                    confidence=conf,
                )


# --------------------------------------------------------------------------
# Master-data change timing
# --------------------------------------------------------------------------


class MasterDataTiming:
    """Change → payment → (optional) revert. Self-approval and unapproved changes."""

    lens_id = "T_master_timing"
    family = LensFamily.TEMPORAL

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        changes = dossier.docs_of("master_change")
        if not changes:
            return

        # index payments by entity_id (skip opening balances)
        pays: dict[str, list[Posting]] = defaultdict(list)
        for p in _economic_postings(dossier):
            if p.entity_id:
                pays[p.entity_id].append(p)

        # group changes by account for reversion detection
        by_account_field: dict[tuple[str, str], list[Document]] = defaultdict(list)

        for doc in changes:
            konto = _field(doc, "KONTO", "ACCOUNT", "CREDITOR", "KREDITOR", "DEBITOR")
            feld = _field(doc, "FELD", "FIELD", "ATTRIBUTE")
            eid = doc.entity_id or konto
            changed_by = _field(doc, "GEAENDERT_VON", "CHANGED_BY", "USER")
            approved_by = _field(doc, "GENEHMIGT_VON", "APPROVED_BY")
            genehmigt = _field(doc, "GENEHMIGT", "APPROVED", "STATUS")
            d = _parse_doc_date(doc)

            if konto and feld:
                by_account_field[(konto, feld)].append(doc)

            # self-approval
            if changed_by and approved_by and changed_by == approved_by:
                title = f"Stammdaten: Selbstfreigabe {eid or konto}"
                rationale = (
                    f"GEAENDERT_VON == GENEHMIGT_VON ({changed_by}), "
                    f"Feld={feld!r}, Konto={konto!r}, Datum={d}."
                )
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=(doc.source,),
                    entity_id=eid or None,
                    confidence=0.4,
                )

            # not approved
            if genehmigt and not _truthy_yes(genehmigt):
                title = f"Stammdaten ungenehmigt: {eid or konto}"
                rationale = (
                    f"GENEHMIGT={genehmigt!r} (nicht Ja/Yes), "
                    f"Feld={feld!r}, Konto={konto!r}, Datum={d}."
                )
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=(doc.source,),
                    entity_id=eid or None,
                    confidence=0.35,
                )

            # change shortly before a payment to the same account
            if not d or not eid:
                continue
            related = pays.get(eid, [])
            near_pays = [
                p
                for p in related
                if 0 <= (p.booking_date - d).days <= _MASTER_PAY_WINDOW
                and abs(p.amount) > 0
            ]
            if not near_pays:
                continue
            # material enough
            near_pays.sort(key=lambda p: abs(p.amount), reverse=True)
            top = near_pays[0]
            if abs(top.amount) < Decimal("1000"):
                continue

            title = f"Stammdaten-Aenderung vor Zahlung: {eid}"
            rationale = (
                f"Aenderung am {d} (Feld={feld!r}) und Zahlung {abs(top.amount):,.2f} "
                f"am {top.booking_date} (Abstand {(top.booking_date - d).days}d, "
                f"Fenster {_MASTER_PAY_WINDOW}d). "
                f"{len(near_pays)} Zahlung(en) im Fenster."
            )
            yield Flag(
                lens_id=self.lens_id,
                family=self.family,
                title=title,
                rationale=rationale,
                evidence=(doc.source, top.source),
                entity_id=eid,
                doc_no=top.doc_no or None,
                amount=abs(top.amount),
                confidence=0.35,
            )

        # reversions: same account+field changed twice within REVERT_WINDOW
        for (konto, feld), docs in by_account_field.items():
            dated = []
            for doc in docs:
                d = _parse_doc_date(doc)
                if d:
                    dated.append((d, doc))
            if len(dated) < 2:
                continue
            dated.sort(key=lambda x: x[0])
            for i in range(len(dated) - 1):
                d0, doc0 = dated[i]
                d1, doc1 = dated[i + 1]
                if (d1 - d0).days > _REVERT_WINDOW:
                    continue
                # only care if there was a payment between them
                eid = doc0.entity_id or konto
                mid_pays = [
                    p
                    for p in pays.get(eid, [])
                    if d0 <= p.booking_date <= d1 and abs(p.amount) > 0
                ]
                if not mid_pays:
                    continue
                mid_pays.sort(key=lambda p: abs(p.amount), reverse=True)
                top = mid_pays[0]
                title = f"Stammdaten-Reversion mit Zahlung: {eid}"
                rationale = (
                    f"Feld {feld!r} Konto {konto} geaendert {d0} und erneut {d1} "
                    f"({(d1 - d0).days}d). Dazwischen Zahlung {abs(top.amount):,.2f}."
                )
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=(doc0.source, doc1.source, top.source),
                    entity_id=eid,
                    doc_no=top.doc_no or None,
                    amount=abs(top.amount),
                    confidence=0.5,
                )


# --------------------------------------------------------------------------
# Velocity / burst detection
# --------------------------------------------------------------------------


class VelocityBurst:
    """Sudden spikes in daily posting volume per user/entity; year-end value clustering."""

    lens_id = "T_velocity_burst"
    family = LensFamily.TEMPORAL

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        econ = _economic_postings(dossier)
        if not econ:
            return

        # --- daily count spikes per user ---
        by_user_day: dict[str, dict[date, list[Posting]]] = defaultdict(lambda: defaultdict(list))
        for p in econ:
            if not p.user:
                continue
            day = p.posted_at.date() if p.posted_at else p.booking_date
            by_user_day[p.user][day].append(p)

        for user, days in by_user_day.items():
            if len(days) < _MIN_VELOCITY_DAYS:
                continue
            counts = [len(v) for v in days.values()]
            med = median(counts)
            mad = median([abs(c - med) for c in counts]) or 1.0
            # strict: median + 8*MAD and absolute floor
            thr = max(med + 8 * mad, med * 4, 30)
            for day, rows in days.items():
                n = len(rows)
                if n < thr:
                    continue
                conf = 0.3
                if n >= thr * 1.5:
                    conf = 0.4
                title = f"Buchungsburst: {user} am {day} ({n} Zeilen)"
                rationale = (
                    f"Benutzer {user}: {n} Erfassungen am {day}, "
                    f"Median/Tag={med:.0f}, MAD={mad:.0f}, Schwelle={thr:.0f}."
                )
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=_sample_evidence(rows),
                    confidence=conf,
                )

        # --- year-end value clustering per entity ---
        by_entity: dict[str, list[Posting]] = defaultdict(list)
        for p in econ:
            if p.entity_id:
                by_entity[p.entity_id].append(p)

        all_dates = [p.booking_date for p in econ]
        if not all_dates:
            return
        max_d = max(all_dates)
        # last 7 days of the max calendar year present in data
        year_end = date(max_d.year, 12, 31)
        window_start = year_end - timedelta(days=6)
        # only apply if dossier reaches into that window
        if max_d < window_start:
            return

        for eid, rows in by_entity.items():
            if len(rows) < 20:
                continue
            total = sum(abs(p.amount) for p in rows)
            if total <= 0:
                continue
            ye_rows = [p for p in rows if window_start <= p.booking_date <= year_end]
            ye_val = sum(abs(p.amount) for p in ye_rows)
            share = float(ye_val / total)
            # 7/365 ~ 1.9%; flag only when share is extreme
            if share < 0.35 or ye_val < Decimal("25000"):
                continue
            title = f"Jahresend-Konzentration: {eid}"
            rationale = (
                f"Entity {eid}: {share:.1%} des Jahreswerts "
                f"({ye_val:,.2f} / {total:,.2f}) in {window_start}..{year_end}, "
                f"n_year_end={len(ye_rows)}/{len(rows)}."
            )
            yield Flag(
                lens_id=self.lens_id,
                family=self.family,
                title=title,
                rationale=rationale,
                evidence=_sample_evidence(ye_rows or rows),
                entity_id=eid,
                amount=ye_val,
                confidence=0.35,
            )


# --------------------------------------------------------------------------
# Sequence gaps
# --------------------------------------------------------------------------


class SequenceGaps:
    """Gaps in ERFASSUNGSNUMMER / JOURNALZEILE; out-of-order timestamps in one journal."""

    lens_id = "T_sequence_gap"
    family = LensFamily.TEMPORAL

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        econ = _economic_postings(dossier)
        if not econ:
            return

        # collect numeric sequences from attrs
        erf_map: dict[int, list[Posting]] = defaultdict(list)
        journal_lines: dict[str, list[tuple[int, Posting]]] = defaultdict(list)

        for p in econ:
            attrs = p.attrs or {}
            erf = attrs.get("ERFASSUNGSNUMMER") or attrs.get("ENTRY_NO") or attrs.get("erfassungsnummer")
            jline = attrs.get("JOURNALZEILE") or attrs.get("JOURNAL_LINE") or attrs.get("journalzeile")
            jname = attrs.get("JOURNALNAME") or attrs.get("JOURNAL") or attrs.get("BUCHUNGSNUMMER") or ""

            if erf and str(erf).strip().isdigit():
                erf_map[int(str(erf).strip())].append(p)
            if jline and str(jline).strip().isdigit() and jname:
                journal_lines[str(jname)].append((int(str(jline).strip()), p))

        # --- ERFASSUNGSNUMMER gaps ---
        if len(erf_map) >= _MIN_SEQUENCE_N:
            nums = sorted(erf_map.keys())
            steps = [nums[i + 1] - nums[i] for i in range(len(nums) - 1)]
            # typical step includes the 1-steps; do not compute median on gaps alone
            med_step = median(steps) if steps else 1
            mad_step = median([abs(s - med_step) for s in steps]) or 1
            # large jump: far above normal numbering step
            cut = max(10, int(med_step + 10 * mad_step), int(med_step * 10))
            for i, g in enumerate(steps):
                if g < cut:
                    continue
                a, b = nums[i], nums[i + 1]
                rows = erf_map[a] + erf_map[b]
                title = f"Luecke in Erfassungsnummern: {a} → {b}"
                rationale = (
                    f"ERFASSUNGSNUMMER springt von {a} auf {b} (Schrittweite {g}), "
                    f"Schwelle {cut}, Median-Schrittweite {med_step}."
                )
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=_sample_evidence(rows),
                    confidence=0.3,
                )

        # --- out-of-order timestamps within a journal ---
        for jname, lines in journal_lines.items():
            if len(lines) < 5:
                continue
            lines = sorted(lines, key=lambda x: x[0])
            prev_ts: datetime | None = None
            prev_p: Posting | None = None
            disorder = []
            for _, p in lines:
                if p.posted_at is None:
                    continue
                if prev_ts is not None and p.posted_at < prev_ts - timedelta(minutes=5):
                    disorder.append((prev_p, p))
                prev_ts = p.posted_at
                prev_p = p
            # multi-day spread of one journal
            times = [p.posted_at for _, p in lines if p.posted_at is not None]
            if len(times) >= 5:
                span = max(times) - min(times)
                if span.days >= 2:
                    rows = [p for _, p in lines]
                    title = f"Journal ueber mehrere Tage erfasst: {jname}"
                    rationale = (
                        f"Journal {jname!r}: Erfassungen spannen {span.days} Tage "
                        f"({min(times).date()} .. {max(times).date()}), n={len(times)}."
                    )
                    yield Flag(
                        lens_id=self.lens_id,
                        family=self.family,
                        title=title,
                        rationale=rationale,
                        evidence=_sample_evidence(rows),
                        confidence=0.3,
                    )

            for a, b in disorder[:5]:
                if a is None:
                    continue
                title = f"Journal zeitlich unsortiert: {jname}"
                rationale = (
                    f"Journal {jname!r}: Zeile mit {b.posted_at} liegt vor "
                    f"vorheriger Zeile {a.posted_at}."
                )
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=(a.source, b.source),
                    confidence=0.35,
                )


# --------------------------------------------------------------------------
# Approval timing
# --------------------------------------------------------------------------


class ApprovalTiming:
    """Implausibly fast / pre-entry / missing approvals from Freigabe-Log."""

    lens_id = "T_approval_timing"
    family = LensFamily.TEMPORAL

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        approvals = dossier.docs_of("approval")
        if not approvals:
            return

        for doc in approvals:
            erfasst_am = _field(doc, "ERFASST_AM", "CREATED_ON", "ENTRY_DATE")
            erfasst_um = _field(doc, "ERFASST_UM", "CREATED_AT", "ENTRY_TIME")
            freigabe = _field(doc, "FREIGABEDATUM", "APPROVAL_DATE", "APPROVED_ON")
            freigabe_um = _field(doc, "FREIGABE_UM", "APPROVAL_TIME", "APPROVED_AT")
            status = _field(doc, "FREIGABESTATUS", "STATUS", "APPROVAL_STATUS")
            ersteller = _field(doc, "ERSTELLER", "CREATOR", "CREATED_BY")
            freigeber = _field(doc, "FREIGEBER", "APPROVER", "APPROVED_BY")

            created = _try_parse_datetime(erfasst_am, erfasst_um) if erfasst_am else None
            if created is None and doc.doc_date:
                created = datetime(doc.doc_date.year, doc.doc_date.month, doc.doc_date.day)

            approved = _try_parse_datetime(freigabe, freigabe_um) if freigabe else None

            # never approved / rejected
            status_l = status.strip().lower()
            if status_l and status_l not in {
                "freigegeben",
                "approved",
                "ok",
                "yes",
                "ja",
                "genehmigt",
            }:
                # open / rejected / missing
                if any(
                    x in status_l
                    for x in ("offen", "open", "pending", "abgelehnt", "rejected", "nein", "no")
                ):
                    title = f"Journal ohne Freigabe: {doc.ref}"
                    rationale = (
                        f"FREIGABESTATUS={status!r}, Journal/Ref={doc.ref!r}, "
                        f"Ersteller={ersteller!r}."
                    )
                    yield Flag(
                        lens_id=self.lens_id,
                        family=self.family,
                        title=title,
                        rationale=rationale,
                        evidence=(doc.source,),
                        confidence=0.35,
                    )

            if created and approved:
                # approval dated before entry
                if approved.date() < created.date():
                    title = f"Freigabe vor Erfassung: {doc.ref}"
                    rationale = (
                        f"FREIGABEDATUM {approved.date()} liegt vor "
                        f"ERFASST_AM {created.date()}, Ref={doc.ref!r}."
                    )
                    yield Flag(
                        lens_id=self.lens_id,
                        family=self.family,
                        title=title,
                        rationale=rationale,
                        evidence=(doc.source,),
                        confidence=0.45,
                    )
                else:
                    delta = approved - created
                    # only flag sub-minute when both have real times (not midnight defaults)
                    has_real_times = bool(erfasst_um) and (
                        bool(freigabe_um) or approved.time().hour + approved.time().minute > 0
                    )
                    if has_real_times and timedelta(0) <= delta <= timedelta(seconds=30):
                        title = f"Freigabe in Sekunden: {doc.ref}"
                        rationale = (
                            f"Freigabe {delta.total_seconds():.0f}s nach Erfassung "
                            f"({created.isoformat(sep=' ')} → {approved.isoformat(sep=' ')}). "
                            f"Ersteller={ersteller!r}, Freigeber={freigeber!r}."
                        )
                        yield Flag(
                            lens_id=self.lens_id,
                            family=self.family,
                            title=title,
                            rationale=rationale,
                            evidence=(doc.source,),
                            confidence=0.4,
                        )

            # self-approval of journal
            if ersteller and freigeber and ersteller == freigeber:
                title = f"Selbstfreigabe Journal: {doc.ref}"
                rationale = (
                    f"ERSTELLER == FREIGEBER ({ersteller}), Ref={doc.ref!r}, "
                    f"Status={status!r}."
                )
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=(doc.source,),
                    confidence=0.4,
                )

# register instances (pipeline calls lens.run(dossier))
register(BackdatingLag())
register(OffHours())
register(MasterDataTiming())
register(VelocityBurst())
register(SequenceGaps())
register(ApprovalTiming())

"""Statistical lenses: Benford, round-number rate, robust outliers, duplicates.

Practice-set flag rates (calibrated post-scoring merge, 26647 postings):
  S_benford             1   0.00%  (partition flags; MAD cut ~0.055)
  S_round_frequency     0   0.00%  (round-1000 baseline ~0.1%)
  S_robust_outlier     28   0.11%  (z>8 and amount>=JET_FLOOR)
  S_duplicate_payment  14   0.05%  (one flag per cluster, not per pair)
  S_amount_precision   12   0.05%
  # all << 2% of rows; no further threshold cuts

Confidence stays in 0.2-0.5: population stats are leads, not verdicts.
Every baseline (round rate, MAD cut) is derived from the dossier itself.
"""

from __future__ import annotations

import math
from collections import defaultdict
from decimal import Decimal
from itertools import combinations
from statistics import median
from typing import Iterable, Sequence

from ..contracts import (
    JET_FLOOR,
    Dossier,
    Flag,
    LensFamily,
    Posting,
    SourceRef,
    register,
)

# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

_MIN_BENFORD_N = 300
_MIN_BENFORD_TWO_N = 500
_MIN_GROUP_ROUND = 20
_MIN_GROUP_OUTLIER = 15
_MIN_DUP_AMOUNT = Decimal("100")  # ignore penny noise
_DUP_WINDOW_DAYS = 14
_NEAR_EDIT_DIST = 2  # transposition of two digits is edit distance 2
_MAX_EVIDENCE = 8


def _abs_amt(p: Posting) -> Decimal:
    return abs(p.amount)


def _sample_evidence(postings: Sequence[Posting], limit: int = _MAX_EVIDENCE) -> tuple[SourceRef, ...]:
    """Representative SourceRefs; always non-empty if postings non-empty."""
    if not postings:
        return (SourceRef(file="(none)", excerpt="no rows"),)
    step = max(1, len(postings) // limit)
    picked = list(postings[::step][:limit])
    return tuple(p.source for p in picked)


def _leading_digit(amount: Decimal) -> int | None:
    if amount <= 0:
        return None
    # normalize via scientific representation
    x = float(amount)
    if x <= 0 or not math.isfinite(x):
        return None
    exp = math.floor(math.log10(x))
    coeff = x / (10**exp)
    d = int(coeff)
    if 1 <= d <= 9:
        return d
    # fallback: first non-zero digit in decimal string
    s = format(amount, "f").replace(".", "").lstrip("0")
    if s and s[0].isdigit():
        return int(s[0])
    return None


def _leading_two_digits(amount: Decimal) -> int | None:
    if amount < 10:
        return None
    x = float(amount)
    if x <= 0 or not math.isfinite(x):
        return None
    exp = math.floor(math.log10(x))
    coeff = x / (10 ** (exp - 1))
    d = int(coeff)
    if 10 <= d <= 99:
        return d
    return None


def _benford_probs_first() -> dict[int, float]:
    return {d: math.log10(1 + 1 / d) for d in range(1, 10)}


def _benford_probs_two() -> dict[int, float]:
    return {d: math.log10(1 + 1 / d) for d in range(10, 100)}


def _mad_statistic(counts: dict[int, int], expected: dict[int, float]) -> float:
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    return sum(abs(counts.get(k, 0) / n - expected[k]) for k in expected) / len(expected)


def _chi_square(counts: dict[int, int], expected: dict[int, float]) -> float:
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    chi = 0.0
    for k, p in expected.items():
        exp = n * p
        if exp <= 0:
            continue
        obs = counts.get(k, 0)
        chi += (obs - exp) ** 2 / exp
    return chi


def _is_round_amount(amount: Decimal, base: Decimal) -> bool:
    """True if |amount| is a non-zero multiple of base (exact thousands etc.)."""
    a = abs(amount)
    if a < base:
        return False
    # whole units of base
    q = a / base
    return q == q.to_integral_value()


def _amount_digit_string(amount: Decimal) -> str:
    """Normalized digit string without sign or decimal point (cents as trailing digits)."""
    a = abs(amount).quantize(Decimal("0.01"))
    return format(a, "f").replace(".", "").lstrip("0") or "0"


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance; short strings only."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > _NEAR_EDIT_DIST:
        return _NEAR_EDIT_DIST + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def _median_abs_dev(values: Sequence[Decimal], med: Decimal) -> Decimal:
    if not values:
        return Decimal("0")
    devs = sorted(abs(v - med) for v in values)
    mid = len(devs) // 2
    if len(devs) % 2:
        return devs[mid]
    return (devs[mid - 1] + devs[mid]) / 2


def _percentile_f(vals: Sequence[float], p: float) -> float:
    s = sorted(vals)
    if not s:
        return 0.0
    if p <= 0:
        return float(s[0])
    if p >= 1:
        return float(s[-1])
    return float(s[int(p * (len(s) - 1))])


# --------------------------------------------------------------------------
# Benford
# --------------------------------------------------------------------------


class BenfordLeadingDigits:
    """Flag partitions whose leading-digit distribution deviates hard from Benford.

    Flags the partition (entity/user/account), not every row inside it.
    """

    lens_id = "S_benford"
    family = LensFamily.STATISTICAL

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        postings = [p for p in dossier.postings if _abs_amt(p) > 0]
        if len(postings) < _MIN_BENFORD_N:
            return

        partitions: dict[tuple[str, str], list[Posting]] = defaultdict(list)
        for p in postings:
            if p.entity_id:
                partitions[("entity", p.entity_id)].append(p)
            if p.user:
                partitions[("user", p.user)].append(p)
            if p.account:
                partitions[("account", p.account)].append(p)

        # first pass: MAD for every large enough partition
        expected1 = _benford_probs_first()
        mads: list[float] = []
        stats: dict[tuple[str, str], tuple[float, float, int, list[Posting]]] = {}

        for key, rows in partitions.items():
            if len(rows) < _MIN_BENFORD_N:
                continue
            counts: dict[int, int] = defaultdict(int)
            for p in rows:
                d = _leading_digit(_abs_amt(p))
                if d is not None:
                    counts[d] += 1
            n = sum(counts.values())
            if n < _MIN_BENFORD_N:
                continue
            mad = _mad_statistic(counts, expected1)
            chi = _chi_square(counts, expected1)
            mads.append(mad)
            stats[key] = (mad, chi, n, rows)

        if not mads:
            return

        # dossier-derived cut: extreme tail of partition MADs.
        # Nigrini nonconformity starts ~0.015; real ledgers often sit 0.02-0.05,
        # so require absolute nonconformity AND being above the pack.
        med_mad = median(mads)
        mad_of_mads = median([abs(m - med_mad) for m in mads]) or 0.005
        cut = max(0.055, med_mad + 1.5 * mad_of_mads)

        expected2 = _benford_probs_two()

        for (kind, pid), (mad, chi, n, rows) in stats.items():
            if mad < cut:
                continue

            # optional two-digit reinforcement
            two_note = ""
            if n >= _MIN_BENFORD_TWO_N:
                c2: dict[int, int] = defaultdict(int)
                for p in rows:
                    d2 = _leading_two_digits(_abs_amt(p))
                    if d2 is not None:
                        c2[d2] += 1
                if sum(c2.values()) >= _MIN_BENFORD_TWO_N:
                    mad2 = _mad_statistic(c2, expected2)
                    two_note = f"; first-two MAD={mad2:.4f}"

            conf = 0.25
            if mad >= cut * 1.3:
                conf = 0.35
            if mad >= cut * 1.6:
                conf = 0.45

            label = {"entity": "Kreditor/Debitor", "user": "Benutzer", "account": "Konto"}.get(
                kind, kind
            )
            title = f"Benford-Abweichung: {label} {pid}"
            rationale = (
                f"Partition {kind}={pid}, n={n}, first-digit MAD={mad:.4f} "
                f"(Schwelle {cut:.4f}, Median-MAD {med_mad:.4f}), "
                f"chi2={chi:.1f}{two_note}. "
                f"Hohe Abweichung von Benford kann auf konstruierte Betraege hindeuten."
            )
            yield Flag(
                lens_id=self.lens_id,
                family=self.family,
                title=title,
                rationale=rationale,
                evidence=_sample_evidence(rows),
                entity_id=pid if kind == "entity" else None,
                doc_no=None,
                amount=None,
                confidence=conf,
            )


# --------------------------------------------------------------------------
# Round-number frequency (population, not single-row K6)
# --------------------------------------------------------------------------


class RoundNumberFrequency:
    """Flag vendors/users whose share of round amounts far exceeds the ledger baseline."""

    lens_id = "S_round_frequency"
    family = LensFamily.STATISTICAL

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        postings = [p for p in dossier.postings if _abs_amt(p) > 0]
        if len(postings) < _MIN_GROUP_ROUND:
            return

        # choose round base from data: prefer 1000 if any exist, else 100
        n1000 = sum(1 for p in postings if _is_round_amount(p.amount, Decimal("1000")))
        base = Decimal("1000") if n1000 >= 3 else Decimal("100")

        baseline = sum(1 for p in postings if _is_round_amount(p.amount, base)) / len(postings)
        min_round_count = 3

        groups: dict[tuple[str, str], list[Posting]] = defaultdict(list)
        for p in postings:
            if p.entity_id:
                groups[("entity", p.entity_id)].append(p)
            if p.user:
                groups[("user", p.user)].append(p)

        for (kind, pid), rows in groups.items():
            if len(rows) < _MIN_GROUP_ROUND:
                continue
            n_round = sum(1 for p in rows if _is_round_amount(p.amount, base))
            if n_round < min_round_count:
                continue
            rate = n_round / len(rows)
            # lift over baseline + absolute floor so rare baselines don't need 80%+
            lift = rate / baseline if baseline > 1e-9 else (999.0 if rate > 0 else 0.0)
            absolute_ok = rate >= max(0.35, baseline + 0.20)
            lift_ok = lift >= 4.0 and rate >= 0.25
            if not (absolute_ok or lift_ok):
                continue

            conf = 0.3
            if rate >= 0.5 or lift >= 8:
                conf = 0.4
            if rate >= 0.7:
                conf = 0.5

            round_rows = [p for p in rows if _is_round_amount(p.amount, base)]
            label = "Kreditor" if kind == "entity" else "Benutzer"
            title = f"Auffaellige Rundbetraege: {label} {pid}"
            rationale = (
                f"{kind}={pid}: {n_round}/{len(rows)} = {rate:.1%} runde Betraege "
                f"(Vielfache von {base}), Ledger-Baseline {baseline:.1%}, "
                f"Lift={lift:.1f}x."
            )
            yield Flag(
                lens_id=self.lens_id,
                family=self.family,
                title=title,
                rationale=rationale,
                evidence=_sample_evidence(round_rows),
                entity_id=pid if kind == "entity" else None,
                amount=max((_abs_amt(p) for p in round_rows), default=None),
                confidence=conf,
            )


# --------------------------------------------------------------------------
# Robust outliers (median + MAD)
# --------------------------------------------------------------------------


class RobustOutliers:
    """Per-vendor / per-account amount outliers via modified z-score (median/MAD)."""

    lens_id = "S_robust_outlier"
    family = LensFamily.STATISTICAL

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        postings = [p for p in dossier.postings if _abs_amt(p) > 0]
        if len(postings) < _MIN_GROUP_OUTLIER:
            return

        groups: dict[tuple[str, str], list[Posting]] = defaultdict(list)
        for p in postings:
            if p.entity_id:
                groups[("entity", p.entity_id)].append(p)
            if p.account:
                groups[("account", p.account)].append(p)

        # modified z-score threshold: 8 is already extreme; 10 is the sharp tail
        z_cut = 8.0

        seen: set[tuple[str, str | None]] = set()  # (doc_no, entity) de-dupe

        for (kind, pid), rows in groups.items():
            if len(rows) < _MIN_GROUP_OUTLIER:
                continue
            vals = [_abs_amt(p) for p in rows]
            med = median(vals)
            mad = _median_abs_dev(vals, med)
            if mad <= 0:
                # all equal — no outlier definition
                continue
            for p in rows:
                a = _abs_amt(p)
                if a < JET_FLOOR:
                    continue
                z = 0.6745 * float(a - med) / float(mad)
                if abs(z) < z_cut:
                    continue
                key = (p.doc_no, p.entity_id)
                if key in seen:
                    continue
                seen.add(key)

                conf = 0.35
                if abs(z) >= 12:
                    conf = 0.45
                if abs(z) >= 16:
                    conf = 0.5

                title = f"Ausreisser-Betrag {a:,.2f} auf {kind} {pid}"
                rationale = (
                    f"Modified z-score={z:.1f} (Schwelle {z_cut}), "
                    f"Median={med:,.2f}, MAD={mad:,.2f}, n={len(rows)}, "
                    f"JET_FLOOR={JET_FLOOR}. Robustes Mass, nicht Mittelwert/Std."
                )
                yield Flag(
                    lens_id=self.lens_id,
                    family=self.family,
                    title=title,
                    rationale=rationale,
                    evidence=(p.source,),
                    entity_id=p.entity_id,
                    doc_no=p.doc_no or None,
                    amount=a,
                    confidence=conf,
                )


# --------------------------------------------------------------------------
# Duplicate / near-duplicate payments
# --------------------------------------------------------------------------


class DuplicatePayments:
    """Same vendor, same/near amount, close in time, different document numbers."""

    lens_id = "S_duplicate_payment"
    family = LensFamily.STATISTICAL

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        by_entity: dict[str, list[Posting]] = defaultdict(list)
        for p in dossier.postings:
            if not p.entity_id:
                continue
            if _abs_amt(p) < _MIN_DUP_AMOUNT:
                continue
            by_entity[p.entity_id].append(p)

        for eid, rows in by_entity.items():
            if len(rows) < 2:
                continue
            # exact amount groups
            by_amt: dict[Decimal, list[Posting]] = defaultdict(list)
            for p in rows:
                by_amt[_abs_amt(p)].append(p)

            reported_pairs: set[frozenset[str]] = set()

            for amt, group in by_amt.items():
                if len(group) < 2:
                    continue
                group = sorted(group, key=lambda p: p.booking_date)
                # cluster postings that fall within the window; one flag per cluster
                clusters: list[list[Posting]] = []
                for p in group:
                    placed = False
                    for cl in clusters:
                        if abs((p.booking_date - cl[-1].booking_date).days) <= _DUP_WINDOW_DAYS:
                            # distinct doc_no preferred; still allow missing doc_no
                            if p.doc_no and any(p.doc_no == q.doc_no for q in cl if q.doc_no):
                                continue
                            cl.append(p)
                            placed = True
                            break
                    if not placed:
                        clusters.append([p])

                for cl in clusters:
                    # need at least two distinct document numbers (or two rows)
                    doc_nos = {p.doc_no for p in cl if p.doc_no}
                    if len(cl) < 2:
                        continue
                    if doc_nos and len(doc_nos) < 2:
                        continue
                    span = (cl[-1].booking_date - cl[0].booking_date).days
                    conf = 0.45 if span <= 3 else 0.35
                    belege = ", ".join(sorted(doc_nos)[:6]) if doc_nos else "(ohne Belegnr)"
                    title = f"Doppelte Zahlung {amt:,.2f} an {eid}"
                    rationale = (
                        f"Gleicher Betrag {amt:,.2f} an entity {eid}, "
                        f"Cluster n={len(cl)}, Spanne {span}d (Fenster {_DUP_WINDOW_DAYS}d), "
                        f"Belege {belege}."
                    )
                    yield Flag(
                        lens_id=self.lens_id,
                        family=self.family,
                        title=title,
                        rationale=rationale,
                        evidence=tuple(p.source for p in cl[:_MAX_EVIDENCE]),
                        entity_id=eid,
                        doc_no=cl[0].doc_no or None,
                        amount=amt,
                        confidence=conf,
                    )

            # near-duplicates: digit transposition / one-edit amount strings
            # only compare within a sliding date window to stay O(n log n)-ish
            dated = sorted(rows, key=lambda p: p.booking_date)
            for i, a in enumerate(dated):
                sa = _amount_digit_string(a.amount)
                for j in range(i + 1, len(dated)):
                    b = dated[j]
                    if (b.booking_date - a.booking_date).days > _DUP_WINDOW_DAYS:
                        break
                    if a.doc_no and b.doc_no and a.doc_no == b.doc_no:
                        continue
                    if _abs_amt(a) == _abs_amt(b):
                        continue  # exact handled above
                    # skip if amounts differ by more than ~2% (cheap prefilter)
                    aa, bb = _abs_amt(a), _abs_amt(b)
                    if aa == 0 or bb == 0:
                        continue
                    ratio = float(max(aa, bb) / min(aa, bb))
                    if ratio > 1.05:
                        continue
                    sb = _amount_digit_string(b.amount)
                    if _edit_distance(sa, sb) > _NEAR_EDIT_DIST:
                        continue
                    pair_key = frozenset(
                        {
                            f"{a.doc_no}|{a.source.line}|near",
                            f"{b.doc_no}|{b.source.line}|near",
                        }
                    )
                    if pair_key in reported_pairs:
                        continue
                    reported_pairs.add(pair_key)

                    title = f"Nahezu-doppelte Zahlung an {eid}"
                    rationale = (
                        f"Betraege {aa:,.2f} vs {bb:,.2f} (Edit-Distanz 1 der Ziffern), "
                        f"entity {eid}, {(b.booking_date - a.booking_date).days} Tage Abstand. "
                        f"Typisches Vertauschungs-/Tippfehlermuster."
                    )
                    yield Flag(
                        lens_id=self.lens_id,
                        family=self.family,
                        title=title,
                        rationale=rationale,
                        evidence=(a.source, b.source),
                        entity_id=eid,
                        doc_no=a.doc_no or b.doc_no,
                        amount=max(aa, bb),
                        confidence=0.3,
                    )


# also surface amount-digit precision clustering as a soft partition flag
class AmountPrecisionCluster:
    """Flag entities where nearly all amounts are whole euros while the ledger has cents.

    Practice-set rate: whole-euro baseline ~21%; only flag when partition is ~100%
    whole and baseline has substantial cent usage, with enough volume.
    Expected flags: handful of entities if any.
    """

    lens_id = "S_amount_precision"
    family = LensFamily.STATISTICAL

    def run(self, dossier: Dossier) -> Iterable[Flag]:
        postings = [p for p in dossier.postings if _abs_amt(p) > 0]
        if len(postings) < 50:
            return

        def is_whole(a: Decimal) -> bool:
            return a == a.to_integral_value()

        baseline = sum(1 for p in postings if is_whole(_abs_amt(p))) / len(postings)
        # only meaningful if ledger normally has cents
        if baseline > 0.85:
            return

        by_entity: dict[str, list[Posting]] = defaultdict(list)
        for p in postings:
            if p.entity_id:
                by_entity[p.entity_id].append(p)

        for eid, rows in by_entity.items():
            if len(rows) < 25:
                continue
            rate = sum(1 for p in rows if is_whole(_abs_amt(p))) / len(rows)
            if rate < 0.98 or rate < baseline + 0.35:
                continue
            title = f"Ungewoehnliche Betragsgenauigkeit: {eid}"
            rationale = (
                f"Entity {eid}: {rate:.1%} ganzzahlige Betraege "
                f"(n={len(rows)}), Ledger-Baseline {baseline:.1%}."
            )
            yield Flag(
                lens_id=self.lens_id,
                family=self.family,
                title=title,
                rationale=rationale,
                evidence=_sample_evidence(rows),
                entity_id=eid,
                confidence=0.25,
            )

# register instances (pipeline calls lens.run(dossier))
register(BenfordLeadingDigits())
register(RoundNumberFrequency())
register(RobustOutliers())
register(DuplicatePayments())
register(AmountPrecisionCluster())

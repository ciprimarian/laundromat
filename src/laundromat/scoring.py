"""Corroboration scoring: turn a pile of Flags into tiered Findings.

Aggregation happens at TWO levels:
  - transaction level (a suspicious payment)  -> subject_kind="transaction"
  - entity level      (a suspicious vendor)   -> subject_kind="entity"

The entity level is usually where the story is: "Vendor X: 5 independent red
flags across 4 lens families" reads like an investigation, whereas the same
data as a list of rows reads like noise.
"""

from __future__ import annotations

import math
from collections import defaultdict
from decimal import Decimal

from .contracts import JET_FLOOR, MATERIALITY, Finding, Flag, LensFamily, Tier


def group_flags(flags: list[Flag]) -> list[Finding]:
    """Bucket raw flags into per-entity and per-transaction findings."""
    by_entity: dict[str, list[Flag]] = defaultdict(list)
    by_txn: dict[str, list[Flag]] = defaultdict(list)

    for flag in flags:
        if flag.entity_id:
            by_entity[flag.entity_id].append(flag)
        if flag.doc_no:
            by_txn[flag.doc_no].append(flag)

    findings = [
        Finding(subject_id=eid, subject_kind="entity", flags=fl)
        for eid, fl in by_entity.items()
    ]
    findings += [
        Finding(subject_id=doc, subject_kind="transaction", flags=fl)
        for doc, fl in by_txn.items()
    ]
    return findings


def score_finding(finding: Finding) -> tuple[float, Tier]:
    """Assign a suspicion score and a tier to one finding.

    Policy: corroboration across independent lens families is the backbone.
    Lenses within one family share failure modes, so two rule hits are one
    opinion; a rule hit plus a graph hit are two. Confidence and materiality
    only order findings within a tier or gate obvious noise -- they can never
    substitute for independence. A single-family finding is capped at REVIEW
    no matter how confident the lens: that cap is the false-positive shield.
    """
    n = len(finding.families)

    # Per-family confidence: the strongest flag speaks for its family, so ten
    # weak flags from one lens cannot inflate the score by volume.
    best_by_family: dict[str, float] = {}
    for flag in finding.flags:
        key = flag.family.value
        best_by_family[key] = max(best_by_family.get(key, 0.0), flag.confidence)
    conf = sum(best_by_family.values()) / len(best_by_family) if best_by_family else 0.0

    # Materiality factor: 0 at the JET floor, 1 at materiality, log-scaled
    # between so a 50k finding is not drowned by a 350k one on a linear axis.
    amount = float(finding.max_amount)
    if amount <= float(JET_FLOOR):
        amt = 0.0
    else:
        span = math.log10(float(MATERIALITY)) - math.log10(float(JET_FLOOR))
        amt = min(1.0, (math.log10(amount) - math.log10(float(JET_FLOOR))) / span)

    # Entities accumulate flags across many transactions; a small breadth
    # bonus rewards patterns over one-off hits without letting volume dominate.
    breadth = min(1.0, len(finding.flags) / 5.0) * 0.5 if finding.subject_kind == "entity" else 0.0

    score = 2.5 * n + 2.0 * conf + 1.5 * amt + breadth

    if n >= 3:
        tier = Tier.HIGH
    elif n == 2:
        # Two independent families on a material amount with real confidence
        # is conviction territory; otherwise report with caveat.
        tier = Tier.HIGH if (amount >= float(MATERIALITY) and conf >= 0.6) else Tier.MEDIUM
    else:
        # Single family: never above REVIEW. Sub-floor amounts with a weak
        # lens self-assessment are not worth a defense-pass call at all.
        if amount < float(JET_FLOOR) and conf < 0.5:
            tier = Tier.DISMISSED
        else:
            tier = Tier.REVIEW

    return score, tier


def score_all(flags: list[Flag]) -> list[Finding]:
    findings = group_flags(flags)
    for finding in findings:
        finding.score, finding.tier = score_finding(finding)
    findings.sort(key=lambda f: f.score, reverse=True)
    return findings

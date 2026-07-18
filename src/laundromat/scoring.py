"""Corroboration scoring: turn a pile of Flags into tiered Findings.

Aggregation happens at TWO levels:
  - transaction level (a suspicious payment)  -> subject_kind="transaction"
  - entity level      (a suspicious vendor)   -> subject_kind="entity"

The entity level is usually where the story is: "Vendor X: 5 independent red
flags across 4 lens families" reads like an investigation, whereas the same
data as a list of rows reads like noise.
"""

from __future__ import annotations

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

    Inputs available on `finding`:
      finding.flags        -> list[Flag]; each has .family, .confidence, .amount
      finding.families     -> set[LensFamily] of DISTINCT families that fired
      finding.max_amount    -> Decimal, largest amount any flag cited
      finding.subject_kind -> "entity" | "transaction"

    Thresholds importable from .contracts: MATERIALITY (400k), JET_FLOOR (25k).

    Return (score, tier). Tier drives what happens next:
      Tier.HIGH / Tier.MEDIUM -> reported directly
      Tier.REVIEW             -> sent to the defense pass for exoneration
    """
    # TODO(human): implement the corroboration scoring rule.
    raise NotImplementedError


def score_all(flags: list[Flag]) -> list[Finding]:
    findings = group_flags(flags)
    for finding in findings:
        finding.score, finding.tier = score_finding(finding)
    findings.sort(key=lambda f: f.score, reverse=True)
    return findings

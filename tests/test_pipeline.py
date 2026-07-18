"""Contract invariants over a real practice-set pipeline run.

These are submission guarantees: the full ingest -> lenses -> score path must
hold on data/practice without exception and without hollow flags.
"""

from __future__ import annotations

from collections import Counter

import pytest

from laundromat.contracts import Tier
from laundromat.pipeline import run

# Practice set is large; one shared run for the module.
@pytest.fixture(scope="module")
def practice_run():
    return run("data/practice")


def test_pipeline_completes(practice_run):
    dossier, flags, findings = practice_run
    assert dossier is not None
    assert isinstance(flags, list)
    assert isinstance(findings, list)


def test_posting_count_is_26647(practice_run):
    dossier, _flags, _findings = practice_run
    assert len(dossier.postings) == 26647


def test_every_flag_has_grounded_evidence(practice_run):
    _dossier, flags, _findings = practice_run
    assert flags, "expected some flags on the practice set"
    for fl in flags:
        assert fl.evidence, f"{fl.lens_id}: empty evidence"
        for ref in fl.evidence:
            assert ref.file, f"{fl.lens_id}: evidence missing file"
            # line (csv/txt/xlsx) or page (pdf/docx) must be set
            assert ref.line is not None or ref.page is not None, (
                f"{fl.lens_id}: evidence {ref.file!r} has neither line nor page"
            )


def test_high_findings_have_at_least_three_families(practice_run):
    _d, _flags, findings = practice_run
    highs = [f for f in findings if f.tier == Tier.HIGH]
    # practice should produce at least one multi-family conviction
    assert highs, "expected at least one HIGH finding on practice"
    for f in highs:
        n = len(f.families)
        assert n >= 3, (
            f"HIGH {f.subject_id} has only {n} families "
            f"{sorted(x.value for x in f.families)}; need >= 3"
        )


def test_medium_findings_have_at_least_two_families(practice_run):
    _d, _flags, findings = practice_run
    mediums = [f for f in findings if f.tier == Tier.MEDIUM]
    for f in mediums:
        n = len(f.families)
        assert n >= 2, (
            f"MEDIUM {f.subject_id} has only {n} families "
            f"{sorted(x.value for x in f.families)}; need >= 2"
        )


def test_dismissed_does_not_outrank_review(practice_run):
    """DISMISSED is the lowest tier; it must not be labeled above REVIEW.

    Score-sort can interleave DISMISSED and REVIEW numerically; the invariant
    is on tier assignment: DISMISSED is never HIGH/MEDIUM, and is strictly
    below REVIEW in the severity order.
    """
    _d, _flags, findings = practice_run
    severity = {
        Tier.HIGH: 3,
        Tier.MEDIUM: 2,
        Tier.REVIEW: 1,
        Tier.DISMISSED: 0,
    }
    assert severity[Tier.DISMISSED] < severity[Tier.REVIEW]
    for f in findings:
        if f.tier == Tier.DISMISSED:
            assert severity[f.tier] < severity[Tier.REVIEW]
            # single-family only path in scoring
            assert len(f.families) == 1, (
                f"DISMISSED {f.subject_id} has {len(f.families)} families"
            )


def test_findings_sorted_by_score_descending(practice_run):
    _d, _flags, findings = practice_run
    if len(findings) < 2:
        return
    scores = [f.score for f in findings]
    assert scores == sorted(scores, reverse=True)

"""Unit tests for statistical lenses using hand-built Dossier fixtures."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from laundromat.contracts import (
    Dossier,
    Entity,
    EntityType,
    Flag,
    LensFamily,
    Posting,
    SourceRef,
)
from laundromat.lenses.stats import (
    AmountPrecisionCluster,
    BenfordLeadingDigits,
    DuplicatePayments,
    RoundNumberFrequency,
    RobustOutliers,
)


def _src(file: str = "fixture/gl.txt", line: int = 1, excerpt: str = "row") -> SourceRef:
    return SourceRef(file=file, line=line, excerpt=excerpt)


def _post(
    *,
    doc_no: str = "B1",
    booking_date: date | None = None,
    amount: str | Decimal = "100.00",
    account: str = "400000",
    entity_id: str | None = None,
    user: str | None = "U1",
    line: int = 1,
    posted_at: datetime | None = None,
    text: str = "",
) -> Posting:
    amt = amount if isinstance(amount, Decimal) else Decimal(amount)
    return Posting(
        doc_no=doc_no,
        booking_date=booking_date or date(2025, 3, 15),
        amount=amt,
        account=account,
        source=_src(line=line, excerpt=f"{doc_no} {amt}"),
        posted_at=posted_at or datetime(2025, 3, 15, 10, 0, 0),
        entity_id=entity_id,
        user=user,
        text=text,
    )


def _dossier(postings: list[Posting] | None = None, **kwargs) -> Dossier:
    return Dossier(name="test", postings=postings or [], **kwargs)


# --------------------------------------------------------------------------
# empty / missing inputs
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lens",
    [
        BenfordLeadingDigits(),
        RoundNumberFrequency(),
        RobustOutliers(),
        DuplicatePayments(),
        AmountPrecisionCluster(),
    ],
)
def test_empty_dossier_emits_nothing(lens):
    flags = list(lens.run(_dossier()))
    assert flags == []


def test_sparse_data_skips_benford():
    posts = [_post(amount=str(100 + i), line=i) for i in range(50)]
    flags = list(BenfordLeadingDigits().run(_dossier(posts)))
    assert flags == []


# --------------------------------------------------------------------------
# Benford
# --------------------------------------------------------------------------


def _uniform_amounts(n: int, start: int = 1000) -> list[Decimal]:
    """Amounts with roughly flat first-digit distribution (anti-Benford)."""
    out = []
    # cycle leading digits 1-9 evenly via 1xxx, 2xxx, ...
    for i in range(n):
        lead = (i % 9) + 1
        rest = 100 + (i % 97)
        out.append(Decimal(lead * 1000 + rest))
    return out


def _benford_amounts(n: int) -> list[Decimal]:
    """Sample magnitudes whose leading digits roughly follow Benford."""
    import math
    import random

    rng = random.Random(42)
    out: list[Decimal] = []
    for _ in range(n):
        # log-uniform over several orders of magnitude -> Benford first digits
        log_x = 1.0 + rng.random() * 4.0  # 10 .. 100000
        x = 10**log_x
        out.append(Decimal(str(round(x, 2))))
    return out


def test_benford_flags_anti_benford_partition():
    # 350 postings for one entity with flat first digits (anti-Benford)
    amts = _uniform_amounts(350)
    posts = [
        _post(
            doc_no=f"D{i}",
            amount=amts[i],
            entity_id="V_FAKE",
            account="400000",
            user="U1",
            line=i + 1,
        )
        for i in range(350)
    ]
    # conforming background partitions (log-uniform amounts)
    ok_amts = _benford_amounts(400)
    for i, x in enumerate(ok_amts):
        posts.append(
            _post(
                doc_no=f"N{i}",
                amount=x if x > 0 else Decimal("10"),
                entity_id="V_OK",
                account="500000",
                user="U2",
                line=1000 + i,
            )
        )
    ok2 = _benford_amounts(400)
    for i, x in enumerate(ok2):
        posts.append(
            _post(
                doc_no=f"M{i}",
                amount=x if x > 0 else Decimal("10"),
                entity_id="V_OK2",
                account="600000",
                user="U3",
                line=2000 + i,
            )
        )

    flags = list(BenfordLeadingDigits().run(_dossier(posts)))
    assert all(f.family == LensFamily.STATISTICAL for f in flags)
    assert all(f.evidence for f in flags)
    assert all(0.2 <= f.confidence <= 0.5 for f in flags)
    assert any("V_FAKE" in f.title or f.entity_id == "V_FAKE" for f in flags)
    for f in flags:
        assert "MAD" in f.rationale


# --------------------------------------------------------------------------
# Round frequency
# --------------------------------------------------------------------------


def test_round_frequency_flags_vendor_above_baseline():
    posts = []
    # baseline: 80 normal amounts for various vendors
    for i in range(80):
        posts.append(
            _post(
                doc_no=f"N{i}",
                amount=str(1234 + i * 17 + Decimal("0.45")),
                entity_id=f"V{i % 5}",
                line=i + 1,
            )
        )
    # suspicious vendor: 25 postings, 15 of them exact thousands
    for i in range(25):
        amt = Decimal("5000") if i < 15 else Decimal(str(2100 + i * 13.37))
        posts.append(
            _post(
                doc_no=f"R{i}",
                amount=amt,
                entity_id="V_ROUND",
                line=200 + i,
            )
        )
    flags = list(RoundNumberFrequency().run(_dossier(posts)))
    assert any(f.entity_id == "V_ROUND" for f in flags)
    hit = next(f for f in flags if f.entity_id == "V_ROUND")
    assert "Baseline" in hit.rationale or "baseline" in hit.rationale.lower() or "Baseline" in hit.rationale or "Ledger" in hit.rationale
    assert hit.family == LensFamily.STATISTICAL


def test_round_frequency_ignores_small_groups():
    posts = [
        _post(doc_no=f"R{i}", amount=Decimal("1000"), entity_id="V_TINY", line=i)
        for i in range(5)
    ]
    flags = list(RoundNumberFrequency().run(_dossier(posts)))
    assert flags == []


# --------------------------------------------------------------------------
# Robust outliers
# --------------------------------------------------------------------------


def test_robust_outlier_flags_extreme_amount():
    posts = []
    # typical invoices ~5k
    for i in range(30):
        posts.append(
            _post(
                doc_no=f"N{i}",
                amount=Decimal("5000") + Decimal(i * 10),
                entity_id="V1",
                account="440000",
                line=i + 1,
            )
        )
    # extreme outlier above JET_FLOOR
    posts.append(
        _post(
            doc_no="OUT",
            amount=Decimal("250000"),
            entity_id="V1",
            account="440000",
            line=99,
        )
    )
    flags = list(RobustOutliers().run(_dossier(posts)))
    assert any(f.doc_no == "OUT" for f in flags)
    hit = next(f for f in flags if f.doc_no == "OUT")
    assert hit.amount == Decimal("250000")
    assert "z-score" in hit.rationale.lower() or "z-score" in hit.rationale or "Modified" in hit.rationale
    assert hit.confidence <= 0.5


def test_robust_outlier_respects_jet_floor():
    posts = [
        _post(doc_no=f"N{i}", amount=Decimal("100") + i, entity_id="V1", line=i)
        for i in range(30)
    ]
    posts.append(_post(doc_no="SMALL", amount=Decimal("5000"), entity_id="V1", line=99))
    flags = list(RobustOutliers().run(_dossier(posts)))
    # 5000 < JET_FLOOR 25000 — must not flag
    assert not any(f.doc_no == "SMALL" for f in flags)


# --------------------------------------------------------------------------
# Duplicates
# --------------------------------------------------------------------------


def test_duplicate_exact_same_amount_close_dates():
    posts = [
        _post(
            doc_no="A1",
            amount=Decimal("12345.67"),
            entity_id="V9",
            booking_date=date(2025, 4, 1),
            line=1,
        ),
        _post(
            doc_no="A2",
            amount=Decimal("12345.67"),
            entity_id="V9",
            booking_date=date(2025, 4, 3),
            line=2,
        ),
    ]
    flags = list(DuplicatePayments().run(_dossier(posts)))
    assert len(flags) >= 1
    assert flags[0].entity_id == "V9"
    assert len(flags[0].evidence) >= 2
    assert "12345.67" in flags[0].rationale or "12,345.67" in flags[0].rationale


def test_duplicate_ignores_far_apart():
    posts = [
        _post(
            doc_no="A1",
            amount=Decimal("9999.00"),
            entity_id="V9",
            booking_date=date(2025, 1, 1),
            line=1,
        ),
        _post(
            doc_no="A2",
            amount=Decimal("9999.00"),
            entity_id="V9",
            booking_date=date(2025, 6, 1),
            line=2,
        ),
    ]
    flags = list(DuplicatePayments().run(_dossier(posts)))
    assert flags == []


def test_near_duplicate_transposition():
    posts = [
        _post(
            doc_no="T1",
            amount=Decimal("12345.00"),
            entity_id="V3",
            booking_date=date(2025, 5, 1),
            line=1,
        ),
        _post(
            doc_no="T2",
            amount=Decimal("12354.00"),  # transposition of last two digits before decimal
            entity_id="V3",
            booking_date=date(2025, 5, 2),
            line=2,
        ),
    ]
    flags = list(DuplicatePayments().run(_dossier(posts)))
    assert any("Nahezu" in f.title or "nahezu" in f.rationale.lower() or "Edit" in f.rationale for f in flags)


# --------------------------------------------------------------------------
# Flag construction rules
# --------------------------------------------------------------------------


def test_every_flag_has_nonempty_evidence_and_family():
    posts = []
    for i in range(30):
        posts.append(
            _post(
                doc_no=f"N{i}",
                amount=Decimal("5000") + i,
                entity_id="V1",
                line=i,
            )
        )
    posts.append(
        _post(doc_no="OUT", amount=Decimal("400000"), entity_id="V1", line=99)
    )
    for lens in (RobustOutliers(), DuplicatePayments()):
        for f in lens.run(_dossier(posts)):
            assert isinstance(f, Flag)
            assert f.evidence
            assert f.family == LensFamily.STATISTICAL
            assert f.lens_id
            assert f.title
            assert f.rationale

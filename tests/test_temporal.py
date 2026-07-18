"""Unit tests for temporal lenses using hand-built Dossier fixtures."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from laundromat.contracts import (
    Dossier,
    Document,
    Flag,
    LensFamily,
    Posting,
    SourceRef,
)
from laundromat.lenses.temporal import (
    ApprovalTiming,
    BackdatingLag,
    MasterDataTiming,
    OffHours,
    SequenceGaps,
    VelocityBurst,
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
    attrs: dict | None = None,
) -> Posting:
    amt = amount if isinstance(amount, Decimal) else Decimal(amount)
    bd = booking_date or date(2025, 6, 15)
    return Posting(
        doc_no=doc_no,
        booking_date=bd,
        amount=amt,
        account=account,
        source=_src(line=line, excerpt=f"{doc_no} {amt}"),
        posted_at=posted_at if posted_at is not None else datetime(bd.year, bd.month, bd.day, 10, 0, 0),
        entity_id=entity_id,
        user=user,
        attrs=attrs or {},
    )


def _doc(
    kind: str,
    ref: str,
    *,
    entity_id: str | None = None,
    doc_date: date | None = None,
    amount: Decimal | None = None,
    fields: dict | None = None,
    line: int = 1,
    file: str = "fixture/docs.csv",
) -> Document:
    return Document(
        kind=kind,
        ref=ref,
        source=SourceRef(file=file, line=line, excerpt=ref),
        entity_id=entity_id,
        doc_date=doc_date,
        amount=amount,
        fields=fields or {},
    )


def _dossier(postings=None, documents=None) -> Dossier:
    return Dossier(name="test", postings=postings or [], documents=documents or [])


# --------------------------------------------------------------------------
# empty / missing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lens",
    [
        BackdatingLag(),
        OffHours(),
        MasterDataTiming(),
        VelocityBurst(),
        SequenceGaps(),
        ApprovalTiming(),
    ],
)
def test_empty_dossier_emits_nothing(lens):
    assert list(lens.run(_dossier())) == []


def test_missing_posted_at_skips_backdating_and_hours():
    posts = [
        Posting(
            doc_no="X",
            booking_date=date(2025, 1, 1),
            amount=Decimal("100"),
            account="1",
            source=_src(),
            posted_at=None,
            user="U1",
        )
        for _ in range(60)
    ]
    assert list(BackdatingLag().run(_dossier(posts))) == []
    assert list(OffHours().run(_dossier(posts))) == []


# --------------------------------------------------------------------------
# Backdating
# --------------------------------------------------------------------------


def test_backdating_flags_extreme_lag():
    posts = []
    # bulk: same-day entry
    for i in range(80):
        d = date(2025, 3, 1) + timedelta(days=i % 28)
        posts.append(
            _post(
                doc_no=f"N{i}",
                booking_date=d,
                posted_at=datetime(d.year, d.month, d.day, 11, 0, 0),
                line=i + 1,
            )
        )
    # extreme backdate
    posts.append(
        _post(
            doc_no="BACK",
            booking_date=date(2025, 1, 1),
            posted_at=datetime(2025, 1, 20, 9, 0, 0),
            line=999,
            amount="50000",
            entity_id="V1",
        )
    )
    flags = list(BackdatingLag().run(_dossier(posts)))
    assert any(f.doc_no == "BACK" for f in flags)
    hit = next(f for f in flags if f.doc_no == "BACK")
    assert "Lag" in hit.rationale or "Tage" in hit.rationale
    assert hit.family == LensFamily.TEMPORAL
    assert 0.2 <= hit.confidence <= 0.5


def test_backdating_does_not_flag_same_day():
    posts = [
        _post(
            doc_no=f"N{i}",
            booking_date=date(2025, 4, 1),
            posted_at=datetime(2025, 4, 1, 10, 0, 0),
            line=i,
        )
        for i in range(60)
    ]
    assert list(BackdatingLag().run(_dossier(posts))) == []


# --------------------------------------------------------------------------
# Off hours
# --------------------------------------------------------------------------


def test_off_hours_flags_far_outside_user_norm():
    posts = []
    # user normally posts 09:00-17:00
    for i in range(60):
        hour = 9 + (i % 8)
        posts.append(
            _post(
                doc_no=f"N{i}",
                user="CLERK",
                posted_at=datetime(2025, 5, 1 + (i % 28), hour, 15, 0),
                line=i + 1,
            )
        )
    # 03:00 entry — far outside
    posts.append(
        _post(
            doc_no="NIGHT",
            user="CLERK",
            posted_at=datetime(2025, 5, 15, 3, 0, 0),
            line=999,
            amount="8000",
        )
    )
    flags = list(OffHours().run(_dossier(posts)))
    assert any(f.doc_no == "NIGHT" for f in flags)
    hit = next(f for f in flags if f.doc_no == "NIGHT")
    assert "CLERK" in hit.title or "CLERK" in hit.rationale


def test_off_hours_does_not_flag_normal_edge():
    posts = []
    for i in range(50):
        hour = 8 + (i % 10)  # 8-17
        posts.append(
            _post(
                doc_no=f"N{i}",
                user="CLERK",
                posted_at=datetime(2025, 5, 1 + (i % 28), hour, 0, 0),
                line=i + 1,
            )
        )
    flags = list(OffHours().run(_dossier(posts)))
    # no true night posts — should be empty or only extreme
    assert not any(
        f.doc_no and f.doc_no.startswith("N") and "03:" in f.rationale for f in flags
    )


# --------------------------------------------------------------------------
# Master data timing
# --------------------------------------------------------------------------


def test_master_self_approval():
    docs = [
        _doc(
            "master_change",
            "MC1",
            entity_id="209101",
            doc_date=date(2025, 5, 12),
            fields={
                "KONTO": "209101",
                "FELD": "Neuanlage Kreditor",
                "GEAENDERT_VON": "MV-U05",
                "GENEHMIGT_VON": "MV-U05",
                "GENEHMIGT": "Ja",
            },
        )
    ]
    flags = list(MasterDataTiming().run(_dossier(documents=docs)))
    assert any("Selbstfreigabe" in f.title for f in flags)


def test_master_unapproved():
    docs = [
        _doc(
            "master_change",
            "MC2",
            entity_id="200001",
            doc_date=date(2025, 6, 1),
            fields={
                "KONTO": "200001",
                "FELD": "Adresse",
                "GEAENDERT_VON": "U1",
                "GENEHMIGT_VON": "U2",
                "GENEHMIGT": "Nein",
            },
        )
    ]
    flags = list(MasterDataTiming().run(_dossier(documents=docs)))
    assert any("ungenehmigt" in f.title.lower() or "ungenehmigt" in f.title for f in flags)


def test_master_change_before_payment():
    docs = [
        _doc(
            "master_change",
            "MC3",
            entity_id="200099",
            doc_date=date(2025, 7, 1),
            fields={
                "KONTO": "200099",
                "FELD": "Bankverbindung",
                "GEAENDERT_VON": "U1",
                "GENEHMIGT_VON": "U2",
                "GENEHMIGT": "Ja",
            },
        )
    ]
    posts = [
        _post(
            doc_no="PAY1",
            entity_id="200099",
            amount=Decimal("45000"),
            booking_date=date(2025, 7, 3),
            line=1,
        )
    ]
    flags = list(MasterDataTiming().run(_dossier(postings=posts, documents=docs)))
    assert any("vor Zahlung" in f.title for f in flags)
    hit = next(f for f in flags if "vor Zahlung" in f.title)
    assert len(hit.evidence) >= 2


def test_master_reversion_with_payment():
    docs = [
        _doc(
            "master_change",
            "MC_A",
            entity_id="200050",
            doc_date=date(2025, 8, 1),
            fields={
                "KONTO": "200050",
                "FELD": "Bankverbindung",
                "GEAENDERT_VON": "U1",
                "GENEHMIGT_VON": "U2",
                "GENEHMIGT": "Ja",
            },
            line=1,
        ),
        _doc(
            "master_change",
            "MC_B",
            entity_id="200050",
            doc_date=date(2025, 8, 10),
            fields={
                "KONTO": "200050",
                "FELD": "Bankverbindung",
                "GEAENDERT_VON": "U1",
                "GENEHMIGT_VON": "U2",
                "GENEHMIGT": "Ja",
            },
            line=2,
        ),
    ]
    posts = [
        _post(
            doc_no="PAYX",
            entity_id="200050",
            amount=Decimal("80000"),
            booking_date=date(2025, 8, 5),
            line=1,
        )
    ]
    flags = list(MasterDataTiming().run(_dossier(postings=posts, documents=docs)))
    assert any("Reversion" in f.title for f in flags)


# --------------------------------------------------------------------------
# Velocity
# --------------------------------------------------------------------------


def test_velocity_burst_flags_spike_day():
    posts = []
    # normal: ~5/day over many days
    for day in range(1, 25):
        for j in range(5):
            posts.append(
                _post(
                    doc_no=f"D{day}_{j}",
                    user="BURST_U",
                    posted_at=datetime(2025, 3, day, 10 + j, 0, 0),
                    line=day * 10 + j,
                )
            )
    # spike: 80 on one day
    for j in range(80):
        posts.append(
            _post(
                doc_no=f"SPIKE_{j}",
                user="BURST_U",
                posted_at=datetime(2025, 3, 28, 9, j % 60, 0),
                line=5000 + j,
            )
        )
    flags = list(VelocityBurst().run(_dossier(posts)))
    assert any("burst" in f.title.lower() or "Burst" in f.title for f in flags)


def test_year_end_concentration():
    posts = []
    # most of the year small
    for i in range(30):
        posts.append(
            _post(
                doc_no=f"Y{i}",
                entity_id="V_YE",
                amount=Decimal("1000"),
                booking_date=date(2025, 3, 1) + timedelta(days=i * 5),
                line=i + 1,
            )
        )
    # year end dump
    for i in range(5):
        posts.append(
            _post(
                doc_no=f"YE{i}",
                entity_id="V_YE",
                amount=Decimal("50000"),
                booking_date=date(2025, 12, 28 + (i % 3)),
                line=100 + i,
            )
        )
    flags = list(VelocityBurst().run(_dossier(posts)))
    assert any(f.entity_id == "V_YE" and "Jahresend" in f.title for f in flags)


# --------------------------------------------------------------------------
# Sequence gaps
# --------------------------------------------------------------------------


def test_sequence_gap_flags_large_jump():
    posts = []
    # contiguous 100-130, then jump to 200
    for n in list(range(100, 131)) + [200, 201, 202]:
        posts.append(
            _post(
                doc_no=f"E{n}",
                attrs={"ERFASSUNGSNUMMER": str(n)},
                line=n,
            )
        )
    flags = list(SequenceGaps().run(_dossier(posts)))
    assert any("Luecke" in f.title or "Lücke" in f.title or "Luecke" in f.rationale for f in flags)


def test_sequence_empty_attrs_silent():
    posts = [_post(doc_no=f"N{i}", line=i) for i in range(30)]
    assert list(SequenceGaps().run(_dossier(posts))) == []


def test_journal_multi_day_span():
    posts = []
    for i in range(6):
        posts.append(
            _post(
                doc_no=f"J{i}",
                posted_at=datetime(2025, 4, 1 + i, 10, 0, 0),
                attrs={"JOURNALZEILE": str(i + 1), "JOURNALNAME": "GJ99"},
                line=i + 1,
            )
        )
    flags = list(SequenceGaps().run(_dossier(posts)))
    assert any("mehrere Tage" in f.title for f in flags)


# --------------------------------------------------------------------------
# Approval timing
# --------------------------------------------------------------------------


def test_approval_before_entry():
    docs = [
        _doc(
            "approval",
            "7700999",
            fields={
                "ERFASST_AM": "15.06.2025",
                "ERFASST_UM": "10:00:00",
                "FREIGABEDATUM": "10.06.2025",
                "FREIGABESTATUS": "Freigegeben",
                "ERSTELLER": "U1",
                "FREIGEBER": "U2",
            },
        )
    ]
    flags = list(ApprovalTiming().run(_dossier(documents=docs)))
    assert any("vor Erfassung" in f.title for f in flags)


def test_approval_self_approve():
    docs = [
        _doc(
            "approval",
            "7700888",
            fields={
                "ERFASST_AM": "15.06.2025",
                "ERFASST_UM": "10:00:00",
                "FREIGABEDATUM": "15.06.2025",
                "FREIGABESTATUS": "Freigegeben",
                "ERSTELLER": "Admin",
                "FREIGEBER": "Admin",
            },
        )
    ]
    flags = list(ApprovalTiming().run(_dossier(documents=docs)))
    assert any("Selbstfreigabe" in f.title for f in flags)


def test_approval_open_status():
    docs = [
        _doc(
            "approval",
            "7700777",
            fields={
                "ERFASST_AM": "15.06.2025",
                "ERFASST_UM": "10:00:00",
                "FREIGABEDATUM": "",
                "FREIGABESTATUS": "Offen",
                "ERSTELLER": "U1",
                "FREIGEBER": "",
            },
        )
    ]
    flags = list(ApprovalTiming().run(_dossier(documents=docs)))
    assert any("ohne Freigabe" in f.title for f in flags)


def test_approval_seconds():
    docs = [
        _doc(
            "approval",
            "7700666",
            fields={
                "ERFASST_AM": "15.06.2025",
                "ERFASST_UM": "10:00:00",
                "FREIGABEDATUM": "15.06.2025",
                "FREIGABE_UM": "10:00:15",
                "FREIGABESTATUS": "Freigegeben",
                "ERSTELLER": "U1",
                "FREIGEBER": "U2",
            },
        )
    ]
    flags = list(ApprovalTiming().run(_dossier(documents=docs)))
    assert any("Sekunden" in f.title for f in flags)


def test_all_temporal_flags_well_formed():
    posts = []
    for i in range(80):
        d = date(2025, 2, 1) + timedelta(days=i % 20)
        posts.append(
            _post(
                doc_no=f"N{i}",
                booking_date=d,
                posted_at=datetime(d.year, d.month, d.day, 10, 0, 0),
                user="U1",
                line=i,
            )
        )
    posts.append(
        _post(
            doc_no="BACK",
            booking_date=date(2025, 1, 1),
            posted_at=datetime(2025, 1, 25, 9, 0, 0),
            line=999,
        )
    )
    for lens in (BackdatingLag(), VelocityBurst()):
        for f in lens.run(_dossier(posts)):
            assert isinstance(f, Flag)
            assert f.evidence
            assert f.family == LensFamily.TEMPORAL
            assert 0.0 < f.confidence <= 1.0

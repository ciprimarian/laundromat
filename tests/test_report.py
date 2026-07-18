"""FastAPI smoke tests for the report UI (no browser)."""

from __future__ import annotations

import io
import zipfile
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from laundromat.contracts import (
    Dossier,
    Entity,
    EntityType,
    Finding,
    Flag,
    LensFamily,
    Posting,
    SourceRef,
    Tier,
)
from laundromat.report import STATE, TraceIndex, _LOCK, app


KNOWN_AMOUNT = Decimal("19729014.76")


def _mini_dossier() -> Dossier:
    src = SourceRef(
        file="supporting_docs/draft_financials.pdf",
        page=1,
        excerpt=f"Sachanlagen {KNOWN_AMOUNT}",
    )
    ent = Entity(
        id="200099",
        type=EntityType.VENDOR,
        name="Nord Transport GmbH",
        source=SourceRef(file="vendors/suppliers.txt", line=2, excerpt="200099;Nord"),
    )
    post = Posting(
        doc_no="ER901427",
        booking_date=date(2025, 6, 15),
        amount=KNOWN_AMOUNT,
        account="040000",
        source=SourceRef(
            file="general_ledger/journal_entries.txt",
            line=42,
            excerpt=f"040000;{KNOWN_AMOUNT};ER901427",
        ),
        entity_id="200099",
        user="MV-U03",
        text="Anlage Zugang",
        attrs={"ledger": "GL"},
    )
    from laundromat.contracts import Document

    doc = Document(
        kind="financial_statements",
        ref="draft",
        source=src,
        amount=KNOWN_AMOUNT,
        fields={"text": f"Sachanlagen {KNOWN_AMOUNT}\nJahresueberschuss 2599841.80"},
    )
    return Dossier(
        name="mini",
        postings=[post],
        entities={"200099": ent},
        documents=[doc],
    )


def _seed_state(dossier: Dossier | None = None) -> Dossier:
    d = dossier or _mini_dossier()
    flag = Flag(
        lens_id="S_robust_outlier",
        family=LensFamily.STATISTICAL,
        title="Ausreisser-Betrag Test",
        rationale="fixture flag for UI smoke test",
        evidence=(d.postings[0].source,),
        entity_id="200099",
        doc_no="ER901427",
        amount=KNOWN_AMOUNT,
        confidence=0.4,
    )
    finding = Finding(
        subject_id="200099",
        subject_kind="entity",
        flags=[flag],
        tier=Tier.HIGH,
        score=4.2,
    )
    with _LOCK:
        STATE.update(
            ready=True,
            error=None,
            dossier=d,
            flags=[flag],
            findings=[finding],
            index=TraceIndex(d),
            run_id=None,
            dossier_path="tests/fixture-mini",
        )
    return d


@pytest.fixture
def client(monkeypatch, tmp_path):
    """TestClient without background practice load; STATE seeded with mini dossier."""

    def _noop_load() -> None:
        _seed_state()

    monkeypatch.setattr("laundromat.report._load", _noop_load)
    monkeypatch.setattr("laundromat.report.RUNS_DIR", tmp_path / "runs")
    # also patch module-level DOSSIER_PATH usage in coverage via STATE
    with TestClient(app) as c:
        _seed_state()  # ensure after lifespan
        yield c


def test_leaderboard_page_200_and_findings(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    body = r.text
    assert "Feststellungen" in body or "Laundromat" in body

    api = client.get("/api/findings")
    assert api.status_code == 200
    data = api.json()
    assert data["status"] == "ready"
    assert data.get("findings"), "expected at least one finding"
    assert any(
        f.get("subject_id") == "200099" or f.get("flag_count", 0) >= 1
        for f in data["findings"]
    )


def test_trace_resolves_known_amount(client: TestClient):
    # German-format amount string as an auditor would type it
    q = "19.729.014,76"
    r = client.get("/api/trace", params={"q": q})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ready"
    assert data.get("amount") is not None
    sections = data.get("sections") or []
    total_hits = sum(s.get("total", 0) for s in sections)
    assert total_hits >= 1, f"expected hits for {q}, got {data}"
    # at least one section should surface the GL or FS hit
    assert any(s.get("total", 0) > 0 for s in sections)


def test_upload_zip_creates_run(client: TestClient, tmp_path, monkeypatch):
    # build a tiny zip that ingest can open (empty-ish but valid directory tree)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "mini_dossier/readme.txt",
            "not a full gdpdu; pipeline should still create a run\n",
        )
        # one begleit-style csv so documents may load
        zf.writestr(
            "mini_dossier/credit_limits.csv",
            "DEBITOR;DEBITORNAME;KREDITLIMIT_EUR;AUSNUTZUNG_31_12_2025_EUR;STATUS;BESICHERUNG\n"
            "100000;TEST CO;10000,00;500,00;ok;none\n",
        )
    buf.seek(0)

    r = client.post(
        "/api/upload",
        files={"file": ("mini.zip", buf.getvalue(), "application/zip")},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "ready"
    assert data.get("run_id")
    assert "counts" in data
    # STATE should point at the new run
    with _LOCK:
        assert STATE.get("run_id") == data["run_id"]
        assert STATE.get("ready") is True
        assert STATE.get("dossier") is not None


def test_upload_rejects_non_zip(client: TestClient):
    r = client.post(
        "/api/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400
    assert r.json()["status"] == "error"

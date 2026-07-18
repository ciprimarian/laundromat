"""FastAPI smoke tests for the report UI (no browser)."""

from __future__ import annotations

import io
import time
import zipfile
from datetime import date
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from laundromat.contracts import (
    Document,
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
from laundromat.report import RUNS, TraceIndex, _LOCK, _new_state, app


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


def _seed_default_run(dossier: Dossier | None = None) -> Dossier:
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
    state = _new_state("mini", "tests/fixture-mini")
    state.update(
        ready=True,
        dossier=d,
        flags=[flag],
        findings=[finding],
        index=TraceIndex(d),
    )
    with _LOCK:
        RUNS["default"] = state
    return d


@pytest.fixture
def client(monkeypatch, tmp_path):
    """TestClient with the default practice load replaced by the mini fixture;
    uploaded runs still go through the real pipeline."""
    import laundromat.report as report

    real_load = report._load

    def _fake_load(run_id: str) -> None:
        if run_id == "default":
            _seed_default_run()
        else:
            real_load(run_id)

    monkeypatch.setattr("laundromat.report._load", _fake_load)
    monkeypatch.setattr("laundromat.report.RUNS_DIR", tmp_path / "runs")
    with TestClient(app) as c:
        _seed_default_run()  # ensure after lifespan
        yield c


def _wait_ready(client: TestClient, run_id: str, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        d = client.get("/api/coverage", params={"run": run_id}).json()
        if d.get("status") == "ready":
            return d
        if d.get("status") == "error":
            return d
        time.sleep(0.25)
    raise AssertionError(f"run {run_id} never became ready")


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


def test_upload_zip_creates_run(client: TestClient):
    """A zip upload creates a new run without touching the default run.

    The upload endpoint kicks off the real pipeline in a thread; the mini
    dossier has no GDPdU tables, which must not crash anything."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "mini_dossier/readme.txt",
            "not a full gdpdu; pipeline should still create a run\n",
        )
        zf.writestr(
            "mini_dossier/credit_limits.csv",
            "DEBITOR;DEBITORNAME;KREDITLIMIT_EUR;AUSNUTZUNG_31_12_2025_EUR;STATUS;BESICHERUNG\n"
            "100000;TEST CO;10000,00;500,00;ok;none\n",
        )

    r = client.post(
        "/upload",
        files=[("files", ("mini.zip", buf.getvalue(), "application/zip"))],
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "ok"
    run_id = data["run"]
    assert run_id and run_id != "default"

    cov = _wait_ready(client, run_id)
    assert cov["status"] == "ready", cov
    assert cov["counts"]["documents"] >= 1  # the credit limit csv

    # default run untouched
    default = client.get("/api/findings").json()
    assert default["status"] == "ready"
    assert default["dossier"] == "mini"


def test_upload_accepts_loose_files(client: TestClient):
    """The brief allows a zip OR multiple loose files."""
    r = client.post(
        "/upload",
        files=[
            (
                "files",
                (
                    "credit_limits.csv",
                    b"DEBITOR;DEBITORNAME;KREDITLIMIT_EUR;STATUS\n"
                    b"100000;TEST CO;10000,00;ok\n",
                    "text/csv",
                ),
            ),
        ],
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "ok"
    cov = _wait_ready(client, data["run"])
    assert cov["status"] == "ready", cov


def test_unknown_run_is_reported(client: TestClient):
    d = client.get("/api/findings", params={"run": "nope"}).json()
    assert d["status"] == "error"

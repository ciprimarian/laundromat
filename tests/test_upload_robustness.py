"""Hostile zip uploads must never 500 with a traceback."""

from __future__ import annotations

import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from laundromat.contracts import Dossier
from laundromat.report import STATE, TraceIndex, _LOCK, app


@pytest.fixture
def client(monkeypatch, tmp_path):
    def _noop() -> None:
        d = Dossier(name="seed")
        with _LOCK:
            STATE.update(
                ready=True,
                error=None,
                dossier=d,
                flags=[],
                findings=[],
                index=TraceIndex(d),
                run_id=None,
                dossier_path="seed",
            )

    monkeypatch.setattr("laundromat.report._load", _noop)
    monkeypatch.setattr("laundromat.report.RUNS_DIR", tmp_path / "runs")
    with TestClient(app) as c:
        d = Dossier(name="seed")
        with _LOCK:
            STATE.update(
                ready=True,
                error=None,
                dossier=d,
                flags=[],
                findings=[],
                index=TraceIndex(d),
                run_id=None,
                dossier_path="seed",
            )
        yield c


def _post_zip(client: TestClient, name: str, raw: bytes):
    return client.post(
        "/api/upload",
        files={"file": (name, raw, "application/zip")},
    )


def _assert_clean_json(r):
    assert r.status_code != 500, r.text
    assert "traceback" not in r.text.lower()
    assert "<html" not in r.text.lower() or r.headers.get("content-type", "").startswith(
        "application/json"
    )
    data = r.json()
    assert "status" in data
    if data["status"] == "error":
        assert data.get("error")
    return data


def test_unknown_files_only_zip(client: TestClient):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.xyz", b"not a table")
        zf.writestr("notes/unknown.bin", b"\x00\x01\x02")
    data = _assert_clean_json(_post_zip(client, "unknown.zip", buf.getvalue()))
    # empty/unknown dossier still produces a run
    assert data["status"] in {"ready", "error"}
    if data["status"] == "ready":
        assert data.get("run_id")
        assert "counts" in data


def test_latin1_filenames_in_zip(client: TestClient):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Begleit/Freigäbe.csv", "a;b\n1;2\n")
        zf.writestr("Krediören/x.txt", b"hello")
    data = _assert_clean_json(_post_zip(client, "latin1.zip", buf.getvalue()))
    assert data["status"] in {"ready", "error"}


def test_empty_zip(client: TestClient):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w"):
        pass
    data = _assert_clean_json(_post_zip(client, "empty.zip", buf.getvalue()))
    assert data["status"] in {"ready", "error"}


def test_nested_directories(client: TestClient):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "outer/inner/deep/credit_limits.csv",
            "DEBITOR;DEBITORNAME;KREDITLIMIT_EUR;AUSNUTZUNG_31_12_2025_EUR;STATUS;BESICHERUNG\n"
            "100001;NESTED CO;5000,00;100,00;ok;none\n",
        )
        zf.writestr("outer/inner/deep/notes.txt", "nested\n")
    data = _assert_clean_json(_post_zip(client, "nested.zip", buf.getvalue()))
    assert data["status"] in {"ready", "error"}
    if data["status"] == "ready":
        assert data.get("run_id")
        assert "counts" in data


def test_oversize_rejected_politely(client: TestClient):
    """60MB of zeros must hit the size cap with a clean JSON 400."""
    payload = b"0" * (60 * 1024 * 1024)
    r = _post_zip(client, "huge.zip", payload)
    assert r.status_code == 400, r.text[:500]
    data = r.json()
    assert data["status"] == "error"
    assert "MB" in data["error"]
    assert "traceback" not in r.text.lower()


def test_corrupt_zip_json_error(client: TestClient):
    r = _post_zip(client, "bad.zip", b"this is not a zip file at all")
    assert r.status_code == 400, r.text
    data = r.json()
    assert data["status"] == "error"
    assert "zip" in data["error"].lower() or "invalid" in data["error"].lower()
    assert "traceback" not in r.text.lower()


def test_non_zip_extension_rejected(client: TestClient):
    r = client.post(
        "/api/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400
    data = r.json()
    assert data["status"] == "error"

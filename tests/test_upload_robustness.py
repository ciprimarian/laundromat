"""Hostile uploads must never produce a traceback: clean JSON error or a working run."""

from __future__ import annotations

import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from laundromat.report import app


@pytest.fixture
def client(monkeypatch, tmp_path):
    import laundromat.report as report

    monkeypatch.setattr(report, "RUNS_DIR", tmp_path / "runs")
    # keep the default practice load out of these tests
    monkeypatch.setattr(report, "_load", lambda run_id: None)
    with TestClient(app) as c:
        yield c


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _post(client: TestClient, name: str, data: bytes):
    return client.post("/upload", files=[("files", (name, data, "application/octet-stream"))])


def test_invalid_zip_clean_error(client: TestClient):
    r = _post(client, "broken.zip", b"this is not a zip at all")
    assert r.status_code == 400
    assert r.json()["status"] == "error"


def test_zip_with_only_unknown_files(client: TestClient):
    data = _zip_bytes({"junk/readme.md": b"nothing auditable here\n"})
    r = _post(client, "junk.zip", data)
    assert r.status_code in (200, 400)
    assert r.json()["status"] in ("ok", "error")


def test_zip_with_latin1_filenames(client: TestClient):
    data = _zip_bytes({"dossier/Bestätigung_Prüfung.csv": b"KONTO;BETRAG\n1;2\n"})
    r = _post(client, "latin.zip", data)
    assert r.status_code in (200, 400)
    assert r.json()["status"] in ("ok", "error")


def test_nested_directories(client: TestClient):
    data = _zip_bytes({f"a/b/c/d/e/f{i}.txt": b"x;y\n" for i in range(5)})
    r = _post(client, "nested.zip", data)
    assert r.status_code in (200, 400)
    assert r.json()["status"] in ("ok", "error")


def test_empty_upload_rejected(client: TestClient):
    r = _post(client, "empty.zip", b"")
    assert r.status_code == 400
    assert r.json()["status"] == "error"


def test_size_cap_rejects_politely(client: TestClient, monkeypatch):
    import laundromat.report as report

    monkeypatch.setattr(report, "MAX_UPLOAD", 1024)
    data = _zip_bytes({"big.txt": b"0" * 10_000})
    r = _post(client, "big.zip", data)
    assert r.status_code == 400
    body = r.json()
    assert body["status"] == "error"


def test_zip_slip_path_traversal_blocked(client: TestClient, tmp_path):
    # an entry attempting to escape the extraction dir must not land outside
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../../evil.txt", b"escaped")
    r = _post(client, "slip.zip", buf.getvalue())
    assert r.status_code in (200, 400)
    escaped = list(tmp_path.parent.glob("evil.txt"))
    assert not escaped, "zip slip escaped the run directory"

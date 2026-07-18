"""Report UI: FastAPI app serving the findings leaderboard, figure tracer
and coverage panel over one loaded dossier.

Run: uvicorn laundromat.report:app  (Dockerfile uses laundromat.report.app:app,
a shim module that re-exports the same instance).

Dossier path: env CORTEA_DOSSIER, then DOSSIER_DIR (compose), then data/practice.
The pipeline loads in a background thread at startup so the server binds
immediately; endpoints answer {"status": "loading"} until it finishes.
"""

from __future__ import annotations

import bisect
import os
import re
import threading
import uuid
import zipfile
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from ..contracts import Dossier, Finding, Flag, SourceRef

DOSSIER_PATH = (
    os.environ.get("CORTEA_DOSSIER")
    or os.environ.get("DOSSIER_DIR")
    or "data/practice"
)
RUNS_DIR = Path(os.environ.get("CORTEA_RUNS_DIR") or "data/runs")
MAX_UPLOAD = 300 * 1024 * 1024  # bytes, compressed

CENT = Decimal("0.01")

# One state dict per run id; "default" is the preloaded practice dossier.
RUNS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


def _new_state(name: str, path: str) -> dict[str, Any]:
    return {
        "ready": False,
        "error": None,
        "dossier": None,
        "flags": [],
        "findings": [],
        "index": None,
        "name": name,
        "path": path,
    }


# --------------------------------------------------------------------------
# Amount parsing / formatting (German and English formats).
# --------------------------------------------------------------------------


def parse_amount(raw: str) -> Decimal | None:
    """'1.234.567,89', '1,234,567.89', '1234567.89', '-47244,00' -> Decimal(abs)."""
    t = raw.strip()
    t = re.sub(r"(?i)(EUR|USD|GBP|CHF|€|\$|£)", "", t).strip()
    t = t.lstrip("+-").strip()
    if not t or not re.fullmatch(r"[\d.,]+", t):
        return None
    if "," in t and "." in t:
        if t.rfind(",") > t.rfind("."):
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", "")
    elif "," in t:
        head, _, tail = t.rpartition(",")
        if t.count(",") == 1 and len(tail) <= 2:
            t = head + "." + tail
        else:
            t = t.replace(",", "")
    elif "." in t:
        head, _, tail = t.rpartition(".")
        if t.count(".") > 1 or (len(tail) == 3 and len(head) <= 3):
            t = t.replace(".", "")
    try:
        return abs(Decimal(t)).quantize(CENT)
    except InvalidOperation:
        return None


def fmt_de(x: Decimal | None) -> str | None:
    if x is None:
        return None
    s = f"{x:,.2f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


_NUM_RE = re.compile(r"\d[\d.,]*\d|\d")


def line_amounts(line: str) -> list[Decimal]:
    out = []
    for tok in _NUM_RE.findall(line):
        v = parse_amount(tok)
        if v is not None:
            out.append(v)
    return out


# --------------------------------------------------------------------------
# Trace index, built once after load.
# --------------------------------------------------------------------------


class TraceIndex:
    def __init__(self, dossier: Dossier) -> None:
        self.by_amount: dict[Decimal, list[tuple[str, int]]] = defaultdict(list)
        self.entity_names: list[tuple[str, str]] = []  # (id, casefolded name)
        self.doc_amounts: dict[int, set[Decimal]] = {}

        for i, p in enumerate(dossier.postings):
            self.by_amount[abs(p.amount).quantize(CENT)].append(("p", i))

        for i, doc in enumerate(dossier.documents):
            vals: set[Decimal] = set()
            if doc.amount is not None:
                vals.add(abs(doc.amount).quantize(CENT))
            for k, v in doc.fields.items():
                if k == "text" or not v:
                    continue
                a = parse_amount(v)
                if a is not None and a != 0:
                    vals.add(a)
            self.doc_amounts[i] = vals
            for a in vals:
                self.by_amount[a].append(("d", i))

        self.sorted_amounts = sorted(self.by_amount)

        for e in dossier.entities.values():
            self.entity_names.append((e.id, (e.name or "").casefold()))

    def amount_refs(self, lo: Decimal, hi: Decimal) -> list[tuple[Decimal, str, int]]:
        out = []
        i = bisect.bisect_left(self.sorted_amounts, lo)
        j = bisect.bisect_right(self.sorted_amounts, hi)
        for a in self.sorted_amounts[i:j]:
            for kind, idx in self.by_amount[a]:
                out.append((a, kind, idx))
        return out


# --------------------------------------------------------------------------
# Startup: run the pipeline in a background thread.
# --------------------------------------------------------------------------


def _load(run_id: str) -> None:
    with _LOCK:
        state = RUNS[run_id]
    try:
        from ..pipeline import run

        dossier, flags, findings = run(state["path"])
        index = TraceIndex(dossier)
        with _LOCK:
            state.update(dossier=dossier, flags=flags, findings=findings, index=index)
    except Exception as e:  # never let the server die on a bad dossier
        with _LOCK:
            state["error"] = f"{type(e).__name__}: {e}"
    finally:
        with _LOCK:
            state["ready"] = True


def _start_run(run_id: str, name: str, path: str) -> None:
    with _LOCK:
        RUNS[run_id] = _new_state(name, path)
    threading.Thread(target=_load, args=(run_id,), daemon=True).start()


@asynccontextmanager
async def _lifespan(_: FastAPI):
    _start_run("default", Path(DOSSIER_PATH).name, DOSSIER_PATH)
    yield


app = FastAPI(title="Laundromat Bericht", lifespan=_lifespan)


def _state(run: str) -> dict[str, Any] | None:
    with _LOCK:
        return RUNS.get(run)


def _not_ready(run: str) -> JSONResponse | None:
    state = _state(run)
    if state is None:
        return JSONResponse({"status": "error", "error": f"unbekannter Lauf '{run}'"})
    with _LOCK:
        if not state["ready"]:
            return JSONResponse({"status": "loading", "dossier_path": state["path"]})
        if state["error"]:
            return JSONResponse({"status": "error", "error": state["error"]})
    return None


# --------------------------------------------------------------------------
# Serialization.
# --------------------------------------------------------------------------


def ref_json(r: SourceRef) -> dict:
    return {
        "file": r.file,
        "line": r.line,
        "page": r.page,
        "sheet": r.sheet,
        "excerpt": r.excerpt,
        "cite": r.cite(),
    }


def flag_json(f: Flag) -> dict:
    return {
        "lens_id": f.lens_id,
        "family": f.family.value,
        "title": f.title,
        "rationale": f.rationale,
        "entity_id": f.entity_id,
        "doc_no": f.doc_no,
        "amount": fmt_de(f.amount),
        "confidence": f.confidence,
        "evidence": [ref_json(r) for r in f.evidence],
    }


def _subject_name(dossier: Dossier, subject_id: str) -> str | None:
    e = dossier.entities.get(subject_id)
    return e.name if e else None


def _finding_json(dossier: Dossier, f: Finding) -> dict:
    return {
        "subject_id": f.subject_id,
        "subject_kind": f.subject_kind,
        "subject_name": _subject_name(dossier, f.subject_id),
        "tier": f.tier.value,
        "score": f.score,
        "families": sorted(fam.value for fam in f.families),
        "flag_count": len(f.flags),
        "max_amount": fmt_de(f.max_amount) if f.max_amount else None,
        "defense_note": f.defense_note,
        "flags": [flag_json(fl) for fl in f.flags],
    }


def _group_flags_only(dossier: Dossier, flags: list[Flag]) -> list[dict]:
    """Flags-only mode: group per subject ourselves, sorted by corroboration."""
    by_subject: dict[tuple[str, str], list[Flag]] = defaultdict(list)
    for fl in flags:
        if fl.entity_id:
            by_subject[("entity", fl.entity_id)].append(fl)
        if fl.doc_no:
            by_subject[("transaction", fl.doc_no)].append(fl)
        if not fl.entity_id and not fl.doc_no:
            by_subject[("lens", fl.lens_id)].append(fl)

    out = []
    for (kind, sid), fls in by_subject.items():
        fams = sorted({f.family.value for f in fls})
        max_amount = max((abs(f.amount) for f in fls if f.amount is not None), default=None)
        out.append(
            {
                "subject_id": sid,
                "subject_kind": kind,
                "subject_name": _subject_name(dossier, sid),
                "tier": None,
                "score": None,
                "families": fams,
                "flag_count": len(fls),
                "max_amount": fmt_de(max_amount),
                "defense_note": None,
                "flags": [flag_json(f) for f in fls],
                "_sort": (len(fams), len(fls), max_amount or Decimal(0)),
            }
        )
    out.sort(key=lambda d: d["_sort"], reverse=True)
    for d in out:
        del d["_sort"]
    return out


# --------------------------------------------------------------------------
# Endpoints.
# --------------------------------------------------------------------------


@app.get("/api/findings")
def api_findings(run: str = "default"):
    if (r := _not_ready(run)) is not None:
        return r
    state = _state(run)
    dossier: Dossier = state["dossier"]
    findings: list[Finding] = state["findings"]
    flags: list[Flag] = state["flags"]
    if findings:
        return {
            "status": "ready",
            "mode": "scored",
            "dossier": dossier.name,
            "findings": [_finding_json(dossier, f) for f in findings],
        }
    return {
        "status": "ready",
        "mode": "flags_only",
        "dossier": dossier.name,
        "findings": _group_flags_only(dossier, flags),
    }


MAX_HITS = 100


def _hit(src: SourceRef, matches: list[str], excerpt: str | None = None, label: str | None = None) -> dict:
    h = ref_json(src)
    if excerpt is not None:
        h["excerpt"] = excerpt
    h["match"] = matches
    h["label"] = label
    return h


@app.get("/api/trace")
def api_trace(q: str = "", run: str = "default"):
    if (r := _not_ready(run)) is not None:
        return r
    q = q.strip()
    if not q:
        return {"status": "ready", "query": q, "sections": [], "error": "Leere Anfrage"}

    state = _state(run)
    dossier: Dossier = state["dossier"]
    index: TraceIndex = state["index"]
    ql = q.casefold()

    # Amount interpretation. Leading zero with no separators reads as an id,
    # not an amount (account and entity numbers like 020000).
    amount = None
    if not (q.lstrip("+-").startswith("0") and "," not in q and "." not in q):
        amount = parse_amount(q)
    lo = hi = None
    if amount is not None and amount > 0:
        lo = (amount * Decimal("0.99")).quantize(CENT)
        hi = (amount * Decimal("1.01")).quantize(CENT)

    # Entities whose id or name matches the query.
    ent_ids: set[str] = set()
    if len(ql) >= 3:
        for eid, name in index.entity_names:
            if ql == eid.casefold() or ql in name:
                ent_ids.add(eid)
    else:
        ent_ids = {eid for eid, _ in index.entity_names if ql == eid.casefold()}

    # Amount candidates from the index.
    amount_hits: dict[tuple[str, int], str] = {}
    if amount is not None and lo is not None:
        for a, kind, idx in index.amount_refs(lo, hi):
            tag = "Betrag exakt" if a == amount else "Betrag ±1%"
            prev = amount_hits.get((kind, idx))
            if prev != "Betrag exakt":
                amount_hits[(kind, idx)] = tag

    # --- Postings scan (account, entity, doc_no, text, attrs) ---
    gl_hits: list[dict] = []
    sub_hits: list[dict] = []
    gl_total = sub_total = 0
    for i, p in enumerate(dossier.postings):
        matches: list[str] = []
        tag = amount_hits.get(("p", i))
        if tag:
            matches.append(tag)
        if q == p.account or q == (p.counter_account or "") or q == p.attrs.get("account_base", ""):
            matches.append("Konto")
        if p.entity_id and (p.entity_id in ent_ids or p.entity_id.casefold() == ql):
            matches.append("Entität")
        dn = p.doc_no.casefold()
        if ql == dn or (len(ql) >= 4 and ql in dn):
            matches.append("Beleg-Nr.")
        if len(ql) >= 3 and ql in p.text.casefold():
            matches.append("Text")
        if len(ql) >= 3 and not matches and any(v == q for v in p.attrs.values()):
            matches.append("Feld")
        if not matches:
            continue
        is_gl = p.attrs.get("ledger", "GL") == "GL"
        label = f"{p.booking_date} | Konto {p.account} | {fmt_de(p.amount)} {p.currency} | {p.doc_no}"
        if is_gl:
            gl_total += 1
            if len(gl_hits) < MAX_HITS:
                gl_hits.append(_hit(p.source, matches, label=label))
        else:
            sub_total += 1
            if len(sub_hits) < MAX_HITS:
                sub_hits.append(_hit(p.source, matches, label=label))

    # --- Documents scan ---
    fs_hits: list[dict] = []
    tb_hits: list[dict] = []
    doc_hits: list[dict] = []
    fs_total = tb_total = doc_total = 0
    for i, doc in enumerate(dossier.documents):
        if doc.kind == "financial_statements":
            text = doc.fields.get("text", "")
            for ln in text.splitlines():
                lmatch: list[str] = []
                if amount is not None and lo is not None:
                    for a in line_amounts(ln):
                        if a == amount:
                            lmatch.append("Betrag exakt")
                            break
                        if lo <= a <= hi:
                            lmatch.append("Betrag ±1%")
                            break
                if len(ql) >= 3 and ql in ln.casefold():
                    lmatch.append("Text")
                if lmatch:
                    fs_total += 1
                    if len(fs_hits) < MAX_HITS:
                        fs_hits.append(_hit(doc.source, lmatch, excerpt=ln.strip()))
            continue

        matches = []
        tag = amount_hits.get(("d", i))
        if tag:
            matches.append(tag)
        rf = doc.ref.casefold()
        if ql == rf or (len(ql) >= 4 and ql in rf):
            matches.append("Beleg-Nr.")
        if doc.entity_id and (doc.entity_id in ent_ids or doc.entity_id.casefold() == ql):
            matches.append("Entität")
        if len(ql) >= 3 and not matches:
            for k, v in doc.fields.items():
                if k != "text" and v and ql in v.casefold():
                    matches.append("Feld")
                    break
        if not matches:
            continue
        label = f"{doc.kind} | {doc.ref}" + (f" | {fmt_de(doc.amount)}" if doc.amount is not None else "")
        if doc.kind == "trial_balance":
            tb_total += 1
            if len(tb_hits) < MAX_HITS:
                tb_hits.append(_hit(doc.source, matches, label=label))
        else:
            doc_total += 1
            if len(doc_hits) < MAX_HITS:
                doc_hits.append(_hit(doc.source, matches, label=label))

    # --- Entity master records ---
    ent_hits = []
    for eid in sorted(ent_ids):
        e = dossier.entities[eid]
        if len(ent_hits) < MAX_HITS:
            ent_hits.append(
                _hit(e.source, ["Entität"], label=f"{e.type.value} {e.id} | {e.name}")
            )

    sections = [
        {"id": "fs", "label": "Jahresabschluss", "total": fs_total, "hits": fs_hits},
        {"id": "tb", "label": "Saldenliste", "total": tb_total, "hits": tb_hits},
        {"id": "gl", "label": "Hauptbuch", "total": gl_total, "hits": gl_hits},
        {"id": "sub", "label": "Nebenbücher (AP/AR/FA)", "total": sub_total, "hits": sub_hits},
        {"id": "docs", "label": "Belege und Dokumente", "total": doc_total, "hits": doc_hits},
        {"id": "ent", "label": "Stammdaten", "total": len(ent_hits), "hits": ent_hits},
    ]
    return {
        "status": "ready",
        "query": q,
        "amount": fmt_de(amount),
        "sections": sections,
    }


@app.get("/api/coverage")
def api_coverage(run: str = "default"):
    if (r := _not_ready(run)) is not None:
        return r
    state = _state(run)
    dossier: Dossier = state["dossier"]
    flags: list[Flag] = state["flags"]

    from ..contracts import REGISTRY

    try:
        from ..lenses import FAILED
        failed = [{"module": m, "reason": why} for m, why in FAILED]
    except Exception:
        failed = []

    per_lens = Counter(f.lens_id for f in flags)
    return {
        "status": "ready",
        "dossier": dossier.name,
        "dossier_path": state["path"],
        "auditor": state.get("auditor"),
        "counts": {
            "postings": len(dossier.postings),
            "entities": len(dossier.entities),
            "documents": len(dossier.documents),
            "source_files": len(
                {p.source.file for p in dossier.postings}
                | {d.source.file for d in dossier.documents}
                | {e.source.file for e in dossier.entities.values()}
            ),
            "flags": len(flags),
            "findings": len(state["findings"]),
        },
        "unparsed": [{"file": f, "reason": r} for f, r in dossier.unparsed],
        "lenses_failed": failed,
        "lenses": [
            {"lens_id": lid, "family": lens.family.value, "flags": per_lens.get(lid, 0)}
            for lid, lens in sorted(REGISTRY.items())
        ],
        "documents_per_kind": dict(sorted(Counter(d.kind for d in dossier.documents).items())),
        "postings_per_ledger": dict(
            sorted(Counter(p.attrs.get("ledger", "GL") for p in dossier.postings).items())
        ),
    }


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract while refusing paths that escape dest (zip slip)."""
    base = dest.resolve()
    for info in zf.infolist():
        target = (dest / info.filename).resolve()
        if base != target and base not in target.parents:
            raise ValueError(f"unsicherer Pfad im Archiv: {info.filename}")
    zf.extractall(dest)


def _dossier_root(dest: Path) -> Path:
    """A zip usually wraps everything in one folder; use it as the root."""
    entries = [p for p in dest.iterdir() if not p.name.startswith(("__MACOSX", "."))]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return dest


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...), auditor: str = Form("")):
    run_id = uuid.uuid4().hex[:8]
    dest = RUNS_DIR / run_id
    try:
        dest.mkdir(parents=True, exist_ok=True)
        total = 0
        saved: list[Path] = []
        for uf in files:
            name = Path(uf.filename or "upload.bin").name  # strip any client path
            target = dest / name
            with open(target, "wb") as out:
                while chunk := await uf.read(1 << 20):
                    total += len(chunk)
                    if total > MAX_UPLOAD:
                        raise ValueError("Upload ueberschreitet das Groessenlimit")
                    out.write(chunk)
            saved.append(target)
        for path in saved:
            if path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as zf:
                    _safe_extract(zf, dest)
                path.unlink()
        root = _dossier_root(dest)
        _start_run(run_id, root.name, str(root))
        if auditor.strip():
            with _LOCK:
                RUNS[run_id]["auditor"] = auditor.strip()[:80]
        return {"status": "ok", "run": run_id}
    except Exception as e:
        return JSONResponse({"status": "error", "error": f"{type(e).__name__}: {e}"}, status_code=400)


@app.get("/", response_class=HTMLResponse)
def index_page():
    return HTMLResponse(PAGE)


# --------------------------------------------------------------------------
# The page. Self-contained: inline CSS + JS, no external requests.
# --------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Laundromat</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><circle cx='13' cy='13' r='8' fill='none' stroke='%230b5cad' stroke-width='3.5'/><line x1='19.5' y1='19.5' x2='28' y2='28' stroke='%230b5cad' stroke-width='3.5' stroke-linecap='round'/></svg>">
<style>
:root { --bg:#f6f7f9; --card:#ffffff; --ink:#1a232e; --mut:#5c6b7a; --line:#dde3ea;
        --acc:#0b5cad; --hi:#b42318; --med:#b54708; --rev:#5c6b7a; --ok:#067647; }
* { box-sizing:border-box; margin:0; padding:0; }
body { background:var(--bg); color:var(--ink);
       font:15px/1.45 -apple-system,"Segoe UI",Roboto,Arial,sans-serif; }
header { background:var(--card); border-bottom:1px solid var(--line);
         padding:12px 24px; display:flex; align-items:center; gap:18px; }
header h1 { font-size:19px; line-height:1.1; }
header .tag { color:var(--mut); font-size:12px; font-style:italic; }
header .sub { color:var(--mut); font-size:13px; }
.langbtn { border:1px solid var(--line); background:var(--card); color:var(--mut);
           border-radius:6px; padding:6px 10px; font-size:12px; font-weight:600;
           cursor:pointer; }
.langbtn:hover { color:var(--acc); border-color:var(--acc); }
nav { display:flex; gap:4px; padding:10px 24px 0; }
nav button { border:1px solid var(--line); border-bottom:none; background:#eef1f5;
             padding:8px 18px; font-size:14px; cursor:pointer; border-radius:6px 6px 0 0; }
nav button.on { background:var(--card); font-weight:600; color:var(--acc); }
main { padding:16px 24px 60px; max-width:1200px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:6px;
        margin-bottom:10px; }
.row { padding:10px 14px; cursor:pointer; display:flex; gap:12px; align-items:center;
       flex-wrap:wrap; }
.row:hover { background:#f2f6fa; }
.badge { display:inline-block; font-size:11px; padding:1px 8px; border-radius:9px;
         border:1px solid var(--line); color:var(--mut); white-space:nowrap; }
.badge.tier-high { color:#fff; background:var(--hi); border-color:var(--hi); }
.badge.tier-medium { color:#fff; background:var(--med); border-color:var(--med); }
.badge.tier-review { color:#fff; background:var(--rev); border-color:var(--rev); }
.badge.tier-dismissed { text-decoration:line-through; }
.badge.m { color:var(--acc); border-color:var(--acc); }
.amt { font-variant-numeric:tabular-nums; font-weight:600; }
.mut { color:var(--mut); font-size:13px; }
.body { border-top:1px solid var(--line); padding:10px 14px; display:none; }
.flag { border:1px solid var(--line); border-radius:5px; margin:8px 0; }
.flag > .fhead { padding:8px 12px; cursor:pointer; }
.flag > .fhead:hover { background:#f2f6fa; }
.flag .fbody { display:none; border-top:1px solid var(--line); padding:8px 12px; }
.ev { margin:6px 0; }
.cite { font:12px ui-monospace,Menlo,Consolas,monospace; color:var(--acc); }
a.cite { cursor:pointer; text-decoration:underline dotted; }
pre.x { font:12px ui-monospace,Menlo,Consolas,monospace; background:#f2f4f7;
        border:1px solid var(--line); border-radius:4px; padding:6px 8px; margin-top:3px;
        white-space:pre-wrap; word-break:break-all; }
#traceq { width:min(560px,90%); padding:9px 12px; font-size:15px;
          border:1px solid var(--line); border-radius:6px; }
#tracebtn { padding:9px 18px; font-size:15px; border:1px solid var(--acc);
            background:var(--acc); color:#fff; border-radius:6px; cursor:pointer; }
h2.sec { font-size:15px; margin:18px 0 6px; }
h2.sec .mut { font-weight:400; }
table { border-collapse:collapse; width:100%; font-size:13px; }
th,td { text-align:left; padding:5px 10px; border-bottom:1px solid var(--line); }
th { color:var(--mut); font-weight:600; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
.note { color:var(--mut); font-size:13px; margin:8px 0; }
.err { color:var(--hi); }
.hint { color:var(--mut); font-size:13px; margin-top:6px; }
#dropzone { border:2px dashed var(--line); border-radius:8px; padding:8px 16px;
            color:var(--mut); font-size:13px; cursor:pointer; user-select:none; }
#dropzone.over { border-color:var(--acc); color:var(--acc); background:#eef4fb; }
#dropzone.busy { border-style:solid; color:var(--acc); }
.strip { display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-bottom:14px; }
.chip { border-radius:8px; padding:7px 14px; font-size:14px; background:var(--card);
        border:1px solid var(--line); }
.chip b { font-size:17px; margin-right:5px; }
.chip.k { border-color:var(--hi); color:var(--hi); }
.chip.m { border-color:var(--med); color:var(--med); }
.chip.e { border-color:var(--ok); color:var(--ok); }
.chip.p { color:var(--mut); }
.strip .vol { color:var(--mut); font-size:13px; margin-left:auto; }
.headline { font-size:15px; }
.subline { margin-top:2px; font-size:13px; color:var(--mut); }
.subline b { color:var(--ink); font-size:14px; }
.score { margin-left:auto; text-align:right; font-variant-numeric:tabular-nums;
         font-weight:600; color:var(--acc); white-space:nowrap; }
.score .mut { display:block; font-weight:400; }
#auditor { border:1px solid var(--line); border-radius:8px; padding:8px 10px;
           font-size:13px; width:150px; }
footer { border-top:1px solid var(--line); padding:14px 24px; color:var(--mut);
         font-size:12px; }
footer a { color:var(--mut); }
</style>
</head>
<body>
<header>
  <div>
    <h1>Laundromat</h1>
    <div class="tag">no number without a source</div>
  </div>
  <span class="sub" id="dossiername"></span>
  <span class="sub" id="status"></span>
  <span style="flex:1"></span>
  <button class="langbtn" id="langbtn" onclick="toggleLang()">DE</button>
  <input id="auditor" maxlength="80">
  <div id="dropzone" title=""></div>
  <input type="file" id="filepick" multiple style="display:none">
</header>
<nav>
  <button id="tab-f" class="on" onclick="show('f')"></button>
  <button id="tab-t" onclick="show('t')"></button>
  <button id="tab-c" onclick="show('c')"></button>
</nav>
<main>
  <section id="view-f"><div class="note" id="loading-f"></div></section>
  <section id="view-t" style="display:none">
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      <input id="traceq" onkeydown="if(event.key==='Enter')trace()">
      <button id="tracebtn" onclick="trace()"></button>
    </div>
    <div class="hint" id="tracehint"></div>
    <div id="traceout"></div>
  </section>
  <section id="view-c" style="display:none"><div class="note"></div></section>
</main>
<script>
"use strict";
const RUN = new URLSearchParams(location.search).get("run") || "default";
function api(path){ return path + (path.includes("?") ? "&" : "?") + "run=" + encodeURIComponent(RUN); }
function esc(s){return String(s==null?"":s).replace(/[&<>"']/g,
  c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}

// -------------------------------------------------------------- i18n chrome.
// Lens output (titles, rationales, excerpts) is never translated.
const I18N = {
en: {
  tab_f:"Findings", tab_t:"Figure Tracer", tab_c:"Coverage",
  drop_idle:"Audit a dossier: drop zip here", drop_busy:"Uploading ...",
  drop_fail:"Upload failed", run_label:"Run", err:"Error",
  auditor:"Auditor", auditor_ph:"Auditor name (optional)",
  loading:"Loading dossier", critical:"critical", medium:"medium",
  cleared:"checked & cleared", review:"under review",
  postings:"postings", entities:"entities", docs_examined:"documents examined",
  mode_note:"Scoring not active yet: sorted by number of independent audit methods and flags per subject.",
  no_findings:"No findings.",
  kind_entity:"Entity", kind_transaction:"Transaction", kind_lens:"Grouped",
  method_one:"independent audit method", methods:"independent audit methods",
  flag_one:"flag", flags:"flags", score:"Score",
  defense:"Defense check", confidence:"Confidence",
  sources:"Sources", sources_click:"click for original excerpt", no_excerpt:"(no excerpt)",
  trace_ph:"Amount (e.g. 19.729.014,76), account, doc no. or name",
  trace_btn:"Trace it",
  trace_hint:"Traces any figure from the financial statements through trial balance and general ledger into subledgers and supporting documents. German or English number format, exact and \u00b11%.",
  trace_searching:"Searching ...", trace_amount:"Interpreted as amount",
  trace_hits:"hits", trace_first:"first", trace_shown:"shown", trace_none:"No hits for",
  lenses:"Audit lenses", loaded:"loaded", failed:"failed",
  th_lens:"Lens", th_family:"Family", th_flags:"Flags", th_module:"Module",
  th_error:"Error", th_file:"File", th_reason:"Reason",
  cov_docs:"Documents per kind", cov_ledger:"Postings per ledger",
  cov_unparsed:"Unparsed files", cov_all:"All files parsed.",
  c_postings:"postings", c_entities:"entities", c_documents:"documents",
  c_flags:"flags", c_findings:"findings",
  tier:{high:"critical", medium:"medium", review:"under review", dismissed:"cleared"},
  fam:{rule:"Rule check", statistical:"Statistics", graph:"Graph analysis",
       temporal:"Timing analysis", reconciliation:"Reconciliation",
       semantic:"Semantics", external:"Web search"},
  match:{"Betrag exakt":"Exact amount","Betrag \u00b11%":"Amount \u00b11%","Konto":"Account",
         "Entit\u00e4t":"Entity","Beleg-Nr.":"Doc no.","Text":"Text","Feld":"Field"},
  sec:{fs:"Financial statements", tb:"Trial balance", gl:"General ledger",
       sub:"Subledgers (AP/AR/FA)", docs:"Supporting documents", ent:"Master data"},
},
de: {
  tab_f:"Feststellungen", tab_t:"Zahlenspur", tab_c:"Abdeckung",
  drop_idle:"Dossier pruefen: Zip hier ablegen", drop_busy:"Wird hochgeladen ...",
  drop_fail:"Upload fehlgeschlagen", run_label:"Lauf", err:"Fehler",
  auditor:"Pruefer", auditor_ph:"Pruefername (optional)",
  loading:"Dossier wird geladen", critical:"kritisch", medium:"mittel",
  cleared:"geprueft & entlastet", review:"in Pruefung",
  postings:"Buchungen", entities:"Entitaeten", docs_examined:"Dokumente geprueft",
  mode_note:"Scoring noch nicht aktiv: Sortierung nach Zahl unabhaengiger Pruefmethoden und Befunden je Subjekt.",
  no_findings:"Keine Feststellungen.",
  kind_entity:"Entitaet", kind_transaction:"Transaktion", kind_lens:"Sammelbefund",
  method_one:"unabhaengige Pruefmethode", methods:"unabhaengige Pruefmethoden",
  flag_one:"Befund", flags:"Befunde", score:"Score",
  defense:"Entlastungspruefung", confidence:"Konfidenz",
  sources:"Quellen", sources_click:"anklicken fuer Originalauszug", no_excerpt:"(kein Auszug)",
  trace_ph:"Betrag (z.B. 19.729.014,76), Konto, Beleg-Nr. oder Name",
  trace_btn:"Spur verfolgen",
  trace_hint:"Verfolgt jede Zahl vom Jahresabschluss ueber Saldenliste und Hauptbuch bis in Nebenbuecher und Belege. Betraege in deutschem oder englischem Format, exakt und \u00b11%.",
  trace_searching:"Suche ...", trace_amount:"Interpretiert als Betrag",
  trace_hits:"Treffer", trace_first:"erste", trace_shown:"gezeigt", trace_none:"Keine Treffer fuer",
  lenses:"Pruefmethoden", loaded:"geladen", failed:"fehlgeschlagen",
  th_lens:"Lens", th_family:"Familie", th_flags:"Befunde", th_module:"Modul",
  th_error:"Fehler", th_file:"Datei", th_reason:"Grund",
  cov_docs:"Dokumente je Art", cov_ledger:"Buchungen je Buch",
  cov_unparsed:"Nicht verarbeitete Dateien", cov_all:"Alle Dateien verarbeitet.",
  c_postings:"Buchungen", c_entities:"Entitaeten", c_documents:"Dokumente",
  c_flags:"Befunde", c_findings:"Feststellungen",
  tier:{high:"kritisch", medium:"mittel", review:"in Pruefung", dismissed:"entlastet"},
  fam:{rule:"Regelpruefung", statistical:"Statistik", graph:"Graphanalyse",
       temporal:"Zeitanalyse", reconciliation:"Abstimmung",
       semantic:"Semantik", external:"Websuche"},
  match:{},
  sec:{fs:"Jahresabschluss", tb:"Saldenliste", gl:"Hauptbuch",
       sub:"Nebenbuecher (AP/AR/FA)", docs:"Belege und Dokumente", ent:"Stammdaten"},
},
};
let LANG = localStorage.getItem("lang") || "en";
function t(k){ const v = I18N[LANG][k]; return v!==undefined?v:(I18N.en[k]!==undefined?I18N.en[k]:k); }
function tm(map, k){ const m = t(map); return (m && m[k]!==undefined)?m[k]:k; }
function toggleLang(){
  LANG = LANG==="en"?"de":"en";
  localStorage.setItem("lang", LANG);
  applyChrome(); loadFindings(); loadCoverage();
  if(document.getElementById("traceq").value.trim()) trace();
}
function applyChrome(){
  document.getElementById("langbtn").textContent = LANG==="en"?"DE":"EN";
  document.getElementById("tab-f").textContent = t("tab_f");
  document.getElementById("tab-t").textContent = t("tab_t");
  document.getElementById("tab-c").textContent = t("tab_c");
  const dz = document.getElementById("dropzone");
  if(!dz.classList.contains("busy")) dz.textContent = t("drop_idle");
  dz.title = t("drop_idle");
  document.getElementById("traceq").placeholder = t("trace_ph");
  document.getElementById("auditor").placeholder = t("auditor_ph");
  document.getElementById("tracebtn").textContent = t("trace_btn");
  document.getElementById("tracehint").textContent = t("trace_hint");
  if(RUN !== "default")
    document.getElementById("status").textContent = t("run_label") + " " + RUN;
}

function show(v){
  for(const k of ["f","t","c"]){
    document.getElementById("view-"+k).style.display = k===v?"":"none";
    document.getElementById("tab-"+k).className = k===v?"on":"";
  }
}
async function getJSON(url){
  const r = await fetch(url);
  return await r.json();
}
function deNum(s){ return s ? parseFloat(String(s).replace(/\./g,"").replace(",",".")) || 0 : 0; }
function evHtml(refs){
  return refs.map(e=>`<div class="ev"><a class="cite" onclick="togNext(this)">${esc(e.cite)}`+
    `${e.sheet?" ["+esc(e.sheet)+"]":""}</a>`+
    `<pre class="x" style="display:none">${esc(e.excerpt)||esc(t("no_excerpt"))}</pre></div>`).join("");
}
function flagHtml(fl){
  const conf = Math.round(fl.confidence*100);
  return `<div class="flag"><div class="fhead" onclick="tog(this)">
    <b>${esc(fl.title)}</b>
    <div class="mut">${esc(tm("fam", fl.family))} \u00b7 ${esc(t("confidence"))} ${conf}%`+
    `${fl.amount?" \u00b7 <span class='amt'>"+esc(fl.amount)+" EUR</span>":""}</div></div>
    <div class="fbody"><div>${esc(fl.rationale)}</div>
    <div class="mut" style="margin-top:6px">${esc(t("sources"))} (${fl.evidence.length}) &ndash; ${esc(t("sources_click"))}:</div>
    ${evHtml(fl.evidence)}</div></div>`;
}
function tog(el){
  const b = el.parentElement.querySelector(".fbody, .body");
  if(b) b.style.display = b.style.display==="block"?"none":"block";
}
function togNext(a){
  const p = a.nextElementSibling;
  if(p) p.style.display = p.style.display==="none"?"block":"none";
}
function strongest(flags){
  return flags.slice().sort((a,b)=>(b.confidence-a.confidence)||(deNum(b.amount)-deNum(a.amount)))[0];
}
function stripHtml(findings, counts){
  const n = x => findings.filter(f=>f.tier===x).length;
  const fmt = x => (x==null?"?":x.toLocaleString(LANG==="de"?"de-DE":"en-US"));
  const rev = n("review");
  return `<div class="strip">
    <span class="chip k"><b>${n("high")}</b>${esc(t("critical"))}</span>
    <span class="chip m"><b>${n("medium")}</b>${esc(t("medium"))}</span>
    <span class="chip e"><b>${n("dismissed")}</b>${esc(t("cleared"))}</span>
    ${rev?`<span class="chip p"><b>${rev}</b>${esc(t("review"))}</span>`:""}
    <span class="vol">${fmt(counts.postings)} ${esc(t("postings"))}, ${fmt(counts.entities)} ${esc(t("entities"))},
      ${fmt(counts.source_files)} ${esc(t("docs_examined"))}</span></div>`;
}
async function loadFindings(){
  const el = document.getElementById("view-f");
  const [d, cov] = await Promise.all([getJSON(api("/api/findings")), getJSON(api("/api/coverage"))]);
  if(d.status==="loading"){ setTimeout(loadFindings,1500);
    el.innerHTML = `<div class="note">${esc(t("loading"))} (${esc(d.dossier_path)}) ...</div>`; return; }
  if(d.status==="error"){ el.innerHTML = `<div class="note err">${esc(t("err"))}: ${esc(d.error)}</div>`; return; }
  document.getElementById("dossiername").textContent = d.dossier;
  if(RUN !== "default" && cov.status==="ready" && cov.auditor)
    document.getElementById("status").textContent =
      t("run_label") + " " + RUN + " \u00b7 " + t("auditor") + ": " + cov.auditor;
  const counts = cov.status==="ready" ? cov.counts : {};
  const kindKey = {entity:"kind_entity", transaction:"kind_transaction", lens:"kind_lens"};
  const mode = d.mode==="scored" ? "" : `<div class="note">${esc(t("mode_note"))}</div>`;
  el.innerHTML = stripHtml(d.findings, counts) + mode + (d.findings.length? d.findings.map(f=>{
    const top = strongest(f.flags);
    const tier = f.tier?`<span class="badge tier-${esc(f.tier)}">${esc(tm("tier", f.tier))}</span>`:"";
    const name = f.subject_name || f.subject_id;
    const idpart = f.subject_name?`<span class="mut">${esc(f.subject_id)} \u00b7 </span>`:"";
    const fams = `${f.families.length} ${esc(f.families.length===1?t("method_one"):t("methods"))}`;
    const nf = `${f.flag_count} ${esc(f.flag_count===1?t("flag_one"):t("flags"))}`;
    const score = f.score!=null?`<div class="score">${f.score.toFixed(2).replace(".", LANG==="de"?",":".")}<span class="mut">${esc(t("score"))}</span></div>`:"";
    return `<div class="card"><div class="row" onclick="tog(this)">
      <div style="flex:1;min-width:0">
        <div class="headline">${tier} <b>${esc(top?top.title:"")}</b></div>
        <div class="subline"><b>${esc(name)}</b> \u00b7 ${idpart}${esc(t(kindKey[f.subject_kind]||"")||f.subject_kind)}
          \u00b7 ${fams} \u00b7 ${nf}
          ${f.max_amount?`\u00b7 <span class="amt">${esc(f.max_amount)} EUR</span>`:""}</div>
      </div>${score}</div>
      <div class="body">${f.defense_note?`<div class="note">${esc(t("defense"))}: ${esc(f.defense_note)}</div>`:""}
      ${f.flags.map(flagHtml).join("")}</div></div>`;
  }).join("") : `<div class="note">${esc(t("no_findings"))}</div>`);
}
async function trace(){
  const q = document.getElementById("traceq").value.trim();
  const out = document.getElementById("traceout");
  if(!q){ out.innerHTML=""; return; }
  out.innerHTML = `<div class="note">${esc(t("trace_searching"))}</div>`;
  const d = await getJSON(api("/api/trace?q="+encodeURIComponent(q)));
  if(d.status!=="ready"){ out.innerHTML =
    `<div class="note err">${esc(d.error||t("loading"))}</div>`; return; }
  let h = d.amount?`<div class="note">${esc(t("trace_amount"))}: <span class="amt">${esc(d.amount)} EUR</span></div>`:"";
  let any=false;
  for(const s of d.sections){
    if(!s.total) continue; any=true;
    h += `<h2 class="sec">${esc(tm("sec", s.id))} <span class="mut">(${s.total} ${esc(t("trace_hits"))}`+
      `${s.total>s.hits.length?", "+esc(t("trace_first"))+" "+s.hits.length+" "+esc(t("trace_shown")):""})</span></h2>`;
    h += s.hits.map(x=>`<div class="card"><div class="row" style="cursor:default">
      ${x.match.map(m=>`<span class="badge m">${esc(tm("match", m))}</span>`).join(" ")}
      ${x.label?`<span>${esc(x.label)}</span>`:""}
      <span class="cite">${esc(x.cite)}${x.sheet?" ["+esc(x.sheet)+"]":""}</span></div>
      <div style="padding:0 14px 10px"><pre class="x">${esc(x.excerpt)||esc(t("no_excerpt"))}</pre></div></div>`).join("");
  }
  out.innerHTML = h + (any?"":`<div class="note">${esc(t("trace_none"))} "${esc(d.query)}".</div>`);
}
async function loadCoverage(){
  const el = document.getElementById("view-c");
  const d = await getJSON(api("/api/coverage"));
  if(d.status==="loading"){ setTimeout(loadCoverage,1500); return; }
  if(d.status==="error"){ el.innerHTML = `<div class="note err">${esc(t("err"))}: ${esc(d.error)}</div>`; return; }
  const c = d.counts;
  let h = `<div class="card"><div class="row" style="cursor:default">
    <b>${esc(d.dossier)}</b><span class="mut">${esc(d.dossier_path)}</span>
    <span class="badge">${c.postings} ${esc(t("c_postings"))}</span>
    <span class="badge">${c.entities} ${esc(t("c_entities"))}</span>
    <span class="badge">${c.documents} ${esc(t("c_documents"))}</span>
    <span class="badge">${c.flags} ${esc(t("c_flags"))}</span>
    <span class="badge">${c.findings} ${esc(t("c_findings"))}</span></div></div>`;
  h += `<h2 class="sec">${esc(t("lenses"))} (${d.lenses.length} ${esc(t("loaded"))}, ${d.lenses_failed.length} ${esc(t("failed"))})</h2>
    <div class="card"><table><tr><th>${esc(t("th_lens"))}</th><th>${esc(t("th_family"))}</th><th>${esc(t("th_flags"))}</th></tr>`+
    d.lenses.map(l=>`<tr><td>${esc(l.lens_id)}</td><td>${esc(tm("fam", l.family))}</td>
      <td class="num">${l.flags}</td></tr>`).join("")+`</table></div>`;
  if(d.lenses_failed.length)
    h += `<div class="card"><table><tr><th>${esc(t("th_module"))}</th><th>${esc(t("th_error"))}</th></tr>`+
      d.lenses_failed.map(l=>`<tr><td>${esc(l.module)}</td><td class="err">${esc(l.reason)}</td></tr>`).join("")+
      `</table></div>`;
  h += `<h2 class="sec">${esc(t("cov_docs"))}</h2><div class="card"><table>`+
    Object.entries(d.documents_per_kind).map(([k,v])=>
      `<tr><td>${esc(k)}</td><td class="num">${v}</td></tr>`).join("")+`</table></div>`;
  h += `<h2 class="sec">${esc(t("cov_ledger"))}</h2><div class="card"><table>`+
    Object.entries(d.postings_per_ledger).map(([k,v])=>
      `<tr><td>${esc(k)}</td><td class="num">${v}</td></tr>`).join("")+`</table></div>`;
  h += `<h2 class="sec">${esc(t("cov_unparsed"))} (${d.unparsed.length})</h2>`;
  h += d.unparsed.length?`<div class="card"><table><tr><th>${esc(t("th_file"))}</th><th>${esc(t("th_reason"))}</th></tr>`+
    d.unparsed.map(u=>`<tr><td>${esc(u.file)}</td><td class="err">${esc(u.reason)}</td></tr>`).join("")+
    `</table></div>` : `<div class="note">${esc(t("cov_all"))}</div>`;
  el.innerHTML = h;
}
const dz = document.getElementById("dropzone");
const fp = document.getElementById("filepick");
dz.onclick = () => fp.click();
fp.onchange = () => uploadFiles(fp.files);
dz.ondragover = e => { e.preventDefault(); dz.classList.add("over"); };
dz.ondragleave = () => dz.classList.remove("over");
dz.ondrop = e => { e.preventDefault(); dz.classList.remove("over"); uploadFiles(e.dataTransfer.files); };
async function uploadFiles(files){
  if(!files || !files.length) return;
  dz.classList.add("busy"); dz.textContent = t("drop_busy");
  const fd = new FormData();
  for(const f of files) fd.append("files", f);
  const aud = document.getElementById("auditor").value.trim();
  if(aud) fd.append("auditor", aud);
  try{
    const r = await fetch("/upload", {method:"POST", body:fd});
    const d = await r.json();
    if(d.status==="ok"){ location.search = "?run=" + encodeURIComponent(d.run); return; }
    dz.textContent = t("err") + ": " + (d.error||t("drop_fail"));
  }catch(err){ dz.textContent = t("err") + ": " + err; }
  dz.classList.remove("busy");
  setTimeout(()=>{ dz.textContent = t("drop_idle"); }, 4000);
}
applyChrome();
loadFindings(); loadCoverage();
</script>
<footer><a href="https://github.com/ciprimarian/laundromat" rel="noopener">github.com/ciprimarian/laundromat</a></footer>
</body>
</html>"""

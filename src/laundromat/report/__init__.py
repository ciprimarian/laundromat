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
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from decimal import Decimal, InvalidOperation
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from ..contracts import Dossier, Finding, Flag, SourceRef

DOSSIER_PATH = (
    os.environ.get("CORTEA_DOSSIER")
    or os.environ.get("DOSSIER_DIR")
    or "data/practice"
)

CENT = Decimal("0.01")

STATE: dict[str, Any] = {
    "ready": False,
    "error": None,
    "dossier": None,
    "flags": [],
    "findings": [],
    "index": None,
}
_LOCK = threading.Lock()


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


def _load() -> None:
    try:
        from ..pipeline import run

        dossier, flags, findings = run(DOSSIER_PATH)
        index = TraceIndex(dossier)
        with _LOCK:
            STATE.update(dossier=dossier, flags=flags, findings=findings, index=index)
    except Exception as e:  # never let the server die on a bad dossier
        with _LOCK:
            STATE["error"] = f"{type(e).__name__}: {e}"
    finally:
        with _LOCK:
            STATE["ready"] = True


@asynccontextmanager
async def _lifespan(_: FastAPI):
    threading.Thread(target=_load, daemon=True).start()
    yield


app = FastAPI(title="Laundromat Bericht", lifespan=_lifespan)


def _not_ready() -> JSONResponse | None:
    with _LOCK:
        if not STATE["ready"]:
            return JSONResponse({"status": "loading", "dossier_path": DOSSIER_PATH})
        if STATE["error"]:
            return JSONResponse({"status": "error", "error": STATE["error"]})
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
def api_findings():
    if (r := _not_ready()) is not None:
        return r
    dossier: Dossier = STATE["dossier"]
    findings: list[Finding] = STATE["findings"]
    flags: list[Flag] = STATE["flags"]
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
def api_trace(q: str = ""):
    if (r := _not_ready()) is not None:
        return r
    q = q.strip()
    if not q:
        return {"status": "ready", "query": q, "sections": [], "error": "Leere Anfrage"}

    dossier: Dossier = STATE["dossier"]
    index: TraceIndex = STATE["index"]
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
def api_coverage():
    if (r := _not_ready()) is not None:
        return r
    dossier: Dossier = STATE["dossier"]
    flags: list[Flag] = STATE["flags"]

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
        "dossier_path": DOSSIER_PATH,
        "counts": {
            "postings": len(dossier.postings),
            "entities": len(dossier.entities),
            "documents": len(dossier.documents),
            "flags": len(flags),
            "findings": len(STATE["findings"]),
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


@app.get("/", response_class=HTMLResponse)
def index_page():
    return HTMLResponse(PAGE)


# --------------------------------------------------------------------------
# The page. Self-contained: inline CSS + JS, no external requests.
# --------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Laundromat Pruefbericht</title>
<style>
:root { --bg:#f6f7f9; --card:#ffffff; --ink:#1a232e; --mut:#5c6b7a; --line:#dde3ea;
        --acc:#0b5cad; --hi:#b42318; --med:#b54708; --rev:#5c6b7a; --ok:#067647; }
* { box-sizing:border-box; margin:0; padding:0; }
body { background:var(--bg); color:var(--ink);
       font:15px/1.45 -apple-system,"Segoe UI",Roboto,Arial,sans-serif; }
header { background:var(--card); border-bottom:1px solid var(--line);
         padding:14px 24px; display:flex; align-items:baseline; gap:18px; }
header h1 { font-size:18px; }
header .sub { color:var(--mut); font-size:13px; }
nav { display:flex; gap:4px; padding:10px 24px 0; }
nav button { border:1px solid var(--line); border-bottom:none; background:#eef1f5;
             padding:8px 18px; font-size:14px; cursor:pointer; border-radius:6px 6px 0 0; }
nav button.on { background:var(--card); font-weight:600; color:var(--acc); }
main { padding:16px 24px 60px; max-width:1200px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:6px;
        margin-bottom:10px; }
.row { padding:10px 14px; cursor:pointer; display:flex; gap:12px; align-items:baseline;
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
</style>
</head>
<body>
<header>
  <h1>Laundromat Pruefbericht</h1>
  <span class="sub" id="dossiername"></span>
  <span class="sub" id="status"></span>
</header>
<nav>
  <button id="tab-f" class="on" onclick="show('f')">Feststellungen</button>
  <button id="tab-t" onclick="show('t')">Zahlenspur</button>
  <button id="tab-c" onclick="show('c')">Abdeckung</button>
</nav>
<main>
  <section id="view-f"><div class="note">Wird geladen ...</div></section>
  <section id="view-t" style="display:none">
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      <input id="traceq" placeholder="Betrag (z.B. 19.729.014,76), Kontonummer, Beleg-Nr. oder Name"
             onkeydown="if(event.key==='Enter')trace()">
      <button id="tracebtn" onclick="trace()">Spur verfolgen</button>
    </div>
    <div class="hint">Verfolgt jede Zahl vom Jahresabschluss ueber Saldenliste und Hauptbuch
      bis in Nebenbuecher und Belege. Betraege in deutschem oder englischem Format,
    exakt und &plusmn;1%.</div>
    <div id="traceout"></div>
  </section>
  <section id="view-c" style="display:none"><div class="note">Wird geladen ...</div></section>
</main>
<script>
"use strict";
function esc(s){return String(s==null?"":s).replace(/[&<>"']/g,
  c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
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
function evHtml(refs){
  return refs.map(e=>`<div class="ev"><span class="cite">${esc(e.cite)}`+
    `${e.sheet?" ["+esc(e.sheet)+"]":""}</span>`+
    `<pre class="x">${esc(e.excerpt)||"(kein Auszug)"}</pre></div>`).join("");
}
function flagHtml(fl,i){
  const conf = Math.round(fl.confidence*100);
  return `<div class="flag"><div class="fhead" onclick="tog(this)">
    <span class="badge m">${esc(fl.family)}</span> <b>${esc(fl.title)}</b>
    <span class="mut">${esc(fl.lens_id)} | Konfidenz ${conf}%`+
    `${fl.amount?" | <span class='amt'>"+esc(fl.amount)+" EUR</span>":""}</span></div>
    <div class="fbody"><div>${esc(fl.rationale)}</div>
    <div class="mut" style="margin-top:6px">Nachweise (${fl.evidence.length}):</div>
    ${evHtml(fl.evidence)}</div></div>`;
}
function tog(el){
  const b = el.parentElement.querySelector(".fbody, .body");
  if(b) b.style.display = b.style.display==="block"?"none":"block";
}
async function loadFindings(){
  const el = document.getElementById("view-f");
  const d = await getJSON("/api/findings");
  if(d.status==="loading"){ setTimeout(loadFindings,1500);
    el.innerHTML = `<div class="note">Dossier wird geladen (${esc(d.dossier_path)}) ...</div>`; return; }
  if(d.status==="error"){ el.innerHTML = `<div class="note err">Fehler: ${esc(d.error)}</div>`; return; }
  document.getElementById("dossiername").textContent = d.dossier;
  const mode = d.mode==="scored" ? "" :
    `<div class="note">Scoring noch nicht aktiv: Sortierung nach Zahl unabhaengiger`+
    ` Lens-Familien und Flag-Anzahl je Subjekt.</div>`;
  const kinds = {entity:"Entitaet", transaction:"Transaktion", lens:"Lens"};
  el.innerHTML = mode + (d.findings.length? d.findings.map(f=>{
    const name = f.subject_name?` <span class="mut">${esc(f.subject_name)}</span>`:"";
    const tier = f.tier?`<span class="badge tier-${esc(f.tier)}">${esc(f.tier)}</span>`:"";
    const score = f.score!=null?`<span class="mut">Score ${f.score.toFixed(2)}</span>`:"";
    return `<div class="card"><div class="row" onclick="tog(this)">
      ${tier}<b>${esc(f.subject_id)}</b>${name}
      <span class="badge">${esc(kinds[f.subject_kind]||f.subject_kind)}</span>
      ${f.families.map(x=>`<span class="badge m">${esc(x)}</span>`).join(" ")}
      <span class="mut">${f.flag_count} Flag${f.flag_count===1?"":"s"}</span>
      ${f.max_amount?`<span class="amt">${esc(f.max_amount)} EUR</span>`:""} ${score}</div>
      <div class="body">${f.defense_note?`<div class="note">Entlastung: ${esc(f.defense_note)}</div>`:""}
      ${f.flags.map(flagHtml).join("")}</div></div>`;
  }).join("") : `<div class="note">Keine Feststellungen.</div>`);
}
async function trace(){
  const q = document.getElementById("traceq").value.trim();
  const out = document.getElementById("traceout");
  if(!q){ out.innerHTML=""; return; }
  out.innerHTML = `<div class="note">Suche ...</div>`;
  const d = await getJSON("/api/trace?q="+encodeURIComponent(q));
  if(d.status!=="ready"){ out.innerHTML =
    `<div class="note err">${esc(d.error||"Dossier noch nicht geladen")}</div>`; return; }
  let h = d.amount?`<div class="note">Interpretiert als Betrag: <span class="amt">${esc(d.amount)} EUR</span></div>`:"";
  let any=false;
  for(const s of d.sections){
    if(!s.total) continue; any=true;
    h += `<h2 class="sec">${esc(s.label)} <span class="mut">(${s.total} Treffer`+
      `${s.total>s.hits.length?", erste "+s.hits.length+" gezeigt":""})</span></h2>`;
    h += s.hits.map(x=>`<div class="card"><div class="row" style="cursor:default">
      ${x.match.map(m=>`<span class="badge m">${esc(m)}</span>`).join(" ")}
      ${x.label?`<span>${esc(x.label)}</span>`:""}
      <span class="cite">${esc(x.cite)}${x.sheet?" ["+esc(x.sheet)+"]":""}</span></div>
      <div style="padding:0 14px 10px"><pre class="x">${esc(x.excerpt)||"(kein Auszug)"}</pre></div></div>`).join("");
  }
  out.innerHTML = h + (any?"":`<div class="note">Keine Treffer fuer "${esc(d.query)}".</div>`);
}
async function loadCoverage(){
  const el = document.getElementById("view-c");
  const d = await getJSON("/api/coverage");
  if(d.status==="loading"){ setTimeout(loadCoverage,1500); return; }
  if(d.status==="error"){ el.innerHTML = `<div class="note err">Fehler: ${esc(d.error)}</div>`; return; }
  const c = d.counts;
  let h = `<div class="card"><div class="row" style="cursor:default">
    <b>${esc(d.dossier)}</b><span class="mut">${esc(d.dossier_path)}</span>
    <span class="badge">${c.postings} Buchungen</span>
    <span class="badge">${c.entities} Entitaeten</span>
    <span class="badge">${c.documents} Dokumente</span>
    <span class="badge">${c.flags} Flags</span>
    <span class="badge">${c.findings} Feststellungen</span></div></div>`;
  h += `<h2 class="sec">Lenses (${d.lenses.length} geladen, ${d.lenses_failed.length} fehlgeschlagen)</h2>
    <div class="card"><table><tr><th>Lens</th><th>Familie</th><th>Flags</th></tr>`+
    d.lenses.map(l=>`<tr><td>${esc(l.lens_id)}</td><td>${esc(l.family)}</td>
      <td class="num">${l.flags}</td></tr>`).join("")+`</table></div>`;
  if(d.lenses_failed.length)
    h += `<div class="card"><table><tr><th>Modul</th><th>Fehler</th></tr>`+
      d.lenses_failed.map(l=>`<tr><td>${esc(l.module)}</td><td class="err">${esc(l.reason)}</td></tr>`).join("")+
      `</table></div>`;
  h += `<h2 class="sec">Dokumente je Art</h2><div class="card"><table>`+
    Object.entries(d.documents_per_kind).map(([k,v])=>
      `<tr><td>${esc(k)}</td><td class="num">${v}</td></tr>`).join("")+`</table></div>`;
  h += `<h2 class="sec">Buchungen je Buch</h2><div class="card"><table>`+
    Object.entries(d.postings_per_ledger).map(([k,v])=>
      `<tr><td>${esc(k)}</td><td class="num">${v}</td></tr>`).join("")+`</table></div>`;
  h += `<h2 class="sec">Nicht verarbeitete Dateien (${d.unparsed.length})</h2>`;
  h += d.unparsed.length?`<div class="card"><table><tr><th>Datei</th><th>Grund</th></tr>`+
    d.unparsed.map(u=>`<tr><td>${esc(u.file)}</td><td class="err">${esc(u.reason)}</td></tr>`).join("")+
    `</table></div>` : `<div class="note">Alle Dateien verarbeitet.</div>`;
  el.innerHTML = h;
}
loadFindings(); loadCoverage();
</script>
</body>
</html>"""

# laundromat

Find out if the books are clean.

**Live demo: https://laundromat.46-101-129-81.sslip.io** (drop a dossier zip on the page to audit it)

An audit agent for the Cortea track of the {Tech: Europe} x Almedia Summer Lock-In hackathon.
It ingests a company dossier (GDPdU export plus supporting documents), runs it through independent
detection lenses, scores findings by cross-lens corroboration, and serves an interactive UI where
every claim links to the exact file, line, and passage it rests on. No number without a source.

On the practice dossier it examines 26,647 postings, 543 entities and 26 documents, and reports
3 critical and 57 medium findings while showing 37 findings it checked and cleared. False alarms
are treated as failures: nothing is reported on the strength of a single detection method.

## How it works

```
dossier dir -> ingest -> canonical tables -> 7 lens families -> flags -> corroboration scoring
                                                                      -> findings + tiers
                                                                      -> defense pass (borderline tier)
                                                                      -> web UI with evidence drill-down
```

- **Ingest** reads the GDPdU index.xml driven tables plus csv/xlsx/pdf/docx supporting documents
  into canonical postings, entities and documents. Every row keeps a source reference
  (file, line or page, verbatim excerpt). Files are recognized by content and bilingual header
  sniffing, not by filename, so an unseen dossier with different naming still loads. Files that
  fail to parse are reported in a coverage panel instead of being dropped silently.
- **Lenses** are independent detectors: deterministic audit rules covering the full K1 to K7 audit
  plan (new vendor quick payment, missing goods receipt, capitalized repairs, cut-off, split
  payments under approval limits, round amounts, off-hours postings) plus backdating; statistical
  tests (Benford per partition, round number frequency, robust outliers, duplicate payments);
  an entity graph (shared addresses and VAT ids, segregation of duties violations, orphan users,
  related party links); temporal analysis (backdating distributions, per-user off-hours baselines,
  master data change timing, approval timing); cross-document reconciliation (sub-ledger to GL,
  three-way match, journal completeness, dormant bank-to-ledger matching); a semantic pass over
  posting texts; and an external vendor footprint check.
- **Scoring** aggregates flags per transaction and per entity. Confidence comes from corroboration
  across lens families that cannot fail the same way. Single-family findings never reach the top
  tier; borderline findings pass through a devil's advocate step that must either exonerate them
  with a grounded reason or promote them. Exonerated findings stay visible as checked and cleared.
- **UI** has three views: a ranked findings board with click-through to evidence, a figure tracer
  that resolves any amount, account or entity to its full provenance chain (including figures no
  lens flagged), and a coverage panel. Drag and drop a dossier zip to start a new audit run; each
  upload gets its own run id and an optional auditor name stamp.

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -e .[ai]
cp .env.example .env   # add your API keys
uvicorn laundromat.report:app --port 8000
```

Or with docker:

```
docker compose up -d --build
```

Point `DOSSIER_DIR` at the dossier directory you want preloaded, or just use the upload dropzone.

## Configuration

```
cp .env.example .env
```

and fill out the `.env` keys:

| Key | Used for |
|---|---|
| `OPENAI_API_KEY` | semantic lens, defense pass, account classification fallback |
| `TAVILY_API_KEY` | vendor web footprint lens |

Both are optional: without keys the AI lenses skip cleanly and the deterministic, statistical,
graph, temporal and reconciliation lenses still run.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): pipeline design, what each lens family catches,
  why corroboration across families controls false positives
- [docs/USAGE.md](docs/USAGE.md): auditing a new dossier, reading the results, docker deployment
- `tools/`: calibrate.py (per-lens fire rates), generalize_check.py (proves filename independence
  by renaming the whole dossier and re-running), false_positive_sweep.py (innocence heuristics on
  reported findings), report_md.py (renders a standalone audit memo), eval_groundtruth.py
  (precision and recall against a ground truth file)
- `tests/`: 90+ tests including pipeline contract invariants and hostile upload handling

## Tools and libraries

- OpenAI API: semantic analysis of posting texts, devil's advocate pass on borderline findings,
  account classification fallback
- Cognee: entity graph corroboration over vendors, customers, shareholders and users
- Tavily: web search for vendor existence checks
- pandas, openpyxl, pdfplumber, python-docx: ingestion
- FastAPI, uvicorn: web UI
- rapidfuzz: near-duplicate entity matching

## Audit thresholds

Materiality and journal entry testing floors are configured in `src/laundromat/contracts.py`
(defaults: materiality 400k, JET floor 25k, approval limit 10k) and can be overridden per dossier.

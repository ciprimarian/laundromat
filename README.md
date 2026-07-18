# laundromat

Find out if the books are clean.

An audit agent for the Cortea track of the {Tech: Europe} x Almedia Summer Lock-In hackathon.
It ingests a company dossier (GDPdU export plus supporting documents), runs it through independent
detection lenses, scores findings by cross-lens corroboration, and serves an interactive UI where
every claim links to the exact file, line, and passage it rests on. No number without a source.

## How it works

```
dossier dir -> ingest -> canonical tables -> 7 lenses -> flags -> corroboration scoring
                                                                -> findings + tiers
                                                                -> defense pass (borderline tier)
                                                                -> web UI with evidence drill-down
```

- **Ingest** reads the GDPdU index.xml driven tables plus csv/xlsx/pdf/docx supporting documents
  into canonical postings, entities and documents. Every row keeps a source reference
  (file, line or page, verbatim excerpt). Files that fail to parse are reported in a coverage
  panel instead of being dropped silently.
- **Lenses** are independent detectors: deterministic audit rules, statistical tests (Benford,
  round numbers, outliers), an entity graph (shared addresses, segregation of duties), temporal
  analysis (backdating, off-hours, master data timing), cross-document reconciliation,
  a semantic pass over posting texts, and an external vendor footprint check.
- **Scoring** aggregates flags per transaction and per entity. Confidence comes from corroboration
  across lens families that cannot fail the same way. Single-family findings never reach the top
  tier; borderline findings pass through a devil's advocate step that must either exonerate them
  with a grounded reason or promote them.
- **UI** has three views: a ranked findings board with click-through to evidence, a figure tracer
  that resolves any amount, account or entity to its full provenance chain, and a coverage panel.

The finding types include: payments without goods receipts, repairs capitalized as fixed assets,
cut-off violations, payment splitting under approval limits, round amount anomalies, unregistered
vendors, self-approval and other segregation of duties violations, backdated postings, sub-ledger
to GL mismatches, and three-way match breaks.

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -e .[ai]
cp .env.example .env   # add your API keys
uvicorn laundromat.report.app:app --port 8000
```

Or with docker:

```
docker compose up -d --build
```

Point `DOSSIER_DIR` at the dossier directory you want to audit.

## Configuration

`.env` keys:

| Key | Used for |
|---|---|
| `OPENAI_API_KEY` | semantic lens, defense pass, account classification fallback |
| `TAVILY_API_KEY` | vendor web footprint lens |

## Tools and libraries

- OpenAI API: semantic analysis of posting texts, devil's advocate pass on borderline findings,
  account classification fallback
- Cognee: entity graph over vendors, customers, shareholders and users
- Tavily: web search for vendor existence checks
- pandas, openpyxl, pdfplumber, python-docx: ingestion
- FastAPI, uvicorn: web UI
- rapidfuzz: near-duplicate entity matching

## Audit thresholds

Materiality and journal entry testing floors are configured in `src/laundromat/contracts.py`
(defaults: materiality 400k, JET floor 25k, approval limit 10k) and can be overridden per dossier.

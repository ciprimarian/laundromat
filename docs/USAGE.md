# Usage

## Requirements

- Python 3.11+
- Optional API keys for partner-backed lenses (OpenAI, Tavily). Core rule,
  statistical, temporal, graph (pure-python), and reconciliation lenses run
  without keys.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
# partner features:
pip install -e ".[ai]"
cp .env.example .env
```

Edit `.env`:

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | no | semantic lens, defense pass, account-class fallback |
| `TAVILY_API_KEY` | no | external vendor web check |
| `DOSSIER_DIR` or `CORTEA_DOSSIER` | no | dossier path (default `data/practice`) |
| `CORTEA_RUNS_DIR` | no | where zip uploads are extracted (default `data/runs`) |

## Point at a dossier

A dossier is a directory tree: GDPdU export (folders with `index.xml` + tables)
plus supporting csv/xlsx/pdf/docx. The practice set is linked at
`data/practice`.

```bash
export DOSSIER_DIR=data/practice
# or
export CORTEA_DOSSIER=/path/to/your/export
```

## Run the UI

```bash
uvicorn laundromat.report.app:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000`.

Tabs:

- **Feststellungen** (findings leaderboard): ranked subjects with tier, families,
  flags, and expandable evidence (file + line/page + excerpt).
- **Zahlenspur** (figure tracer): type an amount (German or English format),
  account, document number, or entity name. Returns hits across financial
  statements, trial balance, GL, sub-ledgers, and master data.
- **Abdeckung** (coverage): posting/document counts, per-lens flag totals,
  unparsed files, import failures.
- **Upload**: zip a dossier directory and drop it here for a new run (see below).

## Audit a new dossier via upload

1. Zip the dossier **directory** so the archive contains the export root (or a
   single top-level folder that is that root). Include GDPdU folders and
   supporting documents.
2. In the UI, open **Upload**, choose the `.zip`, click **Hochladen und pruefen**.
3. The server extracts under `data/runs/<run_id>/`, runs ingest + lenses +
   scoring, and replaces the in-memory view with the new run.
4. Use the leaderboard and figure tracer as usual. Coverage shows the new path
   and counts.

Programmatic upload:

```bash
curl -F "file=@/path/to/dossier.zip" http://127.0.0.1:8000/api/upload
```

Response includes `run_id`, `dossier`, and counts (`postings`, `flags`,
`findings`).

## Docker

```bash
# keys in .env
docker compose up -d --build
```

`compose.yaml` maps host port 80 to the app and mounts `./data` read-only.
Default `DOSSIER_DIR` is `/app/data/practice`. To audit another tree, put it
under `./data/...` and set `DOSSIER_DIR` accordingly, or use the upload tab
(upload writes under `data/runs` on the container filesystem unless you mount
that path writable).

## CLI helpers

```bash
# load + run all lenses + print flag counts
python -m laundromat.pipeline data/practice

# per-lens fire rates and multi-family overlap
python tools/calibrate.py data/practice

# rename every file/dir and assert lenses still fire (filename independence)
python tools/generalize_check.py data/practice
```

## Tests

```bash
pip install pytest httpx python-multipart
pytest -q
```

Report smoke tests use FastAPI `TestClient` (no browser). They cover the
leaderboard page, figure tracer, and zip upload.

## Thresholds

Materiality, JET floor, and approval limit live in
`src/laundromat/contracts.py` (`MATERIALITY`, `JET_FLOOR`, `APPROVAL_LIMIT`).
Change them there for a different engagement; do not hardcode account numbers
or fiscal years inside lenses.

## Troubleshooting

- **UI says loading forever**: check server logs; ingest may be large or a path
  wrong. `/api/coverage` and `/api/findings` return `{"status":"loading"}` until
  ready, or `error` with a message.
- **Partner lenses silent**: missing `OPENAI_API_KEY` / `TAVILY_API_KEY`. Other
  families still run.
- **Many unparsed files**: open Abdeckung. Often a delimiter/encoding issue or a
  sheet without a recognizable header. Ingest never invents columns from
  position alone for sub-ledgers.
- **Upload 400**: only `.zip` is accepted. Empty or corrupt archives are rejected.

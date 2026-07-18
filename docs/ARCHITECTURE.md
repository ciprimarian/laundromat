# Architecture

Laundromat is an auditor-facing pipeline that turns a company data export into
ranked findings. Every claim carries a source pointer (file, line or page,
excerpt). A finding without a source is rejected at construction time.

## Pipeline

```
dossier directory
    -> ingest (GDPdU + Begleitdokumente + pdf/docx)
    -> canonical Dossier (postings, entities, documents)
    -> lenses (independent detectors, one family each)
    -> flags (each with evidence SourceRefs)
    -> corroboration scoring (group by entity / transaction)
    -> findings + tiers (high / medium / review / dismissed)
    -> defense pass (LLM exoneration on REVIEW only)
    -> FastAPI UI (leaderboard, figure tracer, coverage, upload)
```

1. **Ingest** walks a dossier directory. `gdpdu.py` reads tables declared in
   `index.xml` (column names from the index; encoding and delimiters from DTD
   defaults unless overridden). `begleit.py` loads csv/xlsx supporting files by
   bilingual name patterns and header sniffing. `docs.py` loads pdf/docx. Rows
   that fail parse land in `dossier.unparsed` instead of crashing the run.
2. **Lenses** each implement `run(dossier) -> Iterable[Flag]` and register into
   a global registry on import. A lens that cannot find its inputs returns
   nothing. A lens exception is recorded and does not stop the others.
3. **Scoring** (`scoring.py`) groups flags by `entity_id` and by `doc_no`. The
   score is driven by the number of **distinct lens families** on the subject,
   not by the raw flag count. Confidence and amount only order within a tier.
   A single-family subject is capped at REVIEW no matter how loud the lens is.
4. **Defense** (`defense.py`) is an LLM devil's advocate for REVIEW findings.
   It must either promote or exonerate with a grounded note. It does not invent
   new findings.
5. **UI** (`report/`) serves the current run: ranked findings with evidence
   drill-down, a figure tracer for any amount/account/entity string, a coverage
   panel (unparsed files, per-lens flag counts), and a zip upload that starts a
   new run.

## Lens families

Corroboration counts families, not individual lenses. Two rule hits are one
opinion. A rule hit plus a graph hit are two independent opinions.

| Family | Role | Examples |
|---|---|---|
| `rule` | Deterministic audit criteria (K1-K7) | new vendor paid fast, no goods receipt, split payments, round amounts, odd-hours with Admin only as a co-factor |
| `statistical` | Population tests | Benford partitions, round-number rate vs ledger baseline, robust outliers, duplicate payments |
| `temporal` | Timing behaviour | booking-to-entry lag tail, per-user off-hours, master-data change before payment, velocity bursts, sequence gaps, approval timing |
| `graph` | Entity and access joins | self-approval, orphan users, rights matrix violations, shared address/VAT, near-duplicate vendor names, shareholder links |
| `reconciliation` | Arithmetic document ties | three-way match, sales vs goods issue, credit limits, trial balance vs GL, journal line counts vs Freigabe-Log, dormant bank match |
| `semantic` | Text vs account class | repair language on asset accounts, bilingual keyword + optional LLM judge |
| `external` | Outside the dossier | vendor web footprint via Tavily |

## Why multi-family corroboration controls false positives

Each family fails differently:

- A statistical false positive is an unusual but honest tail of a distribution.
- A rule false positive is usually a missing document type or a service invoice
  with no goods receipt by nature.
- A graph false positive is a coincidental name or address collision.
- A temporal false positive is a busy month-end or a legitimate back-posting.

When two or more of those fire on the same vendor or payment, the shared
explanation "normal operations" gets thin. Scoring therefore:

- rewards **breadth across families** first,
- never promotes a single-family subject above REVIEW,
- uses confidence only as a secondary weight.

That is the false-positive shield. Breadth of *detection* is fine; breadth of
*reporting* is not.

## Partner technology

The track requires at least one partner. We use three, each in a narrow place:

| Partner | Where | What |
|---|---|---|
| **OpenAI** | `lenses/semantic.py`, `defense.py`, `ingest/accounts.py` fallback | text/account mismatch judging, REVIEW exoneration, leftover account classification |
| **Cognee** | `lenses/graph_cognee.py` | optional knowledge-graph pass over vendors, customers, shareholders, users (pure-python graph lenses also run without it) |
| **Tavily** | `lenses/external.py` | web footprint for vendors (existence / shell-company signals) |

Statistical, temporal, rule, and reconciliation lenses are pure local code. They
run without API keys. Missing keys degrade only the partner-backed paths.

## Frozen contracts

`src/laundromat/contracts.py` defines `Posting`, `Entity`, `Document`,
`Dossier`, `Flag`, `Finding`, `LensFamily`, thresholds (`MATERIALITY`,
`JET_FLOOR`, `APPROVAL_LIMIT`), and the lens registry. Lenses import from it
and do not edit it. `Flag.evidence` must be a non-empty tuple of `SourceRef`.

## Calibration and generalization

- `tools/calibrate.py` prints per-lens flag counts and multi-family subject
  overlap on a dossier (default `data/practice`). Target: each lens under about
  1-2% of rows.
- `tools/generalize_check.py` copies a dossier, renames every directory and
  file, rewrites GDPdU `index.xml` URLs, and re-runs the pipeline. Any lens
  that drops from N>0 to 0 on the renamed tree is treated as filename-dependent
  and fails the check.

## Project layout (relevant parts)

```
src/laundromat/
  contracts.py          frozen types
  ingest/               gdpdu, begleit, docs, accounts
  lenses/               one module per concern (rules, stats, temporal, ...)
  scoring.py            corroboration
  defense.py            REVIEW exoneration
  pipeline.py           load + run lenses + score
  report/               FastAPI UI
tools/
  calibrate.py
  generalize_check.py
tests/
docs/
```

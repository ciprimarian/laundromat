"""Lane A gate as a regression test. Needs data/practice to exist."""

from collections import Counter
from decimal import Decimal
from pathlib import Path

import pytest

from laundromat.contracts import EntityType
from laundromat.ingest import load_dossier

ROOT = Path(__file__).resolve().parents[1] / "data" / "practice"

pytestmark = pytest.mark.skipif(not ROOT.is_dir(), reason="practice dossier not present")


@pytest.fixture(scope="module")
def dossier():
    return load_dossier(ROOT)


def test_posting_counts(dossier):
    by_ledger = Counter(p.attrs.get("ledger") for p in dossier.postings)
    assert by_ledger["GL"] == 20258
    assert by_ledger["AP"] == 2584
    assert by_ledger["AR"] == 3749
    assert by_ledger["FA"] == 56


def test_entity_counts(dossier):
    by_type = Counter(e.type for e in dossier.entities.values())
    assert by_type[EntityType.VENDOR] == 143
    assert by_type[EntityType.CUSTOMER] == 160
    assert by_type[EntityType.ASSET] == 197
    assert by_type[EntityType.ACCOUNT] == 43


def test_document_counts(dossier):
    kinds = Counter(d.kind for d in dossier.documents)
    assert kinds["goods_receipt"] == 858
    assert kinds["approval"] == 91
    assert kinds["master_change"] == 19
    assert kinds["sales_invoice"] == 2041
    assert kinds["purchase_invoice"] == 8


def test_nothing_unparsed(dossier):
    assert dossier.unparsed == []


def test_every_posting_cites_source(dossier):
    for p in dossier.postings:
        assert p.source.file
        assert p.source.line
        assert p.source.excerpt


def test_reconciles_to_trial_balance(dossier):
    """GL journal with sub-account rollup ties per account to Saldenliste."""
    soll: dict[str, Decimal] = {}
    haben: dict[str, Decimal] = {}
    for p in dossier.postings:
        if p.attrs.get("ledger") != "GL":
            continue
        acct = p.attrs.get("account_base", p.account)
        if p.amount >= 0:
            soll[acct] = soll.get(acct, Decimal(0)) + p.amount
        else:
            haben[acct] = haben.get(acct, Decimal(0)) - p.amount

    rows = [d for d in dossier.documents if d.kind == "trial_balance"]
    assert len(rows) == 43
    checked = 0
    for row in rows:
        f = row.fields
        acct = f.get("Konto") or row.ref
        want_soll = f.get("Soll 2025")
        want_haben = f.get("Haben 2025")
        if not acct or want_soll is None or want_haben is None:
            continue
        assert Decimal(want_soll.replace(",", ".")) == soll.get(acct, Decimal(0)), acct
        assert Decimal(want_haben.replace(",", ".")) == haben.get(acct, Decimal(0)), acct
        checked += 1
    assert checked == 43

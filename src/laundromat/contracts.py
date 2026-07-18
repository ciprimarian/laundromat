"""Frozen contracts. Every lane codes against this file and nothing else.

Do not edit without telling the other lanes -- lenses are written in parallel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Callable, Iterable, Protocol

# --------------------------------------------------------------------------
# Evidence: the "no number without a source" guarantee, enforced structurally.
# A Flag cannot be constructed without at least one SourceRef.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceRef:
    """Points at the exact place a claim came from."""

    file: str  # relative path within the dossier
    line: int | None = None  # 1-based row for csv/txt/xlsx
    page: int | None = None  # 1-based page for pdf/docx
    sheet: str | None = None  # xlsx sheet name
    excerpt: str = ""  # verbatim snippet shown in the UI

    def cite(self) -> str:
        loc = f":{self.line}" if self.line else (f" p.{self.page}" if self.page else "")
        return f"{self.file}{loc}"


# --------------------------------------------------------------------------
# Canonical tables. Ingest (lane A) produces these; every lens consumes them.
# --------------------------------------------------------------------------


class EntityType(str, Enum):
    VENDOR = "vendor"
    CUSTOMER = "customer"
    ASSET = "asset"
    EMPLOYEE = "employee"
    ACCOUNT = "account"


@dataclass(frozen=True)
class Entity:
    id: str
    type: EntityType
    name: str
    source: SourceRef
    address: str | None = None
    iban: str | None = None
    created_at: date | None = None
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Posting:
    doc_no: str
    booking_date: date
    amount: Decimal  # signed, in document currency
    account: str
    source: SourceRef
    posted_at: datetime | None = None  # entry timestamp -- drives the K7 lens
    counter_account: str | None = None
    entity_id: str | None = None  # vendor/customer this posting belongs to
    user: str | None = None  # who entered it
    text: str = ""  # Buchungstext -- the semantic lens reads this
    currency: str = "EUR"
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Document:
    """Non-GL supporting records: goods receipts, approvals, OP lists, etc."""

    kind: str  # "goods_receipt" | "approval" | "master_change" | "op_item" | ...
    ref: str  # the document's own identifier
    source: SourceRef
    entity_id: str | None = None
    doc_date: date | None = None
    amount: Decimal | None = None
    fields: dict[str, str] = field(default_factory=dict)


@dataclass
class Dossier:
    """Everything ingested from one company's data export."""

    name: str
    postings: list[Posting] = field(default_factory=list)
    entities: dict[str, Entity] = field(default_factory=dict)
    documents: list[Document] = field(default_factory=list)
    unparsed: list[tuple[str, str]] = field(default_factory=list)  # (file, reason)

    def docs_of(self, kind: str) -> list[Document]:
        return [d for d in self.documents if d.kind == kind]


# --------------------------------------------------------------------------
# Flags: what a lens emits.
# --------------------------------------------------------------------------


class LensFamily(str, Enum):
    """Corroboration counts DISTINCT families, not distinct lenses.

    Two rule-based hits are one opinion; a rule hit + a graph hit are two.
    """

    RULE = "rule"
    STATISTICAL = "statistical"
    GRAPH = "graph"
    TEMPORAL = "temporal"
    RECONCILIATION = "reconciliation"
    SEMANTIC = "semantic"
    EXTERNAL = "external"


@dataclass(frozen=True)
class Flag:
    lens_id: str  # e.g. "K2_no_goods_receipt"
    family: LensFamily
    title: str  # one line, auditor-readable
    rationale: str  # why this is suspicious, in plain German/English
    evidence: tuple[SourceRef, ...]  # NEVER empty -- validated below
    entity_id: str | None = None
    doc_no: str | None = None
    amount: Decimal | None = None
    confidence: float = 0.5  # lens-local self-assessment, 0..1

    def __post_init__(self) -> None:
        if not self.evidence:
            raise ValueError(f"{self.lens_id}: a flag without evidence is not a finding")


class Tier(str, Enum):
    HIGH = "high"  # report as fraud
    MEDIUM = "medium"  # report with caveat
    REVIEW = "review"  # goes to the defense pass before it may be reported
    DISMISSED = "dismissed"  # defense pass exonerated it


@dataclass
class Finding:
    """Post-corroboration: one subject, all the flags raised against it."""

    subject_id: str  # entity id, or doc_no for transaction-level findings
    subject_kind: str  # "entity" | "transaction"
    flags: list[Flag]
    tier: Tier = Tier.REVIEW
    score: float = 0.0
    defense_note: str | None = None

    @property
    def families(self) -> set[LensFamily]:
        return {f.family for f in self.flags}

    @property
    def max_amount(self) -> Decimal:
        return max((f.amount or Decimal(0) for f in self.flags), default=Decimal(0))


# --------------------------------------------------------------------------
# The Lens interface + registry. One module per lens, zero shared state.
# --------------------------------------------------------------------------


class Lens(Protocol):
    lens_id: str
    family: LensFamily

    def run(self, dossier: Dossier) -> Iterable[Flag]: ...


REGISTRY: dict[str, Lens] = {}


def register(lens: Lens) -> Lens:
    """Decorator. Import the module and the lens joins the pipeline."""
    if lens.lens_id in REGISTRY:
        raise ValueError(f"duplicate lens_id {lens.lens_id}")
    REGISTRY[lens.lens_id] = lens
    return lens


# Audit thresholds from Pruefungsplanung_JET_2025.docx.
# Kept here so the judging dossier can override them in one place.
MATERIALITY = Decimal("400000")
JET_FLOOR = Decimal("25000")
APPROVAL_LIMIT = Decimal("10000")

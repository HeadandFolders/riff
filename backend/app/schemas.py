"""Core data model.

Three tiers of graph: a stone is one chunk of understanding, a castle is a
paper plus the branch towers built while reading it, a kingdom is a cluster of
castles. Sessions and hypotheses are process state and live alongside.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ConceptState(str, Enum):
    KNOWN = "known"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class SessionState(str, Enum):
    QUEUED = "queued"
    READING = "reading"
    BRANCHING = "branching"
    SUSPENDED = "suspended"
    EXAMINING = "examining"
    COMMITTED = "committed"


class HypothesisStatus(str, Enum):
    DRAFT = "draft"
    OPEN = "open"
    TRIAGED = "triaged"
    RUNNING = "running"
    RESOLVED = "resolved"
    ABANDONED = "abandoned"


class PrepStatus(str, Enum):
    PENDING = "pending"
    PREPARING = "preparing"
    READY = "ready"
    FAILED = "failed"


class ExplanationStrategy(str, Enum):
    PRIMER = "primer"
    ANALOGY = "analogy"
    CODE_GROUNDING = "code_grounding"
    WORKED_EXAMPLE = "worked_example"


class Section(BaseModel):
    id: str
    number: str
    title: str
    text: str
    figure_refs: list[str] = Field(default_factory=list)
    concept_labels: list[str] = Field(default_factory=list)


class Paper(BaseModel):
    id: str
    title: str
    source: Literal["arxiv", "pdf"]
    source_ref: str
    authors: list[str] = Field(default_factory=list)
    published: str | None = None
    repo_url: str | None = None
    future_work_section_id: str | None = None
    sections: list[Section] = Field(default_factory=list)
    prep_status: PrepStatus = PrepStatus.PENDING
    prep_error: str | None = None
    kingdom_id: str | None = None
    created_at: datetime = Field(default_factory=_now)


class Stone(BaseModel):
    """One chunk of understanding: a section you read or a primer that filled a gap."""

    id: str
    paper_id: str
    section_id: str | None = None
    kind: Literal["section", "primer"]
    title: str
    text: str
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)


class Concept(BaseModel):
    id: str
    label: str
    state: ConceptState
    # Stone ids that justify the current state, newest last.
    evidence: list[str] = Field(default_factory=list)
    first_seen_paper_id: str | None = None
    resolved_by_stone_id: str | None = None
    updated_at: datetime = Field(default_factory=_now)


class Edge(BaseModel):
    id: str
    from_id: str
    to_id: str
    kind: Literal["prerequisite", "similarity", "branch", "member"]
    weight: float = 1.0


class Kingdom(BaseModel):
    id: str
    label: str
    summary: str | None = None
    paper_ids: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=_now)


class ConceptVerdict(BaseModel):
    label: str
    state: ConceptState
    similarity: float
    nearest_stone_id: str | None = None


class SectionPlan(BaseModel):
    """Cartographer output. Rendered as the 'before you read' scaffold."""

    section_id: str
    assumes: list[ConceptVerdict] = Field(default_factory=list)
    already_held: list[str] = Field(default_factory=list)
    new_here: list[str] = Field(default_factory=list)

    @property
    def gap_count(self) -> int:
        return sum(1 for c in self.assumes if c.state is ConceptState.UNKNOWN)


class ReadingEvent(BaseModel):
    at: datetime = Field(default_factory=_now)
    kind: Literal[
        "started",
        "section_opened",
        "lost",
        "already_knew",
        "branch_opened",
        "branch_resolved",
        "check_passed",
        "check_failed",
        "suspended",
        "resumed",
        "committed",
    ]
    section_id: str | None = None
    concept_label: str | None = None
    strategy: ExplanationStrategy | None = None
    note: str | None = None


class Session(BaseModel):
    id: str
    paper_id: str
    state: SessionState = SessionState.QUEUED
    cursor_section_id: str | None = None
    branching_concept: str | None = None
    events: list[ReadingEvent] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=_now)
    last_touched_at: datetime = Field(default_factory=_now)

    @property
    def is_resumable(self) -> bool:
        return self.state is SessionState.SUSPENDED and self.cursor_section_id is not None


class Hypothesis(BaseModel):
    id: str
    claim: str
    confidence: int = Field(ge=1, le=5)
    falsifier: str
    status: HypothesisStatus = HypothesisStatus.DRAFT
    paper_id: str
    section_id: str | None = None
    lineage_stone_ids: list[str] = Field(default_factory=list)
    # Papers Triage found that already test this. Empty is the interesting case.
    prior_art: list[str] = Field(default_factory=list)
    committed_at: datetime = Field(default_factory=_now)
    resolved_at: datetime | None = None
    resolution: Literal["held", "failed", "superseded"] | None = None


class ExperimentSpec(BaseModel):
    objective: str
    baseline_repo: str | None = None
    baseline_commit: str | None = None
    deltas: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    acceptance: str | None = None
    out_of_scope: list[str] = Field(default_factory=list)
    compute_estimate: str | None = None


class Proposal(BaseModel):
    """The handoff artifact: markdown any agent can read, frontmatter any agent can parse."""

    id: str
    hypothesis_id: str
    title: str
    understanding: str
    builds_on: str
    the_gap: str
    how_to_test: str
    unresolved: list[str] = Field(default_factory=list)
    spec: ExperimentSpec
    created_at: datetime = Field(default_factory=_now)


class StrategyStat(BaseModel):
    """Counts backing the Explainer's strategy choice, keyed by gap type."""

    id: str
    gap_kind: str
    strategy: ExplanationStrategy
    offered: int = 0
    resolved: int = 0

    @property
    def resolution_rate(self) -> float:
        return self.resolved / self.offered if self.offered else 0.0


class QueueEntry(BaseModel):
    paper_id: str
    title: str
    gap_count: int
    estimated_minutes: int
    prep_status: PrepStatus
    blocking_concepts: list[str] = Field(default_factory=list)

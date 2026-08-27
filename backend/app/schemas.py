"""Core data model.

Three tiers of graph: a stone is one chunk of understanding, a castle is a
paper plus the branch towers built while reading it, a kingdom is a cluster of
castles. Sessions, assessments and hypotheses are process state alongside.

Two things are deliberately kept apart. Whether the graph *contains* material
for a concept is a retrieval question, answered cheaply by vector search.
Whether the reader *understands* it is a reasoning question, answered only by
Gemini grading something the reader said. The first is a prefilter. The second
is the verdict.
"""

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ConceptPresence(str, Enum):
    """Retrieval-derived: does the graph hold material for this concept?"""

    IN_GRAPH = "in_graph"
    ABSENT = "absent"


class Understanding(str, Enum):
    """Reasoning-derived: what did Gemini conclude from what the reader said?"""

    UNASSESSED = "unassessed"
    SOLID = "solid"
    PARTIAL = "partial"
    MISCONCEIVED = "misconceived"
    ABSENT = "absent"


#: Understanding levels that make a concept block a paper in the queue.
BLOCKING_UNDERSTANDING = frozenset(
    {Understanding.ABSENT, Understanding.MISCONCEIVED, Understanding.PARTIAL}
)

RETEST_INTERVALS: dict[Understanding, timedelta] = {
    Understanding.MISCONCEIVED: timedelta(days=3),
    Understanding.PARTIAL: timedelta(days=10),
    Understanding.SOLID: timedelta(days=45),
}


class SessionState(str, Enum):
    QUEUED = "queued"
    READING = "reading"
    BRANCHING = "branching"
    ASSESSING = "assessing"
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


class SectionRef(BaseModel):
    """Lightweight section descriptor carried on the paper document.

    Section prose lives in a subcollection so listing papers never pays for it.
    """

    id: str
    number: str
    title: str
    concept_labels: list[str] = Field(default_factory=list)


class SectionBody(BaseModel):
    """Section prose, stored at papers/{paper_id}/sections/{id}."""

    id: str
    paper_id: str
    number: str
    title: str
    text: str
    figure_refs: list[str] = Field(default_factory=list)


class Paper(BaseModel):
    id: str
    title: str
    source: Literal["arxiv", "pdf"]
    source_ref: str
    authors: list[str] = Field(default_factory=list)
    published: str | None = None
    repo_url: str | None = None
    future_work_section_id: str | None = None
    section_index: list[SectionRef] = Field(default_factory=list)

    prep_status: PrepStatus = PrepStatus.PENDING
    prep_error: str | None = None
    kingdom_id: str | None = None

    # Denormalised so the queue is a single ordered read rather than a fan-out.
    # Recomputed only for papers touching a concept whose state changed.
    gap_count: int = 0
    blocking_concepts: list[str] = Field(default_factory=list)
    estimated_minutes: int = 0
    frontier_stale: bool = True

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


class Kingdom(BaseModel):
    """A cluster of papers sharing embedding space, labelled by Gemini."""

    id: str
    label: str
    summary: str = ""
    paper_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)


class Edge(BaseModel):
    """A typed link between two graph nodes.

    ``member`` and ``appears_in`` links are derived at read time from the paper
    and reverse-index documents, so they are never stored as edges.
    """

    id: str
    from_id: str
    to_id: str
    kind: Literal["prerequisite", "similarity", "branch"]
    weight: float = 1.0
    created_at: datetime = Field(default_factory=_now)


class Misconception(BaseModel):
    """A specific wrong belief, in the reader's own terms, kept for retesting."""

    id: str
    concept_label: str
    #: Normalised form of ``concept_label``; queries filter on this.
    concept_key: str = ""
    belief: str
    correction: str
    severity: Literal["minor", "moderate", "blocking"] = "moderate"
    status: Literal["open", "addressed", "recurring"] = "open"
    times_observed: int = 1
    paper_id: str | None = None
    section_id: str | None = None
    session_id: str | None = None
    first_seen_at: datetime = Field(default_factory=_now)
    last_seen_at: datetime = Field(default_factory=_now)
    next_retest_at: datetime | None = None


class Concept(BaseModel):
    id: str
    label: str
    presence: ConceptPresence = ConceptPresence.ABSENT
    understanding: Understanding = Understanding.UNASSESSED
    # Prefilter score kept for diagnostics only; never the verdict.
    retrieval_similarity: float = 0.0
    evidence: list[str] = Field(default_factory=list)
    misconception_ids: list[str] = Field(default_factory=list)
    assessment_count: int = 0
    last_assessed_at: datetime | None = None
    next_retest_at: datetime | None = None
    first_seen_paper_id: str | None = None
    updated_at: datetime = Field(default_factory=_now)

    @property
    def is_gap(self) -> bool:
        if self.understanding is Understanding.UNASSESSED:
            return self.presence is ConceptPresence.ABSENT
        return self.understanding in BLOCKING_UNDERSTANDING


class DetectedMisconception(BaseModel):
    """One misconception Gemini found in the reader's explanation."""

    concept_label: str
    belief: str = Field(description="What the reader appears to believe, in their terms")
    correction: str = Field(description="What is actually the case, and why the belief fails")
    severity: Literal["minor", "moderate", "blocking"]


class UnderstandingVerdict(BaseModel):
    """Assessor output. The authoritative judgement on a concept."""

    concept_label: str
    level: Understanding
    reasoning: str = Field(description="Why this level, citing what the reader said")
    missing_pieces: list[str] = Field(default_factory=list)
    misconceptions: list[DetectedMisconception] = Field(default_factory=list)
    followup_question: str | None = Field(
        default=None,
        description="One question that would resolve the remaining doubt",
    )


class ConceptCandidate(BaseModel):
    """Prefilter result: what retrieval alone can say before anyone is asked anything."""

    label: str
    presence: ConceptPresence
    similarity: float
    nearest_stone_id: str | None = None
    understanding: Understanding = Understanding.UNASSESSED
    open_misconceptions: list[str] = Field(default_factory=list)

    @property
    def needs_assessment(self) -> bool:
        return (
            self.presence is ConceptPresence.IN_GRAPH
            and self.understanding is Understanding.UNASSESSED
        )


class SectionPlan(BaseModel):
    """Cartographer output. Rendered as the 'before you read' scaffold."""

    section_id: str
    assumes: list[ConceptCandidate] = Field(default_factory=list)
    already_held: list[str] = Field(default_factory=list)
    new_here: list[str] = Field(default_factory=list)
    revisit: list[str] = Field(
        default_factory=list,
        description="Concepts with open misconceptions that this section touches",
    )

    @property
    def gap_count(self) -> int:
        return sum(1 for c in self.assumes if c.presence is ConceptPresence.ABSENT)


class ReadingEvent(BaseModel):
    at: datetime = Field(default_factory=_now)
    kind: Literal[
        "started",
        "section_opened",
        "lost",
        "already_knew",
        "branch_opened",
        "branch_resolved",
        "explained",
        "assessed",
        "misconception_found",
        "misconception_cleared",
        "check_passed",
        "check_failed",
        "suspended",
        "resumed",
        "committed",
    ]
    section_id: str | None = None
    concept_label: str | None = None
    strategy: ExplanationStrategy | None = None
    understanding: Understanding | None = None
    note: str | None = None


class Session(BaseModel):
    id: str
    paper_id: str
    state: SessionState = SessionState.QUEUED
    cursor_section_id: str | None = None
    branching_concept: str | None = None
    assessing_concept: str | None = None
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
    misconception_count: int = 0
    estimated_minutes: int
    prep_status: PrepStatus
    blocking_concepts: list[str] = Field(default_factory=list)


class QueuePage(BaseModel):
    entries: list[QueueEntry]
    next_cursor: str | None = None
    total_ready: int | None = None


class GraphNode(BaseModel):
    id: str
    kind: Literal["kingdom", "paper", "concept", "stone"]
    label: str
    kingdom_id: str | None = None
    understanding: Understanding | None = None
    misconception_count: int = 0
    degree: int = 0
    paper_id: str | None = None


class GraphEdge(BaseModel):
    source: str
    target: str
    kind: Literal["prerequisite", "similarity", "branch", "member", "appears_in"]
    weight: float = 1.0


class GraphView(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    truncated: bool = False

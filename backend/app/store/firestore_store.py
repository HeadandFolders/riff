"""Firestore-backed graph and process store.

Firestore holds both tiers of state: the knowledge graph (stones, concepts,
edges, kingdoms) using native vector KNN for similarity, and the process log
(sessions, hypotheses, strategy counts). One database, no external service.
"""

from __future__ import annotations

import uuid
from functools import lru_cache

from google.cloud import firestore
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector

from ..config import settings
from ..schemas import (
    Concept,
    ConceptState,
    ConceptVerdict,
    Edge,
    Hypothesis,
    Kingdom,
    Paper,
    PrepStatus,
    Proposal,
    QueueEntry,
    ReadingEvent,
    Session,
    Stone,
)

PAPERS = "papers"
STONES = "stones"
CONCEPTS = "concepts"
EDGES = "edges"
KINGDOMS = "kingdoms"
SESSIONS = "sessions"
HYPOTHESES = "hypotheses"
PROPOSALS = "proposals"
STRATEGIES = "strategies"

DISTANCE_FIELD = "_distance"

# Rough reading pace used for queue estimates; refined from real session
# durations once there are enough of them.
MINUTES_PER_SECTION = 6
MINUTES_PER_GAP = 4


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _concept_key(label: str) -> str:
    return label.strip().lower().replace(" ", "-")


class GraphStore:
    def __init__(self, client: firestore.Client | None = None) -> None:
        cfg = settings()
        self.client = client or firestore.Client(
            project=cfg.project_id, database=cfg.firestore_database
        )

    # ---------- papers ----------

    def save_paper(self, paper: Paper) -> None:
        self.client.collection(PAPERS).document(paper.id).set(
            paper.model_dump(mode="json")
        )

    def get_paper(self, paper_id: str) -> Paper | None:
        snap = self.client.collection(PAPERS).document(paper_id).get()
        return Paper(**snap.to_dict()) if snap.exists else None

    def set_prep_status(
        self, paper_id: str, status: PrepStatus, error: str | None = None
    ) -> None:
        self.client.collection(PAPERS).document(paper_id).update(
            {"prep_status": status.value, "prep_error": error}
        )

    def list_papers(self) -> list[Paper]:
        return [Paper(**d.to_dict()) for d in self.client.collection(PAPERS).stream()]

    # ---------- stones and vector search ----------

    def add_stone(self, stone: Stone, embedding: list[float]) -> None:
        payload = stone.model_dump(mode="json")
        payload["embedding"] = Vector(embedding)
        self.client.collection(STONES).document(stone.id).set(payload)

    def nearest_stones(
        self, embedding: list[float], limit: int = 5
    ) -> list[tuple[str, float]]:
        """Return (stone_id, cosine similarity) for the closest stones."""
        query = self.client.collection(STONES).find_nearest(
            vector_field="embedding",
            query_vector=Vector(embedding),
            distance_measure=DistanceMeasure.COSINE,
            limit=limit,
            distance_result_field=DISTANCE_FIELD,
        )
        results: list[tuple[str, float]] = []
        for snap in query.get():
            data = snap.to_dict() or {}
            distance = data.get(DISTANCE_FIELD)
            if distance is None:
                continue
            results.append((snap.id, 1.0 - float(distance)))
        return results

    # ---------- concepts ----------

    def get_concept(self, label: str) -> Concept | None:
        snap = self.client.collection(CONCEPTS).document(_concept_key(label)).get()
        return Concept(**snap.to_dict()) if snap.exists else None

    def save_concept(self, concept: Concept) -> None:
        self.client.collection(CONCEPTS).document(concept.id).set(
            concept.model_dump(mode="json")
        )

    def classify_concept(self, label: str, embedding: list[float]) -> ConceptVerdict:
        """Decide whether the reader already holds a concept.

        An explicit Concept record wins over similarity: it was written by a
        feedback event ("I already knew that", or a resolved branch), which is
        ground truth about this reader rather than an inference about the graph.
        """
        recorded = self.get_concept(label)
        if recorded is not None:
            return ConceptVerdict(
                label=label,
                state=recorded.state,
                similarity=1.0,
                nearest_stone_id=(recorded.evidence[-1] if recorded.evidence else None),
            )

        cfg = settings()
        nearest = self.nearest_stones(embedding, limit=1)
        if not nearest:
            return ConceptVerdict(label=label, state=ConceptState.UNKNOWN, similarity=0.0)

        stone_id, similarity = nearest[0]
        if similarity >= cfg.known_threshold:
            state = ConceptState.KNOWN
        elif similarity >= cfg.partial_threshold:
            state = ConceptState.PARTIAL
        else:
            state = ConceptState.UNKNOWN
        return ConceptVerdict(
            label=label,
            state=state,
            similarity=similarity,
            nearest_stone_id=stone_id,
        )

    def mark_concept(
        self, label: str, state: ConceptState, evidence_stone_id: str | None = None
    ) -> Concept:
        existing = self.get_concept(label)
        evidence = list(existing.evidence) if existing else []
        if evidence_stone_id and evidence_stone_id not in evidence:
            evidence.append(evidence_stone_id)
        concept = Concept(
            id=_concept_key(label),
            label=label,
            state=state,
            evidence=evidence,
            first_seen_paper_id=existing.first_seen_paper_id if existing else None,
            resolved_by_stone_id=evidence_stone_id
            or (existing.resolved_by_stone_id if existing else None),
        )
        self.save_concept(concept)
        return concept

    # ---------- edges and kingdoms ----------

    def add_edge(self, edge: Edge) -> None:
        self.client.collection(EDGES).document(edge.id).set(edge.model_dump(mode="json"))

    def save_kingdom(self, kingdom: Kingdom) -> None:
        self.client.collection(KINGDOMS).document(kingdom.id).set(
            kingdom.model_dump(mode="json")
        )

    def list_kingdoms(self) -> list[Kingdom]:
        return [
            Kingdom(**d.to_dict()) for d in self.client.collection(KINGDOMS).stream()
        ]

    # ---------- sessions ----------

    def save_session(self, session: Session) -> None:
        self.client.collection(SESSIONS).document(session.id).set(
            session.model_dump(mode="json")
        )

    def get_session(self, session_id: str) -> Session | None:
        snap = self.client.collection(SESSIONS).document(session_id).get()
        return Session(**snap.to_dict()) if snap.exists else None

    def append_event(self, session_id: str, event: ReadingEvent) -> None:
        self.client.collection(SESSIONS).document(session_id).update(
            {
                "events": firestore.ArrayUnion([event.model_dump(mode="json")]),
                "last_touched_at": firestore.SERVER_TIMESTAMP,
            }
        )

    def resumable_sessions(self) -> list[Session]:
        docs = (
            self.client.collection(SESSIONS)
            .where(filter=firestore.FieldFilter("state", "==", "suspended"))
            .stream()
        )
        return [Session(**d.to_dict()) for d in docs]

    # ---------- hypotheses and proposals ----------

    def save_hypothesis(self, hypothesis: Hypothesis) -> None:
        self.client.collection(HYPOTHESES).document(hypothesis.id).set(
            hypothesis.model_dump(mode="json")
        )

    def open_hypotheses(self) -> list[Hypothesis]:
        docs = (
            self.client.collection(HYPOTHESES)
            .where(filter=firestore.FieldFilter("status", "in", ["open", "triaged", "running"]))
            .stream()
        )
        return [Hypothesis(**d.to_dict()) for d in docs]

    def save_proposal(self, proposal: Proposal) -> None:
        self.client.collection(PROPOSALS).document(proposal.id).set(
            proposal.model_dump(mode="json")
        )

    # ---------- queue ----------

    def queue(self) -> list[QueueEntry]:
        """Papers ordered by distance from the reader's current frontier."""
        entries: list[QueueEntry] = []
        for paper in self.list_papers():
            unknown: list[str] = []
            for section in paper.sections:
                for label in section.concept_labels:
                    concept = self.get_concept(label)
                    if concept is None or concept.state is ConceptState.UNKNOWN:
                        if label not in unknown:
                            unknown.append(label)
            entries.append(
                QueueEntry(
                    paper_id=paper.id,
                    title=paper.title,
                    gap_count=len(unknown),
                    estimated_minutes=(
                        len(paper.sections) * MINUTES_PER_SECTION
                        + len(unknown) * MINUTES_PER_GAP
                    ),
                    prep_status=paper.prep_status,
                    blocking_concepts=unknown[:5],
                )
            )
        entries.sort(key=lambda e: (e.gap_count, e.estimated_minutes))
        return entries


@lru_cache
def get_store() -> GraphStore:
    return GraphStore()

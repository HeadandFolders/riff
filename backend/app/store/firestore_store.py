"""Firestore-backed graph and process store.

Scale notes, because the naive shape of this does not survive a real library:

* Section prose lives in a ``papers/{id}/sections`` subcollection, so listing
  papers never streams paper bodies.
* The queue reads denormalised ``gap_count`` off the paper document and is a
  single ordered, paginated query — not a fan-out over sections and concepts.
* Concept state is loaded once per request as a projection, not per label.
* When a concept's state changes, only the papers that actually mention it are
  recomputed, found through the ``concept_papers`` reverse index.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from functools import lru_cache
from typing import Iterable, Iterator, Sequence

from google.cloud import firestore
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector

from ..config import settings
from ..schemas import (
    RETEST_INTERVALS,
    Concept,
    ConceptCandidate,
    ConceptPresence,
    Edge,
    GraphEdge,
    GraphNode,
    GraphView,
    Hypothesis,
    Kingdom,
    Misconception,
    Paper,
    PrepStatus,
    Proposal,
    QueueEntry,
    QueuePage,
    SectionBody,
    Session,
    Stone,
    Understanding,
    UnderstandingVerdict,
)

PAPERS = "papers"
SECTIONS = "sections"
STONES = "stones"
CONCEPTS = "concepts"
CONCEPT_PAPERS = "concept_papers"
MISCONCEPTIONS = "misconceptions"
EDGES = "edges"
KINGDOMS = "kingdoms"
SESSIONS = "sessions"
HYPOTHESES = "hypotheses"
PROPOSALS = "proposals"

DISTANCE_FIELD = "_distance"
BATCH_LIMIT = 450

MINUTES_PER_SECTION = 6
MINUTES_PER_GAP = 4

_PAPER_QUEUE_FIELDS = [
    "title",
    "gap_count",
    "estimated_minutes",
    "prep_status",
    "blocking_concepts",
]
_CONCEPT_FLAG_FIELDS = ["label", "presence", "understanding", "misconception_ids"]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def concept_key(label: str) -> str:
    return label.strip().lower().replace(" ", "-")[:200]


def _chunked(items: Sequence[str], size: int = 30) -> Iterator[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class GraphStore:
    def __init__(self, client: firestore.Client | None = None) -> None:
        cfg = settings()
        self.client = client or firestore.Client(
            project=cfg.project_id, database=cfg.firestore_database
        )

    # ------------------------------------------------------------------ papers

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

    def save_section_body(self, body: SectionBody) -> None:
        (
            self.client.collection(PAPERS)
            .document(body.paper_id)
            .collection(SECTIONS)
            .document(body.id)
            .set(body.model_dump(mode="json"))
        )

    def get_section_body(self, paper_id: str, section_id: str) -> SectionBody | None:
        snap = (
            self.client.collection(PAPERS)
            .document(paper_id)
            .collection(SECTIONS)
            .document(section_id)
            .get()
        )
        return SectionBody(**snap.to_dict()) if snap.exists else None

    def register_paper_concepts(self, paper_id: str, labels: Iterable[str]) -> None:
        """Maintain the reverse index that keeps frontier recomputation bounded."""
        batch = self.client.batch()
        writes = 0
        for label in {concept_key(l) for l in labels if l.strip()}:
            ref = self.client.collection(CONCEPT_PAPERS).document(label)
            batch.set(
                ref,
                {"label": label, "paper_ids": firestore.ArrayUnion([paper_id])},
                merge=True,
            )
            writes += 1
            if writes >= BATCH_LIMIT:
                batch.commit()
                batch = self.client.batch()
                writes = 0
        if writes:
            batch.commit()

    def papers_touching(self, labels: Iterable[str]) -> set[str]:
        keys = sorted({concept_key(l) for l in labels if l.strip()})
        if not keys:
            return set()
        paper_ids: set[str] = set()
        for chunk in _chunked(keys):
            refs = [self.client.collection(CONCEPT_PAPERS).document(k) for k in chunk]
            for snap in self.client.get_all(refs):
                if snap.exists:
                    paper_ids.update((snap.to_dict() or {}).get("paper_ids", []))
        return paper_ids

    # ---------------------------------------------------------------- concepts

    def concept_flags(self) -> tuple[set[str], set[str]]:
        """One projection read for the whole concept vocabulary.

        Returns (gap_keys, misconceived_keys). Absent from both means the
        concept is understood well enough not to block anything.
        """
        gaps: set[str] = set()
        misconceived: set[str] = set()
        query = self.client.collection(CONCEPTS).select(_CONCEPT_FLAG_FIELDS)
        for snap in query.stream():
            data = snap.to_dict() or {}
            understanding = data.get("understanding", Understanding.UNASSESSED.value)
            presence = data.get("presence", ConceptPresence.ABSENT.value)
            if understanding == Understanding.MISCONCEIVED.value:
                misconceived.add(snap.id)
                gaps.add(snap.id)
                continue
            if understanding == Understanding.UNASSESSED.value:
                if presence == ConceptPresence.ABSENT.value:
                    gaps.add(snap.id)
            elif understanding in (
                Understanding.ABSENT.value,
                Understanding.PARTIAL.value,
            ):
                gaps.add(snap.id)
        return gaps, misconceived

    def get_concept(self, label: str) -> Concept | None:
        snap = self.client.collection(CONCEPTS).document(concept_key(label)).get()
        return Concept(**snap.to_dict()) if snap.exists else None

    def get_concepts(self, labels: Sequence[str]) -> dict[str, Concept]:
        keys = sorted({concept_key(l) for l in labels if l.strip()})
        found: dict[str, Concept] = {}
        for chunk in _chunked(keys):
            refs = [self.client.collection(CONCEPTS).document(k) for k in chunk]
            for snap in self.client.get_all(refs):
                if snap.exists:
                    concept = Concept(**(snap.to_dict() or {}))
                    found[concept.id] = concept
        return found

    def save_concept(self, concept: Concept) -> None:
        self.client.collection(CONCEPTS).document(concept.id).set(
            concept.model_dump(mode="json")
        )

    # ------------------------------------------------- stones / vector prefilter

    def add_stone(self, stone: Stone, embedding: list[float]) -> None:
        payload = stone.model_dump(mode="json")
        payload["embedding"] = Vector(embedding)
        self.client.collection(STONES).document(stone.id).set(payload)

    def nearest_stones(
        self, embedding: list[float], limit: int = 5
    ) -> list[tuple[str, float]]:
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
            if distance is not None:
                results.append((snap.id, 1.0 - float(distance)))
        return results

    def prefilter_concept(self, label: str, embedding: list[float]) -> ConceptCandidate:
        """Cheap retrieval pass. Says nothing about understanding.

        This exists to decide what is worth asking about, and to avoid paying
        Gemini for concepts the graph has never heard of. The verdict on
        understanding comes from the Assessor.
        """
        recorded = self.get_concept(label)
        nearest = self.nearest_stones(embedding, limit=1)
        similarity = nearest[0][1] if nearest else 0.0
        stone_id = nearest[0][0] if nearest else None

        presence = (
            ConceptPresence.IN_GRAPH
            if similarity >= settings().presence_threshold
            else ConceptPresence.ABSENT
        )
        if recorded is not None:
            open_ids = recorded.misconception_ids
            return ConceptCandidate(
                label=label,
                presence=max(presence, recorded.presence, key=_presence_rank),
                similarity=similarity,
                nearest_stone_id=stone_id,
                understanding=recorded.understanding,
                open_misconceptions=open_ids,
            )
        return ConceptCandidate(
            label=label,
            presence=presence,
            similarity=similarity,
            nearest_stone_id=stone_id,
        )

    # ----------------------------------------------------------- assessments

    def record_assessment(
        self,
        verdict: UnderstandingVerdict,
        *,
        paper_id: str | None = None,
        section_id: str | None = None,
        session_id: str | None = None,
        evidence_stone_id: str | None = None,
    ) -> Concept:
        """Persist a reasoning verdict, its misconceptions, and reschedule retests."""
        key = concept_key(verdict.concept_label)
        existing = self.get_concept(verdict.concept_label)
        now = datetime.now(timezone.utc)

        misconception_ids: list[str] = []
        batch = self.client.batch()
        for detected in verdict.misconceptions:
            prior = self._find_misconception(key, detected.belief)
            if prior is not None:
                prior.times_observed += 1
                prior.status = "recurring"
                prior.last_seen_at = now
                prior.correction = detected.correction
                prior.severity = detected.severity
                prior.next_retest_at = now + RETEST_INTERVALS[Understanding.MISCONCEIVED]
                record = prior
            else:
                record = Misconception(
                    id=new_id("misc"),
                    concept_label=verdict.concept_label,
                    concept_key=key,
                    belief=detected.belief,
                    correction=detected.correction,
                    severity=detected.severity,
                    paper_id=paper_id,
                    section_id=section_id,
                    session_id=session_id,
                    next_retest_at=now + RETEST_INTERVALS[Understanding.MISCONCEIVED],
                )
            batch.set(
                self.client.collection(MISCONCEPTIONS).document(record.id),
                record.model_dump(mode="json"),
            )
            misconception_ids.append(record.id)

        # Misconceptions not restated in a passing assessment are considered cleared.
        if verdict.level in (Understanding.SOLID, Understanding.PARTIAL) and existing:
            for stale_id in set(existing.misconception_ids) - set(misconception_ids):
                batch.update(
                    self.client.collection(MISCONCEPTIONS).document(stale_id),
                    {"status": "addressed", "last_seen_at": now.isoformat()},
                )

        evidence = list(existing.evidence) if existing else []
        if evidence_stone_id and evidence_stone_id not in evidence:
            evidence.append(evidence_stone_id)

        concept = Concept(
            id=key,
            label=verdict.concept_label,
            presence=(
                existing.presence
                if existing
                else ConceptPresence.IN_GRAPH
                if evidence_stone_id
                else ConceptPresence.ABSENT
            ),
            understanding=verdict.level,
            retrieval_similarity=existing.retrieval_similarity if existing else 0.0,
            evidence=evidence,
            misconception_ids=misconception_ids,
            assessment_count=(existing.assessment_count + 1) if existing else 1,
            last_assessed_at=now,
            next_retest_at=now + RETEST_INTERVALS.get(verdict.level, RETEST_INTERVALS[Understanding.PARTIAL]),
            first_seen_paper_id=(
                existing.first_seen_paper_id if existing else paper_id
            ),
        )
        batch.set(
            self.client.collection(CONCEPTS).document(key),
            concept.model_dump(mode="json"),
        )
        batch.commit()

        self.mark_frontier_stale([verdict.concept_label])
        return concept

    def _find_misconception(self, concept_key_: str, belief: str) -> Misconception | None:
        docs = (
            self.client.collection(MISCONCEPTIONS)
            .where(filter=firestore.FieldFilter("concept_key", "==", concept_key_))
            .where(filter=firestore.FieldFilter("status", "in", ["open", "recurring"]))
            .limit(20)
            .stream()
        )
        needle = belief.strip().lower()[:80]
        for snap in docs:
            record = Misconception(**(snap.to_dict() or {}))
            if record.belief.strip().lower()[:80] == needle:
                return record
        return None

    def misconceptions_for(self, label: str, limit: int = 10) -> list[Misconception]:
        docs = (
            self.client.collection(MISCONCEPTIONS)
            .where(
                filter=firestore.FieldFilter("concept_key", "==", concept_key(label))
            )
            .where(filter=firestore.FieldFilter("status", "in", ["open", "recurring"]))
            .limit(limit)
            .stream()
        )
        return [Misconception(**(d.to_dict() or {})) for d in docs]

    def open_misconceptions(self, limit: int = 100) -> list[Misconception]:
        docs = (
            self.client.collection(MISCONCEPTIONS)
            .where(filter=firestore.FieldFilter("status", "in", ["open", "recurring"]))
            .limit(limit)
            .stream()
        )
        return [Misconception(**(d.to_dict() or {})) for d in docs]

    def misconceptions_due(self, now: datetime | None = None) -> list[Misconception]:
        cutoff = (now or datetime.now(timezone.utc)).isoformat()
        docs = (
            self.client.collection(MISCONCEPTIONS)
            .where(filter=firestore.FieldFilter("status", "in", ["open", "recurring"]))
            .where(filter=firestore.FieldFilter("next_retest_at", "<=", cutoff))
            .limit(50)
            .stream()
        )
        return [Misconception(**(d.to_dict() or {})) for d in docs]

    # -------------------------------------------------------- frontier / queue

    def mark_frontier_stale(self, changed_labels: Iterable[str]) -> set[str]:
        """Flag only the papers that mention a changed concept."""
        paper_ids = self.papers_touching(changed_labels)
        batch = self.client.batch()
        writes = 0
        for paper_id in paper_ids:
            batch.update(
                self.client.collection(PAPERS).document(paper_id),
                {"frontier_stale": True},
            )
            writes += 1
            if writes >= BATCH_LIMIT:
                batch.commit()
                batch = self.client.batch()
                writes = 0
        if writes:
            batch.commit()
        return paper_ids

    def recompute_frontier(self, paper_ids: Iterable[str] | None = None) -> int:
        """Recompute denormalised queue fields.

        With no argument, recomputes every paper flagged stale. Cost is one
        concept projection plus one read and one write per affected paper.
        """
        gaps, misconceived = self.concept_flags()
        known = gaps | misconceived

        if paper_ids is None:
            docs = (
                self.client.collection(PAPERS)
                .where(filter=firestore.FieldFilter("frontier_stale", "==", True))
                .stream()
            )
            papers = [Paper(**(d.to_dict() or {})) for d in docs]
        else:
            ids = sorted(set(paper_ids))
            papers = []
            for chunk in _chunked(ids):
                refs = [self.client.collection(PAPERS).document(i) for i in chunk]
                papers.extend(
                    Paper(**(s.to_dict() or {})) for s in self.client.get_all(refs) if s.exists
                )

        batch = self.client.batch()
        writes = 0
        for paper in papers:
            blocking: list[str] = []
            for section in paper.section_index:
                for label in section.concept_labels:
                    key = concept_key(label)
                    if key in gaps or key not in known:
                        if label not in blocking:
                            blocking.append(label)
            batch.update(
                self.client.collection(PAPERS).document(paper.id),
                {
                    "gap_count": len(blocking),
                    "blocking_concepts": blocking[:8],
                    "estimated_minutes": (
                        len(paper.section_index) * MINUTES_PER_SECTION
                        + len(blocking) * MINUTES_PER_GAP
                    ),
                    "frontier_stale": False,
                },
            )
            writes += 1
            if writes >= BATCH_LIMIT:
                batch.commit()
                batch = self.client.batch()
                writes = 0
        if writes:
            batch.commit()
        return len(papers)

    def queue(self, limit: int = 20, cursor: str | None = None) -> QueuePage:
        """Ordered, paginated, and independent of library size.

        Requires a composite index on (prep_status asc, gap_count asc,
        estimated_minutes asc) over the papers collection.
        """
        query = (
            self.client.collection(PAPERS)
            .where(
                filter=firestore.FieldFilter("prep_status", "==", PrepStatus.READY.value)
            )
            .order_by("gap_count")
            .order_by("estimated_minutes")
            .select(_PAPER_QUEUE_FIELDS)
        )
        if cursor:
            anchor = self.client.collection(PAPERS).document(cursor).get()
            if anchor.exists:
                query = query.start_after(anchor)

        snaps = list(query.limit(limit).stream())
        _, misconceived = self.concept_flags()

        entries: list[QueueEntry] = []
        for snap in snaps:
            data = snap.to_dict() or {}
            blocking = data.get("blocking_concepts", [])
            entries.append(
                QueueEntry(
                    paper_id=snap.id,
                    title=data.get("title", "(untitled)"),
                    gap_count=int(data.get("gap_count", 0)),
                    misconception_count=sum(
                        1 for b in blocking if concept_key(b) in misconceived
                    ),
                    estimated_minutes=int(data.get("estimated_minutes", 0)),
                    prep_status=PrepStatus(
                        data.get("prep_status", PrepStatus.PENDING.value)
                    ),
                    blocking_concepts=blocking[:5],
                )
            )

        return QueuePage(
            entries=entries,
            next_cursor=snaps[-1].id if len(snaps) == limit else None,
            total_ready=self._count_ready(),
        )

    def _count_ready(self) -> int | None:
        try:
            result = (
                self.client.collection(PAPERS)
                .where(
                    filter=firestore.FieldFilter(
                        "prep_status", "==", PrepStatus.READY.value
                    )
                )
                .count()
                .get()
            )
            return int(result[0][0].value)
        except Exception:
            # Aggregation support varies by client version; the queue works without it.
            return None

    # ----------------------------------------------------- edges and kingdoms

    def add_edge(self, edge: Edge) -> None:
        self.client.collection(EDGES).document(edge.id).set(edge.model_dump(mode="json"))

    def add_edges(self, edges: Sequence[Edge]) -> None:
        batch = self.client.batch()
        for index, edge in enumerate(edges, start=1):
            batch.set(
                self.client.collection(EDGES).document(edge.id),
                edge.model_dump(mode="json"),
            )
            if index % BATCH_LIMIT == 0:
                batch.commit()
                batch = self.client.batch()
        batch.commit()

    def save_kingdom(self, kingdom: Kingdom) -> None:
        self.client.collection(KINGDOMS).document(kingdom.id).set(
            kingdom.model_dump(mode="json")
        )

    def list_kingdoms(self) -> list[Kingdom]:
        return [
            Kingdom(**(d.to_dict() or {}))
            for d in self.client.collection(KINGDOMS).stream()
        ]

    def graph_view(
        self, kingdom_id: str | None = None, limit: int = 600
    ) -> GraphView:
        """Nodes and edges for the map, capped so the client stays responsive."""
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        truncated = False

        kingdoms = self.list_kingdoms()
        for kingdom in kingdoms:
            if kingdom_id and kingdom.id != kingdom_id:
                continue
            nodes.append(
                GraphNode(
                    id=kingdom.id,
                    kind="kingdom",
                    label=kingdom.label,
                    degree=len(kingdom.paper_ids),
                )
            )

        paper_query = self.client.collection(PAPERS).select(["title", "kingdom_id"])
        if kingdom_id:
            paper_query = paper_query.where(
                filter=firestore.FieldFilter("kingdom_id", "==", kingdom_id)
            )
        paper_ids: set[str] = set()
        for snap in paper_query.limit(limit).stream():
            data = snap.to_dict() or {}
            paper_ids.add(snap.id)
            nodes.append(
                GraphNode(
                    id=snap.id,
                    kind="paper",
                    label=data.get("title", "(untitled)"),
                    kingdom_id=data.get("kingdom_id"),
                )
            )
            if data.get("kingdom_id"):
                edges.append(
                    GraphEdge(
                        source=snap.id, target=data["kingdom_id"], kind="member"
                    )
                )

        remaining = max(limit - len(nodes), 0)
        concept_snaps = list(
            self.client.collection(CONCEPTS)
            .select(["label", "understanding", "misconception_ids"])
            .limit(remaining + 1)
            .stream()
        )
        if len(concept_snaps) > remaining:
            truncated = True
            concept_snaps = concept_snaps[:remaining]

        concept_labels: set[str] = set()
        for snap in concept_snaps:
            data = snap.to_dict() or {}
            concept_labels.add(snap.id)
            nodes.append(
                GraphNode(
                    id=snap.id,
                    kind="concept",
                    label=data.get("label", snap.id),
                    understanding=Understanding(
                        data.get("understanding", Understanding.UNASSESSED.value)
                    ),
                    misconception_count=len(data.get("misconception_ids", [])),
                )
            )

        # concept -> paper edges come from the reverse index we already maintain
        for snap in self.client.collection(CONCEPT_PAPERS).stream():
            if snap.id not in concept_labels:
                continue
            for pid in (snap.to_dict() or {}).get("paper_ids", []):
                if pid in paper_ids:
                    edges.append(
                        GraphEdge(source=snap.id, target=pid, kind="appears_in")
                    )

        node_ids = {n.id for n in nodes}
        for snap in self.client.collection(EDGES).limit(limit * 2).stream():
            edge = Edge(**(snap.to_dict() or {}))
            if edge.from_id in node_ids and edge.to_id in node_ids:
                edges.append(
                    GraphEdge(
                        source=edge.from_id,
                        target=edge.to_id,
                        kind=edge.kind,
                        weight=edge.weight,
                    )
                )

        degree: dict[str, int] = {}
        for edge in edges:
            degree[edge.source] = degree.get(edge.source, 0) + 1
            degree[edge.target] = degree.get(edge.target, 0) + 1
        for node in nodes:
            node.degree = max(node.degree, degree.get(node.id, 0))

        return GraphView(nodes=nodes, edges=edges, truncated=truncated)

    # ---------------------------------------------------------------- sessions

    def save_session(self, session: Session) -> None:
        self.client.collection(SESSIONS).document(session.id).set(
            session.model_dump(mode="json")
        )

    def get_session(self, session_id: str) -> Session | None:
        snap = self.client.collection(SESSIONS).document(session_id).get()
        return Session(**(snap.to_dict() or {})) if snap.exists else None

    def append_event(self, session_id: str, event) -> None:
        self.client.collection(SESSIONS).document(session_id).update(
            {
                "events": firestore.ArrayUnion([event.model_dump(mode="json")]),
                "last_touched_at": firestore.SERVER_TIMESTAMP,
            }
        )

    def resumable_sessions(self, limit: int = 20) -> list[Session]:
        docs = (
            self.client.collection(SESSIONS)
            .where(filter=firestore.FieldFilter("state", "==", "suspended"))
            .order_by("last_touched_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        return [Session(**(d.to_dict() or {})) for d in docs]

    # ------------------------------------------------ hypotheses and proposals

    def save_hypothesis(self, hypothesis: Hypothesis) -> None:
        self.client.collection(HYPOTHESES).document(hypothesis.id).set(
            hypothesis.model_dump(mode="json")
        )

    def open_hypotheses(self, limit: int = 100) -> list[Hypothesis]:
        docs = (
            self.client.collection(HYPOTHESES)
            .where(
                filter=firestore.FieldFilter(
                    "status", "in", ["open", "triaged", "running"]
                )
            )
            .limit(limit)
            .stream()
        )
        return [Hypothesis(**(d.to_dict() or {})) for d in docs]

    def save_proposal(self, proposal: Proposal) -> None:
        self.client.collection(PROPOSALS).document(proposal.id).set(
            proposal.model_dump(mode="json")
        )


_PRESENCE_ORDER = {ConceptPresence.ABSENT: 0, ConceptPresence.IN_GRAPH: 1}


def _presence_rank(presence: ConceptPresence) -> int:
    return _PRESENCE_ORDER[presence]


@lru_cache
def get_store() -> GraphStore:
    return GraphStore()

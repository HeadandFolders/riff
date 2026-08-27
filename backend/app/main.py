from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import agents, ingest
from .config import settings
from .schemas import (
    GraphView,
    Kingdom,
    Misconception,
    Paper,
    QueuePage,
    SectionBody,
    Session,
    UnderstandingVerdict,
)
from .store import GraphStore, get_store

app = FastAPI(title="riff", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProbeRequest(BaseModel):
    concept_label: str
    section_title: str = ""
    section_excerpt: str = ""


class ArxivRequest(BaseModel):
    reference: str = Field(
        description="arXiv id, abs link, or pdf link — all resolve to the same paper"
    )


class AssessRequest(BaseModel):
    concept_label: str
    explanation: str
    section_title: str = ""
    section_excerpt: str = ""
    paper_id: str | None = None
    section_id: str | None = None
    session_id: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    cfg = settings()
    return {"status": "ok", "project": cfg.project_id, "model": cfg.reasoning_model}


@app.get("/queue", response_model=QueuePage)
def queue(
    limit: int | None = None,
    cursor: str | None = None,
    store: GraphStore = Depends(get_store),
) -> QueuePage:
    return store.queue(limit=limit or settings().queue_page_size, cursor=cursor)


@app.get("/graph", response_model=GraphView)
def graph(
    kingdom_id: str | None = None,
    limit: int | None = None,
    store: GraphStore = Depends(get_store),
) -> GraphView:
    return store.graph_view(
        kingdom_id=kingdom_id, limit=limit or settings().graph_node_limit
    )


@app.get("/kingdoms", response_model=list[Kingdom])
def kingdoms(store: GraphStore = Depends(get_store)) -> list[Kingdom]:
    return store.list_kingdoms()


@app.post("/papers/arxiv", response_model=Paper, status_code=202)
def add_arxiv_paper(
    request: ArxivRequest,
    background: BackgroundTasks,
    store: GraphStore = Depends(get_store),
) -> Paper:
    """Accept the link and prepare in the background; poll the paper for status."""
    try:
        url = ingest.canonical_arxiv_url(request.reference)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    paper = ingest.stub_paper(store, "arxiv", url, url)
    background.add_task(ingest.prepare_arxiv, store, paper.id)
    return paper


@app.post("/papers/pdf", response_model=Paper, status_code=202)
async def add_pdf_paper(
    background: BackgroundTasks,
    pdf: UploadFile = File(...),
    store: GraphStore = Depends(get_store),
) -> Paper:
    payload = await pdf.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty pdf upload")
    name = pdf.filename or "dropped.pdf"
    paper = ingest.stub_paper(store, "pdf", name, name)
    background.add_task(ingest.prepare_pdf, store, paper.id, payload)
    return paper


@app.get("/papers/{paper_id}", response_model=Paper)
def get_paper(paper_id: str, store: GraphStore = Depends(get_store)) -> Paper:
    paper = store.get_paper(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="paper not found")
    return paper


@app.get("/papers/{paper_id}/sections/{section_id}", response_model=SectionBody)
def get_section(
    paper_id: str, section_id: str, store: GraphStore = Depends(get_store)
) -> SectionBody:
    body = store.get_section_body(paper_id, section_id)
    if body is None:
        raise HTTPException(status_code=404, detail="section not found")
    return body


@app.get("/sessions/resumable", response_model=list[Session])
def resumable(store: GraphStore = Depends(get_store)) -> list[Session]:
    return store.resumable_sessions()


@app.get("/misconceptions", response_model=list[Misconception])
def misconceptions(
    due_only: bool = False, store: GraphStore = Depends(get_store)
) -> list[Misconception]:
    return store.misconceptions_due() if due_only else store.open_misconceptions()


@app.post("/probe", response_model=agents.Probe)
def make_probe(request: ProbeRequest) -> agents.Probe:
    return agents.probe(
        request.concept_label, request.section_title, request.section_excerpt
    )


@app.post("/assess", response_model=UnderstandingVerdict)
def assess(
    request: AssessRequest, store: GraphStore = Depends(get_store)
) -> UnderstandingVerdict:
    verdict = agents.assess_text(
        request.concept_label,
        request.explanation,
        section_title=request.section_title,
        section_excerpt=request.section_excerpt,
        priors=store.misconceptions_for(request.concept_label),
    )
    store.record_assessment(
        verdict,
        paper_id=request.paper_id,
        section_id=request.section_id,
        session_id=request.session_id,
    )
    return verdict


@app.post("/assess/audio", response_model=UnderstandingVerdict)
async def assess_audio(
    concept_label: str = Form(...),
    section_title: str = Form(""),
    section_excerpt: str = Form(""),
    paper_id: str | None = Form(None),
    section_id: str | None = Form(None),
    session_id: str | None = Form(None),
    audio: UploadFile = File(...),
    store: GraphStore = Depends(get_store),
) -> UnderstandingVerdict:
    payload = await audio.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty audio upload")
    verdict = agents.assess_audio(
        concept_label,
        payload,
        mime_type=audio.content_type or "audio/webm",
        section_title=section_title,
        section_excerpt=section_excerpt,
        priors=store.misconceptions_for(concept_label),
    )
    store.record_assessment(
        verdict, paper_id=paper_id, section_id=section_id, session_id=session_id
    )
    return verdict


@app.post("/admin/frontier/recompute")
def recompute_frontier(store: GraphStore = Depends(get_store)) -> dict[str, int]:
    return {"papers_recomputed": store.recompute_frontier()}

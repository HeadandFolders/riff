from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .schemas import Kingdom, Paper, QueueEntry, Session
from .store import GraphStore, get_store

app = FastAPI(title="riff", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    cfg = settings()
    return {
        "status": "ok",
        "project": cfg.project_id,
        "model": cfg.reasoning_model,
    }


@app.get("/queue", response_model=list[QueueEntry])
def queue(store: GraphStore = Depends(get_store)) -> list[QueueEntry]:
    return store.queue()


@app.get("/papers/{paper_id}", response_model=Paper)
def get_paper(paper_id: str, store: GraphStore = Depends(get_store)) -> Paper:
    paper = store.get_paper(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="paper not found")
    return paper


@app.get("/kingdoms", response_model=list[Kingdom])
def kingdoms(store: GraphStore = Depends(get_store)) -> list[Kingdom]:
    return store.list_kingdoms()


@app.get("/sessions/resumable", response_model=list[Session])
def resumable(store: GraphStore = Depends(get_store)) -> list[Session]:
    return store.resumable_sessions()

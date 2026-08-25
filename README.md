# riff

A reading partner for research papers that holds your place across days, explains
each section starting from what you already understand, and refuses to let a paper
close until you have committed a falsifiable hypothesis about its future work.

Built for the [All Things Agentic Hackathon](https://allthingsagentichackathon.devpost.com/)
— Collaborative Partner track.

## The problem

Reading research is gated by three things that compound:

1. **You can't finish a paper in one sitting.** Papers have no resume button, so every
   restart costs the mental stack you had built.
2. **Missing prerequisites stall you mid-read.** You hit a concept the paper assumes,
   and chasing it means losing your place — so you bail to the abstract.
3. **Abstract-skimming produces no artifact.** You keep up, and generate nothing.

riff attacks all three: sessions are durable, gaps are detected and filled inline
without leaving the section, and every paper ends in a committed prediction that
accumulates into a timestamped research ledger.

## The loop

```
arXiv link ─┐
            ├─► prepared ─► gap map ─► section read ⇄ branch primer
PDF drop ───┘                              │
                                           ▼
                              comprehension check ──(fail)──► re-read
                                           │
                                           ▼
                              future-work interrogation
                                           │
                                           ▼
                          hypothesis: claim + confidence + falsifier
                                           │
                                           ▼
                         proposal doc ─► scout monitors ─► resolved
```

## Agents

| Agent | Job |
| --- | --- |
| `ReadingCoordinator` | Owns the session state machine, routes each turn |
| `Cartographer` | Extracts concepts, classifies known / partial / unknown by vector search |
| `Explainer` | Teaches one section, chooses primer / analogy / code grounding |
| `Examiner` | Comprehension checks, future-work interrogation, hypothesis stress-test |
| `Triage` | Scores falsifiability, cost, prior-art collision; renders the proposal |
| `Scout` | One scheduled query per open hypothesis, proposes resolutions |

Something becomes an agent only if it needs its own reasoning loop and can fail
independently. Fetching, sectioning, and persistence are tools, not agents.

## The graph

Three tiers, each built from the one below by embedding similarity:

- **stone** — one chunk of understanding: a section you read, or a primer that filled a gap
- **castle** — a paper plus every branch tower built while reading it
- **kingdom** — a cluster of castles sharing embedding space, labelled by Gemini

## Stack

| Requirement | Choice |
| --- | --- |
| Gemini 3.5+ | `gemini-3.5-flash` via Vertex AI — reasoning, PDF and audio understanding |
| Google agent framework | Google ADK |
| Google Cloud infrastructure | Cloud Run, Firestore (+ native vector KNN), Cloud Storage, Pub/Sub, Cloud Scheduler |

One external dependency: the [alphaXiv MCP server](https://www.alphaxiv.org/docs/mcp)
for retrieval only — pre-structured content for arXiv links, full-text search for
the scouts, repository files when grounding a claim against real code. All judgement
stays on Gemini.

There is no PDF text-extraction layer. Gemini reads dropped PDFs directly, which
avoids two-column layouts and inline math — where this kind of project usually dies.

## Setup

### Prerequisites

- Python 3.11+
- A Google Cloud project with billing enabled
- `gcloud` CLI, authenticated

### 1. Enable services

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  firestore.googleapis.com \
  run.googleapis.com \
  storage.googleapis.com \
  pubsub.googleapis.com \
  cloudscheduler.googleapis.com
```

### 2. Create the Firestore database and vector index

```bash
gcloud firestore databases create --location=nam5

gcloud firestore indexes composite create \
  --collection-group=stones \
  --query-scope=COLLECTION \
  --field-config=field-path=embedding,vector-config='{"dimension":"768","flat":"{}"}'
```

The dimension must match `RIFF_EMBEDDING_DIMENSIONS`.

### 3. Configure

```bash
cd backend
cp .env.example .env   # then fill in RIFF_PROJECT_ID and RIFF_ALPHAXIV_API_KEY
```

### 4. Install and run

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
gcloud auth application-default login

uvicorn app.main:app --reload --port 8080
```

Verify with `curl http://localhost:8080/health`.

## Deploy

```bash
gcloud run deploy riff-api \
  --source backend \
  --region us-central1 \
  --min-instances 0 \
  --set-env-vars RIFF_PROJECT_ID=$(gcloud config get-value project)
```

`--min-instances 0` matters: the service scales to zero between demos, so idle
cost is effectively nothing.

## Cost notes

Vertex AI is the only meaningful spend against the $150 hackathon credit. Three
habits keep it low:

- **Send sections, never whole papers.** Re-sending a full PDF each turn is how you
  burn a hundred dollars in an afternoon.
- **Prepare once per paper and cache it.** Sectioning, tagging, and the concept
  inventory are computed on ingest, not per session.
- **Batch the scouts weekly**, all open hypotheses in one call.

Do not plan on GPUs. Trial-account quota is usually the blocker, not price.

## Status

Built:

- [x] Data model and Firestore store with native vector KNN
- [x] Concept classification with explicit feedback overriding similarity
- [x] Gap-ranked reading queue
- [ ] Ingest — arXiv via alphaXiv MCP, PDF via Gemini multimodal
- [ ] Cartographer, Explainer, Examiner, Triage, Scout
- [ ] Session state machine with suspend and resume
- [ ] Reading pane and kingdom map
- [ ] Proposal document rendering

Deferred past submission: PWA voice mode, implementation coach, research diary
generator, execution handoff to an external agent runner.

## Submission checklist

- [ ] Public repo with this README and reproducible setup
- [ ] Architecture diagram
- [ ] Demo video under 4 minutes, showing suspend/resume and the hypothesis gate live
- [ ] Proof the backend ran on Google Cloud
- [ ] `pip freeze > requirements.lock.txt`

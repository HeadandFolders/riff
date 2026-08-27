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

## Understanding is a reasoning problem, not a similarity problem

The tempting shortcut is to embed everything and call cosine similarity
comprehension. It cannot work: similarity tells you the graph holds material
*near* a concept, which is a completely different claim from you being able to
derive a result. It also cannot distinguish a reader who understands a mechanism
from one who has merely seen the words.

So riff separates the two:

| Stage | Question | Method | Authority |
| --- | --- | --- | --- |
| Prefilter | Does the graph hold material for this concept? | Vector KNN | Decides what's worth asking about |
| Assessment | Does the reader understand it? | Gemini grades their own explanation | The verdict |

The prefilter exists purely as a cost control — it stops us paying for a Gemini
call on concepts the graph has never encountered. Every judgement that matters
comes from the `Assessor`, which reads an explanation you wrote or spoke and
returns one of `solid`, `partial`, `misconceived`, or `absent`.

Two consequences worth knowing:

- **`misconceived` is worse than `absent`.** Confidently believing something false
  silently corrupts everything built on top of it, so the Assessor prefers that
  verdict whenever a stated belief is actively wrong. Admitting ignorance is
  scored `absent` and never penalised.
- **Misconceptions are named, stored, and retested.** Each one records what you
  believed in your own words plus why it fails, and is rescheduled for retest
  (3 days for a misconception, 10 for partial, 45 for solid). Recurrence is
  tracked rather than re-detected as a fresh problem.

## The loop

```
arXiv link ─┐
            ├─► prepared ─► gap map ─► section read ⇄ branch primer
PDF drop ───┘                              │
                                           ▼
                          explain it back → Assessor verdict
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
| `Cartographer` | Extracts concepts, ranks which are worth assessing |
| `Explainer` | Teaches one section, chooses primer / analogy / code grounding |
| `Assessor` | Grades your explanation, names misconceptions, schedules retests |
| `Examiner` | Comprehension checks, future-work interrogation, hypothesis stress-test |
| `Triage` | Scores falsifiability, cost, prior-art collision; renders the proposal |
| `Scout` | One scheduled query per open hypothesis, proposes resolutions |

Something becomes an agent only if it needs its own reasoning loop and can fail
independently. Fetching, sectioning, and persistence are tools, not agents.

## The graph

Three tiers, each built from the one below:

- **stone** — one chunk of understanding: a section you read, or a primer that filled a gap
- **castle** — a paper plus every branch tower built while reading it
- **kingdom** — a cluster of castles sharing embedding space, labelled by Gemini

The map at `/` renders all three, with concepts coloured by *understanding* rather
than by topic — so a red node is a live misconception you can click into, not
decoration.

## Designed for a real library, not ten papers

The naive shape of this collapses somewhere around fifty papers. What keeps it flat:

- **Section prose lives in a subcollection** (`papers/{id}/sections/{id}`), so listing
  or ranking papers never streams paper bodies.
- **The queue reads denormalised `gap_count`** off the paper document and is a single
  ordered, paginated query — not a fan-out over every section and concept.
- **Concept state loads once per request** as a field projection, not once per label.
- **Concept changes recompute only affected papers**, found through the
  `concept_papers` reverse index, rather than rebuilding the whole frontier.

Net effect: queue cost is O(page size) instead of O(papers x sections x concepts).

## Stack

| Requirement | Choice |
| --- | --- |
| Gemini 3.5+ | `gemini-3.5-flash` via Vertex AI — reasoning, PDF and audio understanding |
| Google agent framework | Google ADK |
| Google Cloud infrastructure | Cloud Run, Firestore (+ native vector KNN), Cloud Storage, Pub/Sub, Cloud Scheduler |

One external dependency: the [alphaXiv MCP server](https://www.alphaxiv.org/docs/mcp)
for retrieval only — pre-structured content for arXiv links, full-text search for
the scouts, repository files when grounding a claim against real code.

There is no PDF text-extraction layer. Gemini reads dropped PDFs directly, which
avoids two-column layouts and inline math — where this kind of project usually dies.
Spoken answers likewise go straight to Gemini, with no transcription step.

## Setup

### Prerequisites

- Python 3.11+
- Node.js 20+ (frontend)
- A Google Cloud project with billing enabled, and the `gcloud` CLI authenticated

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

### 2. Create the database and indexes

```bash
gcloud firestore databases create --location=nam5
```

Vector index for the retrieval prefilter — the dimension must match
`RIFF_EMBEDDING_DIMENSIONS`:

```bash
gcloud firestore indexes composite create \
  --collection-group=stones \
  --query-scope=COLLECTION \
  --field-config='field-path=embedding,vector-config={dimension=768,flat}'
```

Composite index the queue depends on:

```bash
gcloud firestore indexes composite create \
  --collection-group=papers \
  --query-scope=COLLECTION \
  --field-config=field-path=prep_status,order=ascending \
  --field-config=field-path=gap_count,order=ascending \
  --field-config=field-path=estimated_minutes,order=ascending
```

Composite indexes for misconception lookup and retest scheduling:

```bash
gcloud firestore indexes composite create \
  --collection-group=misconceptions \
  --query-scope=COLLECTION \
  --field-config=field-path=concept_key,order=ascending \
  --field-config=field-path=status,order=ascending

gcloud firestore indexes composite create \
  --collection-group=misconceptions \
  --query-scope=COLLECTION \
  --field-config=field-path=status,order=ascending \
  --field-config=field-path=next_retest_at,order=ascending

gcloud firestore indexes composite create \
  --collection-group=sessions \
  --query-scope=COLLECTION \
  --field-config=field-path=state,order=ascending \
  --field-config=field-path=last_touched_at,order=descending
```

If you skip one, Firestore's error response contains a direct console link that
creates exactly the missing index — the fastest way to fix a typo here.

### 3. Backend

```bash
cd backend
cp .env.example .env        # fill in RIFF_PROJECT_ID and RIFF_ALPHAXIV_API_KEY

python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS / Linux

pip install -r requirements.txt
gcloud auth application-default login

uvicorn app.main:app --reload --port 8080
```

Verify with `curl http://localhost:8080/health`.

`RIFF_VERTEX_LOCATION` defaults to `global`, which is the only Vertex endpoint
serving Gemini 3.x — regional endpoints answer 404 for those models. It is
deliberately separate from `RIFF_LOCATION`, which is the region for Cloud Run
and the PDF bucket.

### 4. Frontend

```bash
cd frontend
npm install
npm run dev                 # http://localhost:5173, proxies /api to :8080
```

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /queue?limit&cursor` | Papers ordered by distance from your frontier |
| `GET /graph?kingdom_id&limit` | Nodes and edges for the map |
| `GET /papers/{id}` | Paper metadata and section index |
| `GET /papers/{id}/sections/{id}` | Section prose |
| `GET /misconceptions?due_only` | Open misconceptions, or those due for retest |
| `POST /probe` | One question that reveals whether you understand a concept |
| `POST /assess` | Grade a written explanation, persist the verdict |
| `POST /assess/audio` | Same, from spoken audio, no transcription step |
| `POST /admin/frontier/recompute` | Recompute stale queue rankings |

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

Vertex AI is the only meaningful spend against the $150 hackathon credit:

- **Send sections, never whole papers.** Re-sending a full PDF each turn is how you
  burn a hundred dollars in an afternoon.
- **Prepare once per paper and cache it.** Sectioning, tagging, and the concept
  inventory are computed on ingest, not per session.
- **Cap assessments per section** via `RIFF_MAX_ASSESSMENTS_PER_SECTION`. Each one is
  a Gemini call, so the Cartographer ranks candidates and only the top few are graded.
- **Batch the scouts weekly**, all open hypotheses in one call.

Do not plan on GPUs. Trial-account quota is usually the blocker, not price.

## Status

Built:

- [x] Data model, Firestore store, native vector KNN prefilter
- [x] Reasoning-based understanding assessment with tracked, retested misconceptions
- [x] Gap-ranked queue that stays flat as the library grows
- [x] Graph API and navigable kingdom map with pan, zoom, search, and filtering
- [ ] Ingest — arXiv via alphaXiv MCP, PDF via Gemini multimodal
- [ ] Cartographer, Explainer, Examiner, Triage, Scout
- [ ] Session state machine with suspend and resume
- [ ] Reading pane
- [ ] Proposal document rendering

Deferred past submission: PWA voice mode, implementation coach, research diary
generator, execution handoff to an external agent runner.

## Submission checklist

- [ ] Public repo with this README and reproducible setup
- [ ] Architecture diagram
- [ ] Demo video under 4 minutes, showing suspend/resume and the hypothesis gate live
- [ ] Proof the backend ran on Google Cloud
- [ ] `pip freeze > requirements.lock.txt`

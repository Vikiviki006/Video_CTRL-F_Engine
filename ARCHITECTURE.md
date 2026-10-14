# VideoCtrl-F — Architecture & Technology Stack

A standalone reference for how the system is put together and what it is built
from. `Readme.md` remains the long-form narrative (file-by-file tour, scoring
derivation, evaluation reference); this document is the architecture view.

**Status of this document.** Everything in §1–§6 was verified by running the
system on this machine. §7 separates what is implemented from what is intended.
§8 lists the actual installed versions. Gaps and defects found during that work
are recorded in [`MISSING_FILES.md`](MISSING_FILES.md).

---

## 1. What the system does

A multimodal video search engine. A video is decoded into keyframes, each
keyframe is embedded into a shared image–text space, and those embeddings are
indexed for nearest-neighbour search. A text query is embedded into the same
space; the closest frames come back as timestamps and clips.

On top of that sits a **relational graph** per video — scenes, frames, and
measured frame-to-frame similarity — used to re-rank results and, importantly,
to explain *why* each result was returned.

The interesting claim is not that any one component is novel. It is that the
vector-only baseline is preserved unchanged, so the graph layer's effect can be
measured as a controlled experiment rather than asserted.

---

## 2. System context

```
                          ┌──────────────────────────┐
                          │      User / Browser      │
                          └────────────┬─────────────┘
                                       │  HTTP
                     ┌─────────────────┴─────────────────┐
                     ▼                                   ▼
          ┌────────────────────┐              ┌──────────────────┐
          │  React + Vite SPA   │              │     FastAPI      │
          │  :5173              │───── /api ──▶│     :8000        │
          └────────────────────┘              └────────┬─────────┘
                                                      │
                            ┌─────────────────────────┴──────────────────┐
                            │                                            │
                   ┌────────▼────────┐                      ┌────────────▼───────────┐
                   │  Ingest path    │                      │     Query path         │
                   │  (write)        │                      │     (read)             │
                   └────────┬────────┘                      └────────────┬───────────┘
                            │                                            │
              ┌─────────────┼─────────────┐                  ┌─────────────┼─────────────┐
              ▼             ▼             ▼                  ▼             ▼             ▼
        OpenCV        SigLIP-2       FFmpeg            SigLIP-2      pgvector       Graph
      keyframe      image tower     (clips)            text tower       ANN         expansion
      extraction                          │                              │             │
              └─────────────┬─────────────┘                              └──────┬──────┘
                            ▼                                                   ▼
                   ┌─────────────────┐                              ┌──────────────────┐
                   │  Filesystem     │                              │  4-term re-rank  │
                   │ media/frames/   │                              │  + evidence      │
                   │ clips           │                              └──────────────────┘
                   └─────────────────┘
                                        both meet at
                    ┌───────────────────────────────────┐
                    │  PostgreSQL 16 + pgvector 0.6.0   │
                    │  videos · video_frames ·          │
                    │  graph_nodes · graph_edges        │
                    └───────────────────────────────────┘
```

**Four bounded external dependencies:** PostgreSQL+pgvector (required, no
fallback), the SigLIP-2 checkpoint (~1.5 GB, one-time download), FFmpeg (clip
generation only), and a CUDA GPU (optional — CPU works but is much slower).

---

## 3. The three flows

### 3.1 Upload / indexing (write)

```
POST /api/v1/videos (multipart)
  ├─ validate extension {mp4,mov,mkv,webm,avi}
  ├─ save to MEDIA_ROOT/<uuid><ext>, enforce MAX_UPLOAD_MB
  ├─ OpenCV probe → duration, fps, frame_count
  ├─ INSERT videos (status='queued'); commit
  └─ BackgroundTasks → process_video_background   [own session]

process_video_background
  ├─ status='processing'; commit
  ├─ extract_adaptive_keyframes(path)
  │     decode every frame → grayscale → resize 64×36
  │     mean-abs-difference vs last RETAINED frame, normalised by 255
  │     keep if  (diff ≥ FRAME_DIFF_THRESHOLD and Δt ≥ MIN_FRAME_INTERVAL)
  │           or  Δt ≥ MAX_FRAME_INTERVAL          ← forced keep
  ├─ (no frames?) → status='failed'
  ├─ process_and_store_embeddings(...)
  │     per batch of BATCH_SIZE: write 320×180 q85 thumbnail,
  │     encode_images(), INSERT video_frames, commit
  ├─ build_graph_for_video(...)                    ← the graph layer
  │     VIDEO node
  │     FRAME node per retained keyframe (frame_id FK)
  │     SCENE nodes by timestamp clustering
  │     CONTAINS / BELONGS_TO    (deterministic containment)
  │     FOLLOWED_BY / PRECEDES   (real timestamp order)
  │     SIMILAR_TO               (measured cosine ≥ 0.80)
  │     extractor output, or nothing (NullGraphExtractor)
  │     ONE transaction for the whole graph
  └─ backfill duration/fps/frame_count, status='ready'
     on ANY exception → rollback, status='failed', error_message
```

The whole graph is written in one transaction, so a partial graph is impossible.
`ON DELETE CASCADE` makes orphaned edges structurally impossible too, not merely
unlikely.

**Measured on the indexed video:** 3108 decoded frames → **38 retained
keyframes** (`98.78%` reduction), 40 graph nodes, 270 edges.

### 3.2 Search, baseline (frozen)

```
POST /api/v1/search  { query, video_id?, top_k, min_score }
  1. encode_text(query)                       → 1152-d L2-normalised
  2. SELECT … 1 - (f.embedding <=> CAST(:emb AS vector)) AS score
       WHERE v.status='ready' AND score >= :min_score
       ORDER BY f.embedding <=> CAST(:emb AS vector) ASC
       LIMIT top_k * 3                        ← over-fetch for grouping
  3. _temporal_group(results, TEMPORAL_GROUP_WINDOW)
       sort by timestamp, group where gap ≤ window, keep best per group
  4. truncate to top_k
```

This route is the ablation's control condition. It is deliberately not
"improved".

### 3.3 Search, graph-aware

```
POST /api/v1/search/graph  { query, video_id?, top_k, graph_depth }
  1. encode_text(query)                              [timed: embedding]
  2. same pgvector candidate query,
     LIMIT = top_k * GRAPH_CANDIDATE_MULTIPLIER     [timed: vector_retrieval]
  3. expand_from_seeds(seeds, max_depth = min(graph_depth, GRAPH_MAX_DEPTH),
                                 max_nodes = GRAPH_MAX_NODES)
       level 0 : SELECT nodes WHERE frame_id IN (…)            — 1 query
       level d : SELECT edges touching the frontier             — 1 query
                 SELECT the discovered nodes                    — 1 query
       hard stop at BOTH caps                        [timed: graph_expansion]
  4. rank_candidates(…)                               [timed: graph_reranking]
  5. response carries the full breakdown + evidence chain
```

Same SQL shape as the baseline, differing only in `LIMIT` and an
`embedding IS NOT NULL` guard — so any difference in output is attributable to
the graph rather than to a different candidate generator.

`graph_depth = 0` disables expansion and reproduces the vector-only ordering
through the *same* code path. That is what makes condition A a control rather
than a separate implementation.

---

## 4. Data model

### 4.1 Relational schema

```
videos ──1:N──▶ video_frames          (embedding vector(1152))
                    │                          ▲
                    │ frame_id FK              │ source_node_id / target_node_id
                    ▼                          │  ON DELETE CASCADE
              graph_nodes ──────────────────────┤
                    ▲                          │
                    └──────── graph_edges ──────┘
```

Deleting a `videos` row cascades to `video_frames` → `graph_nodes` → every
`graph_edges` row touching those nodes. Orphaned edges are impossible by schema,
not by convention.

### 4.2 Graph schema

```
VIDEO ──CONTAINS──▶ SCENE ──CONTAINS──▶ FRAME
FRAME ──BELONGS_TO──▶ SCENE
FRAME ──FOLLOWED_BY / PRECEDES──▶ FRAME        (timestamp ordering)
FRAME ──SIMILAR_TO──▶ FRAME                     (measured cosine ≥ 0.80)
FRAME ──DEPICTS──▶ CONCEPT / OBJECT / TEXT      ⬜ EMPTY — see §7
```

Every edge stores a **`basis`** in its metadata recording *why* it exists:
`deterministic_containment`, `timestamp_ordering`, or
`measured_cosine_similarity`. No code path invents an edge.

### 4.3 Vector index

```sql
CREATE INDEX idx_video_frames_embedding
    ON video_frames USING hnsw (embedding vector_cosine_ops);
```

Because embeddings are L2-normalised, inner product equals cosine similarity,
which is what `vector_cosine_ops` assumes.

---

## 5. Scoring

```
FinalScore = ( α·Vector + β·Graph + γ·Temporal + δ·Confidence )
             ──────────────────────────────────────────────────
                            α + β + γ + δ
```

| Term | Weight | Default | What it measures |
|---|---|---|---|
| α `VECTOR_WEIGHT` | vector | 0.65 | `clip01(1 − cosine_distance)` |
| β `GRAPH_WEIGHT` | graph | 0.20 | 4-term convex combination (below) |
| γ `TEMPORAL_WEIGHT` | temporal | 0.10 | density of the timeline neighbourhood |
| δ `CONFIDENCE_WEIGHT` | confidence | 0.05 | geometric mean of node confidences |

### GraphScore

```
                    a_d·DirectSimilarity + a_s·SceneOverlap
GraphScore =  ─────────────────────────────────────────────────────────────
                    + a_c·EdgeConfidence   + a_t·TemporalAdjacency
                             a_d + a_s + a_c + a_t
```

| Sub-score | Coeff | Definition |
|---|---|---|
| `DirectSimilarity` | `a_d` 0.40 | strongest measured `SIMILAR_TO` weight from the candidate to any seed |
| `SceneOverlap` | `a_s` 0.25 | `min(1, |SCENE(F) ∩ SCENE(S)| / max(1,|SCENE(S)|))` |
| `EdgeConfidence` | `a_c` 0.20 | mean confidence over traversed edges |
| `TemporalAdjacency` | `a_t` 0.15 | `max 1/(1 + gap/W)` over real temporal edges |

Each sub-score is traceable to an edge the bounded traversal actually traversed,
so no term can exceed its evidence.

### TemporalScore

```
TemporalScore(r) = max(0, 1 − min_{r'≠r} |t(r) − t(r')| / W)
```

Isolated spikes score low; a contiguous run of relevant frames scores high.
The `r' ≠ r` exclusion matters: without it every candidate scores exactly 1.0
and the term is a useless constant. (This was one of four bugs the test suite
caught — see `Readme.md`.)

### Tie-breaking

```
key = (−final_score, −vector_score, timestamp_seconds, frame_id)
```

Total and deterministic, so two runs over identical data produce byte-identical
orderings. That is what makes the ablation reproducible.

---

## 6. Bounded traversal — the load-bearing invariant

Three guarantees, each covered by tests:

| Guarantee | How |
|---|---|
| **No unbounded traversal** | `GRAPH_MAX_DEPTH` and `GRAPH_MAX_NODES` are hard stops enforced in code, not delegated to the database |
| **No N+1 queries** | breadth-first over level-sized batches: `1 + 2 × depth` round-trips regardless of graph size |
| **No partial state** | one transaction per video's graph; failure rolls back and sets `status='failed'` |

Measured on the indexed video at depth 2: 40 nodes reached, 258 edges traversed,
with `graph_truncated_depth = true` — the depth cap was actually hit and
enforced, so the cap is demonstrably live rather than decorative.

---

## 7. Implemented vs. intended

This separation is the point of the project, so it is stated explicitly.

### ✅ Implemented and verified running

* Adaptive keyframe extraction with a structural-difference heuristic
* SigLIP-2 image **and** text towers → 1152-d L2-normalised embeddings
* pgvector HNSW cosine search
* Temporal grouping and de-duplication
* Relational graph: schema, construction, bounded traversal
* 4-term transparent re-ranking with a full score breakdown
* "Why this result?" evidence chains (related nodes + concrete edge paths)
* Alembic additive migration (`0001_graph_layer`)
* Evaluation harness: 6 IR metric families, 4 ablation conditions, depth sweep,
  latency, indexing, storage — with genuine `N/A` semantics
* 8 figure scripts, all rendering from CSV
* React frontend with vector/graph mode toggle and graph depth selector
* Docker Compose stack (Postgres+pgvector, backend, frontend)

### ⬜ Intended but empty in the shipped configuration

```
     Semantic Relations ──┐
     Temporal Relations ──┼──▶ Graph Re-ranking  ← implemented, but…
     CONCEPT Relations ──┘        │
            ▲                     ▼
            └── EMPTY       only similarity + temporal + scene structure
```

`graph_extraction_service.py` defines a `GraphExtractor` **Protocol** and ships
a `NullGraphExtractor` that returns `[]`. There is **no detector, no OCR engine,
no ASR model and no captioner** in this repository, so `DEPICTS` edges are never
written and `CONCEPT`/`OBJECT`/`TEXT` node types stay empty.

Consequences, all confirmed by measurement rather than assumption:

* The **ConfidenceScore term is identically 1.0** for every candidate — verified:
  all 40 nodes have `confidence = 1.0`. It is ranking-neutral, so the ablation
  deliberately sets its weight to `0` rather than letting a constant masquerade
  as signal.
* An unknown `GRAPH_EXTRACTOR` value is a **hard error**, never a silent
  fallback to `"null"`.
* `GRAPH_ENABLED=false` restores byte-identical pre-graph behaviour.

Populating this branch with invented labels would have been the easiest way to
fabricate a richer-looking result. The code is structured specifically to make
that impossible.

---

## 8. Technology stack

### 8.1 Backend — Python 3.12

| Layer | Choice | Version installed | Role |
|---|---|---|---|
| API framework | **FastAPI** | 0.115.0 | async routes, auto OpenAPI at `/docs` |
| ASGI server | **Uvicorn** | 0.30.6 | serves the app; `--reload` for development |
| Data validation | **Pydantic** | 2.9.2 | request/response models |
| Settings | **pydantic-settings** | 2.5.2 | all 32 settings from `.env` |
| ORM | **SQLAlchemy** | 2.0.35 | models, sessions, `text()` for pgvector SQL |
| DB driver | **psycopg2-binary** | 2.9.9 | PostgreSQL wire protocol |
| Vector types | **pgvector** | 0.3.4 | `Vector` column type for SQLAlchemy |
| Migrations | **Alembic** | 1.13.3 | versioned, additive schema evolution |
| Uploads | **python-multipart** | 0.0.9 | multipart form parsing |
| Embeddings | **PyTorch** | 2.11.0+cu128 | tensor ops, CUDA |
| Model | **Transformers** | 5.17.0 | `AutoModel` / `AutoProcessor` for SigLIP-2 |
| Imaging | **Pillow** | 10.4.0 | frame → PIL for the image tower |
| Video decode | **opencv-python-headless** | 4.10.0.84 | frame decode + metadata probe |
| Numerics | **NumPy** | 2.5.4 | array math (⚠ see below) |
| Plotting | **Matplotlib** | 3.9.2 | the 8 figures |
| Tables | **pandas** | 2.2.3 | CSV reading for plots |
| Testing | **pytest**, **pytest-asyncio** | 8.3.3 / 0.24.0 | 178 tests |
| HTTP client | **httpx** | 0.27.2 | test client |

⚠ `requirements.txt` pins `numpy==1.26.4` but **2.5.4 is installed**. The suite
passes on the installed version, but the pin and the environment disagree. Worth
reconciling deliberately rather than leaving to drift.

### 8.2 Embedding model

| Property | Value |
|---|---|
| Checkpoint | `google/siglip2-so400m-patch14-384` |
| Architecture | dual-tower contrastive image–text |
| Output | 1152-d, L2-normalised (inner product = cosine) |
| Input | 384×384, patch 14 |
| Cache | `HF_HOME=./data/models` (~1.5 GB) |
| Loader | `AutoProcessor` + `AutoModel`, thread-safe singleton |

**API compatibility note.** `transformers` 5.x returns
`BaseModelOutputWithPooling` from `get_*_features()`, where 4.x returned a bare
tensor. `model_service._as_embedding_tensor()` accepts both. Since
`requirements.txt` has no upper bound on `transformers`, this is a real
portability hazard, not a hypothetical — see `MISSING_FILES.md` §2.

### 8.3 Database — PostgreSQL 16 + pgvector

| Component | Version | Note |
|---|---|---|
| PostgreSQL | 16.15 | running inside **WSL2** on port **5433** |
| pgvector | 0.6.0 | HNSW available (≥ 0.5) |
| Tables | `videos`, `video_frames`, `graph_nodes`, `graph_edges` | + `alembic_version` |

**Why WSL2.** The Windows-side PostgreSQL 18 could not be given pgvector — no C
compiler, and `share\extension` is not writable without administrator rights.
WSL2 had root and network access, so `postgresql-16-pgvector` installed cleanly.
Trade-off: the WSL2 IP changes on reboot and the VM suspends when idle, so
`.env` needs updating after a reboot. **The Docker path remains the supported
deployment route**; no code was changed to accommodate this.

### 8.4 Frontend — React 18 + Vite 5

| Choice | Version | Role |
|---|---|---|
| React | 18.3.1 | UI |
| Vite | 5.4.3 | dev server + production build |
| TypeScript | 5.5.4 | typed API boundary and components |
| Tailwind CSS | 3.4.10 | utility styling |
| axios | 1.7.7 | HTTP client |
| lucide-react | 0.441.0 | icons |
| PostCSS / autoprefixer | 8.4.45 / 10.4.20 | Tailwind pipeline |

### 8.5 External binaries

| Tool | Version | Required for |
|---|---|---|
| **FFmpeg** | 9.0.2 | 5-second clip generation. **Not** needed for indexing or search. |
| Node.js | 24.15.0 | frontend build (≥ 18 required) |
| Docker Desktop | not installed | the documented deployment path |

### 8.6 Hardware actually used

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop, **4.0 GB VRAM** |
| CUDA | 12.8 |
| torch device | `cuda` — `torch.cuda.is_available() == True` |

4 GB is tight for a 1152-d, 888-weight-block model at fp32. It works, but the
first query pays a ~25 s model-load cost; subsequent queries run at ~130–160 ms.
Enabling the commented-out warm-up in `main.py` would remove that from the first
user request.

---

## 9. Repository map

```
VideoCtrlF/
├── backend/
│   ├── app/
│   │   ├── main.py                    FastAPI app, CORS, static mounts, /health
│   │   ├── core/{config,logging}.py   32 settings; logging
│   │   ├── db/{session.py,init.sql}   engine + bootstrap schema
│   │   ├── models/{video,frame,graph}.py
│   │   ├── schemas/{search,video,graph}.py
│   │   ├── controllers/               thin HTTP error mapping
│   │   ├── services/
│   │   │   ├── model_service.py       SigLIP-2 singleton (image + text)
│   │   │   ├── embedding_service.py   adaptive keyframes, thumbnails
│   │   │   ├── video_service.py       upload, background indexing
│   │   │   ├── search_service.py      ⚠ FROZEN baseline (cast bug fixed)
│   │   │   ├── clip_service.py        FFmpeg clip generation
│   │   │   ├── graph_extraction_service.py  Protocol + NullGraphExtractor
│   │   │   ├── graph_service.py       construction + bounded traversal
│   │   │   ├── graph_ranking_service.py      4-term scoring + ablations
│   │   │   └── graph_search_service.py       graph-aware pipeline
│   │   └── api/v1/routers/{videos,search,graph}.py
│   ├── alembic/versions/0001_graph_layer.py
│   └── tests/                         178 tests
├── frontend/src/{api.ts,App.tsx,hooks/,components/}
├── experiments/
│   ├── datasets/                      evaluation.csv + template
│   ├── evaluation/                    metrics, ranking, reporting, CLIs
│   ├── visualization/                 8 figure scripts
│   └── results/                       generated CSVs, SUMMARY.md, figures/
├── scripts/                           environment bootstrap helpers
├── docs/{AUDIT_REPORT,EVALUATION}.md
├── RUN.ps1                            one entry point: doctor/db/init/serve/
│                                      test/index/eval/plots/results
├── MISSING_FILES.md                   gaps + the 6 defects found and fixed
├── ARCHITECTURE.md                    this file
├── AGENT.md                           agent-facing runbook
└── Readme.md                          long-form project documentation
```

---

## 10. API surface

| Route | Purpose |
|---|---|
| `POST /api/v1/videos` | upload (triggers background indexing) |
| `GET /api/v1/videos` | list library |
| `GET /api/v1/videos/{id}` | single video + status |
| `POST /api/v1/search` | **baseline**, vector-only (frozen) |
| `POST /api/v1/search/graph` | **graph-aware** search + score breakdown |
| `GET /api/v1/videos/{id}/graph` | full video graph |
| `POST /api/v1/search/clip/{frame_id}` | 5 s FFmpeg clip centred on a timestamp |
| `GET /health` | liveness |
| `GET /docs` | OpenAPI UI |

Verified live in this session: all 7 original + 2 graph routes respond, graph
serialisation returns 40 nodes / 270 edges, the score breakdown's `contributions`
sum exactly to `final_score`, clip generation produced a 264 KB MP4, and
thumbnails serve correctly.

---

## 11. Verification summary

| Check | Result |
|---|---|
| Backend test suite | **178 passed** (`VIDEOCTRLF_RUN_DB_TESTS=1` too) |
| Frontend production build | **succeeds** (`tsc -b && vite build`) |
| Database schema | `videos`, `video_frames`, `graph_nodes`, `graph_edges`; HNSW index present; alembic at `0001_graph_layer` |
| End-to-end smoke test | **all checks passed** — every route exercised against real data |
| `RUN.ps1` (all 11 tasks) | **each task verified** — status, doctor, db, init, serve, test, index, eval, plots, results |
| Indexing | 3108 → 38 frames (98.78% reduction), 40 nodes, 270 edges |
| Evaluation | 12 queries, 12 scorable, all metric families measured |
| Figures | 8/8 rendered from CSV; `model_comparison.png` correctly captioned `N/A` |

Two findings that change how the results should be read — both detailed in
`MISSING_FILES.md`:

1. **Graph re-ranking did not alter the ranking on this corpus.** All four
   ablation conditions and all four graph depths produce identical orderings,
   because `graph_score` and `temporal_score` are constants across all
   candidates (one SCENE node spans the whole video; all frames are near
   duplicates of the same UI). Verified by instrumenting per-candidate scores.
2. **Absolute retrieval quality is below the random baseline** (Recall@10 0.104
   vs ≈ 0.26 for random over 38 frames). The corpus is a single screen recording
   of one dark-themed application, whose distinguishing evidence is a few pixels
   tall in a 320×180 thumbnail.

**No claim of retrieval improvement is supported by the present data, and none is
made.** What the data supports is that the architecture is implemented,
testable, reproducible, and honestly instrumented — including the parts that did
not work.

---

## 12. Team

Vignesh K (23AM069) · Yogaazhaki S (23AM072) · Deepika S (23AM016)
Project duration: 8 months
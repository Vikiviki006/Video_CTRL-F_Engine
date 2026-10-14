# VideoCtrl-F

**An adaptive keyframe and SigLIP-2 based semantic video retrieval framework,
extended with a relational knowledge-graph layer and graph-aware re-ranking.**

A multimodal video search engine: upload a video, it is decoded adaptively into
keyframes, each keyframe is embedded with SigLIP-2, and those embeddings are
indexed in PostgreSQL + pgvector for fast cosine-similarity search. A text
query is embedded in the same space, and the closest frames are returned as
timestamps and clips. On top of that, each video also gets a **relational
graph** — scenes, frames, and measured frame-to-frame similarity — which is used
to re-rank results and to explain why each result was returned.

---

## Table of contents

1. [Why plain vector search is not enough](#1-why-plain-vector-search-is-not-enough)
2. [The proposed solution](#2-the-proposed-solution)
3. [Architecture](#3-architecture)
4. [Repository structure](#4-repository-structure)
5. [File-by-file documentation](#5-file-by-file-documentation)
6. [The three flows, with worked examples](#6-the-three-flows)
7. [Scoring, in full](#7-scoring-in-full)
8. [Evaluation](#8-evaluation)
9. [Visualization](#9-visualization)
10. [Proposed contribution](#10-proposed-contribution)
11. [Configuration reference](#11-configuration-reference)
12. [Running it](#12-running-it)
13. [Known limitations and current status](#13-known-limitations-and-current-status)

---

## 1. Why plain vector search is not enough

A pure vector index answers one question well: *which stored frame is
semantically closest to this text?* SigLIP-2 maps text and images into a shared
space where cosine similarity is meaningful, and pgvector's HNSW index makes
that nearest-neighbour search fast. For "find me a frame containing a
whiteboard", it works well.

It has three specific weaknesses, and this project addresses each:

**1. A frame is scored in isolation.** Nothing in a bare `ORDER BY embedding
<=> query` knows that a frame two seconds away shows the same whiteboard from a
different angle, or that a frame thirty seconds away shows a completely
unrelated shot. Semantic similarity and temporal structure are separate signals,
and only the first one is used.

**2. Near-duplicate results crowd out diversity.** A slide that stays on screen
for ten seconds produces many near-identical keyframes, all of which score
almost identically. They occupy every top-K slot while genuinely different but
slightly lower-scoring moments are pushed out. A ranking that also knows about
time can prefer *distinct* relevant moments over *repeats of the same one*.

**3. A result is a bare number.** A cosine score of `0.31` is not an argument.
It gives the user no way to judge whether the result is trustworthy, and it
gives a developer no way to debug why it was ranked first. In a domain like
surveillance, lecture capture or broadcast archive search, "why this result?"
is usually the first question asked.

The relational graph layer is the response: a per-video graph of scenes, frames
and *measured* frame-to-frame similarity, traversed under hard bounds, feeding a
transparent four-term ranking that also reports its own reasoning.

---

## 2. The proposed solution

```
Video
  → adaptive keyframe extraction        (OpenCV, structural-difference heuristic)
  → SigLIP-2 image embedding            (1152-d, L2-normalised)
  → PostgreSQL + pgvector, HNSW index   (cosine)
  → PLUS: a per-video relational graph
        • VIDEO / SCENE / FRAME nodes
        • CONTAINS, BELONGS_TO           (deterministic containment)
        • FOLLOWED_BY, PRECEDES          (measured timestamp ordering)
        • SIMILAR_TO                     (measured cosine similarity ≥ threshold)
        • DEPICTS                        (only if a real extractor is plugged in)

Text query
  → SigLIP-2 text embedding
  → pgvector top-N candidate frames
  → bounded graph expansion              (max_depth AND max_nodes, batched)
  → weighted re-ranking with a full score breakdown
  → results + "why this result?" evidence
```

The design commitments that make this defensible:

| Commitment | How it is enforced |
|---|---|
| No fabricated relationships | Every edge records a `basis` in its metadata. `SIMILAR_TO` weights are measured cosine similarities; `FOLLOWED_BY` comes from real timestamp ordering. There is no code path that invents an edge. |
| No unbounded traversal | `GRAPH_MAX_DEPTH` and `GRAPH_MAX_NODES` are hard stops, enforced in code rather than delegated to the database, and covered by tests. |
| No N+1 queries | Traversal is breadth-first over level-sized batches: `1 + 2 × depth` round-trips regardless of graph size. |
| No partial state on failure | A video's whole graph is written in one transaction. Failure rolls back and sets `video.status = failed`. Orphaned edges are structurally impossible via `ON DELETE CASCADE`. |
| Backward compatibility | `POST /api/v1/search` and every other pre-existing endpoint are byte-for-byte unchanged. Graph mode is a **new route**. |
| Honest absence | With no detector available, the graph has no concept layer, and the `ConfidenceScore` term is documented as ranking-neutral rather than quietly doing nothing. |

---

## 3. Architecture

### 3.1 Full combined architecture

```
                         USER
                          │
                          ▼
                  ┌───────────────┐
                  │ React Frontend│
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │    FastAPI    │
                  └───────┬───────┘
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
      Video Processing           Query Processing
             │                         │
             ▼                         ▼
      Adaptive Frames            SigLIP-2 Text
             │                         │
             ▼                         ▼
       SigLIP-2 Vision           Query Vector
             │                         │
             ▼                         ▼
       Frame Embeddings          pgvector Search
             │                         │
             └───────────┬─────────────┘
                         ▼
                Top-K Candidate Frames
                         │
                         ▼
                Relational Graph (bounded expansion)
                         │
              ┌──────────┼───────────┐
              ▼          ▼           ▼
           Semantic   Temporal     Concept
           Relations  Relations    Relations
              │          │           │
              └──────────┼───────────┘
                         ▼
                  Graph Re-ranking
                         │
                         ▼
                 Final Search Results
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
         Timestamp                Clip
              │                     │
              └──────────┬──────────┘
                         ▼
                    React UI
```

> **Honest note on the "Concept Relations" branch.** That branch is drawn
> because it is the intended design, but **it is empty in the shipped
> configuration.** This repository contains no object detector, no OCR engine,
> no ASR model and no captioner. `graph_extraction_service.py` defines the
> `GraphExtractor` interface and ships a `NullGraphExtractor` that returns `[]`.
> `CONCEPT`, `OBJECT` and `TEXT` node types exist in the schema and stay empty
> until a real extractor is registered. See
> [§13](#13-known-limitations-and-current-status).

### 3.2 Indexing data flow (what is actually stored)

```
video ─┬─► videos                    (id, path, duration, fps, frame_count, status)
       │
       ├─► video_frames              (id, video_id, timestamp_seconds, frame_number,
       │                              thumbnail_path, embedding vector(1152))
       │        ▲
       │        │ frame_id FK, ON DELETE CASCADE
       │        │
       ├─► graph_nodes              (VIDEO, SCENE, FRAME per retained keyframe)
       │        ▲
       │        │ source_node_id / target_node_id FKs, ON DELETE CASCADE
       │        │
       └─► graph_edges              (CONTAINS, BELONGS_TO, FOLLOWED_BY,
                                     PRECEDES, SIMILAR_TO)
```

Deleting a `videos` row cascades to `video_frames`, which cascades to
`graph_nodes`, which cascades to every `graph_edges` row touching those nodes.
Orphaned edges are impossible by schema, not by convention.

---

## 4. Repository structure

```
VideoCtrlF/
├── backend/
│   ├── app/
│   │   ├── main.py                       FastAPI app, CORS, static mounts, /health
│   │   ├── core/
│   │   │   ├── config.py                 Settings (14 original + 18 graph settings)
│   │   │   └── logging.py
│   │   ├── db/
│   │   │   ├── session.py                engine, SessionLocal, Base, get_db
│   │   │   └── init.sql                  bootstrap schema for a FRESH volume
│   │   ├── models/
│   │   │   ├── video.py                  Video
│   │   │   ├── frame.py                  VideoFrame (embedding vector(1152))
│   │   │   └── graph.py                  GraphNode, GraphEdge, NodeType, RelationshipType
│   │   ├── schemas/
│   │   │   ├── search.py                 FROZEN - existing search contract
│   │   │   ├── video.py
│   │   │   └── graph.py                  NEW - graph request/response models
│   │   ├── controllers/                  thin HTTP error mapping
│   │   ├── services/
│   │   │   ├── model_service.py          SigLIP-2 singleton (images + text)
│   │   │   ├── embedding_service.py      adaptive keyframes, thumbnails, embedding
│   │   │   ├── video_service.py          upload, background indexing, clip pipeline
│   │   │   ├── search_service.py         FROZEN - the vector-only baseline
│   │   │   ├── clip_service.py           FFmpeg clip generation
│   │   │   ├── graph_extraction_service.py  GraphExtractor protocol + NullGraphExtractor
│   │   │   ├── graph_service.py          graph construction + bounded traversal
│   │   │   ├── graph_ranking_service.py  the four-term scoring formula
│   │   │   └── graph_search_service.py   the graph-aware retrieval pipeline
│   │   └── api/v1/
│   │       ├── router.py                 + one line: include the graph router
│   │       └── routers/
│   │           ├── videos.py             FROZEN
│   │           ├── search.py             FROZEN
│   │           └── graph.py              NEW - the two additive endpoints
│   ├── alembic/                          versioned migrations (new)
│   │   ├── env.py
│   │   └── versions/0001_graph_layer.py
│   ├── alembic.ini
│   └── tests/                            178 tests
├── frontend/
│   └── src/
│       ├── api.ts                        + graph client functions and types
│       ├── App.tsx                       + plain/graph mode toggle
│       ├── hooks/useVideoSearch.ts       FROZEN
│       ├── hooks/useGraphSearch.ts       NEW
│       └── components/
│           ├── ResultCard.tsx            FROZEN
│           ├── GraphResultCard.tsx       NEW - "Why this result?" panel
│           ├── GraphSearchResults.tsx    NEW
│           ├── SearchBar.tsx             + optional children slot
│           └── TimelineHeatmap.tsx       narrowed prop type (widening only)
├── experiments/
│   ├── datasets/                         annotation CSVs + README
│   ├── evaluation/                       metrics, ranking, reporting, CLIs
│   ├── visualization/                    the eight figure scripts
│   └── results/                          generated output
├── scripts/                              environment bootstrap helpers
├── docs/
│   ├── AUDIT_REPORT.md                   Phase 0 repository audit
│   └── EVALUATION.md                     every metric, defined mathematically
├── docker-compose.yml                    FROZEN
└── Makefile
```

---

## 5. File-by-file documentation

Not a filename list — what each important file does, why it exists, what calls
it, and what it connects to.

### 5.1 Pre-existing files (reused as-is)

**`backend/app/services/model_service.py`** — the SigLIP-2 singleton. Wraps
`AutoProcessor` + `AutoModel` for `google/siglip2-so400m-patch14-384`, exposing
`encode_images(list[PIL])` and `encode_text(str | list[str])`. Both apply
`F.normalize(p=2, dim=-1)`, so outputs are 1152-d and L2-normalised — which is
what makes the inner product *equal* to cosine similarity and what pgvector's
`vector_cosine_ops` assumes. Called by `embedding_service` (indexing) and
`graph_search_service` (query). **Untouched.**

**`backend/app/services/embedding_service.py`** — the adaptive keyframe
heuristic and the embedding/insert loop. The heuristic: decode every frame,
grayscale, resize to 64×36, and compute the mean absolute difference against
the last *retained* frame normalised by 255; keep when
`(diff ≥ FRAME_DIFF_THRESHOLD and Δt ≥ MIN_FRAME_INTERVAL_SECONDS)` or
`Δt ≥ MAX_FRAME_INTERVAL_SECONDS`. Also writes 320×180 q85 thumbnails named
`{frame_number:08d}.jpg`. The graph layer consumes its output rather than
re-implementing it. **Untouched.**

**`backend/app/services/search_service.py`** — the vector-only baseline. Text
embeds, one pgvector query with the same `<=>` operator and filters as the graph
path, over-fetches `top_k × 3`, groups nearby timestamps in Python, keeps the
best frame per group, truncates to `top_k`. **Untouched, deliberately** — it is
the control condition in the ablation, so changing it would change the baseline.

**`backend/app/schemas/search.py`** — `SearchRequest` / `SearchResult` /
`SearchResponse`. **Untouched.** The graph response uses *different* models in
`schemas/graph.py`, because widening these would change the response for a
request that did not ask for a graph.

**`backend/app/services/video_service.py`** — one insertion point changed: after
`process_and_store_embeddings` commits, `process_video_background` now calls
`graph_service.build_graph_for_video(...)` and commits. The builder only
adds and flushes, so the graph lands in the background task's transaction; a
failure rolls back and re-raises, and the pre-existing handler sets
`status="failed"`. `GRAPH_ENABLED=false` restores byte-identical original
behaviour.

**`backend/app/api/v1/router.py`** — one appended line,
`api_router.include_router(graph.router, tags=["Graph"])`. Existing includes
keep their original order, so the pre-existing route table is unchanged.

**`backend/app/core/config.py`** - 18 fields appended after the original 14. No
original field was renamed, retyped, or given a new default.

### 5.2 New backend files

**`backend/app/models/graph.py`** — `GraphNode`, `GraphEdge`, and the
`NodeType` / `RelationshipType` vocabularies. `node_type` and `relationship` are
`String` + `CHECK` constraint rather than native PG `ENUM`, mirroring the
existing `videos.status VARCHAR(32)` + Pydantic-enum style and keeping the
vocabulary extensible. The JSONB column is literally named `metadata` on disk
but exposed as `node_metadata` / `edge_metadata` in Python, because `metadata`
is reserved on declarative classes. `UNIQUE (video_id, node_type, entity_key)`
makes node construction idempotent, so a rebuild converges instead of
duplicating. Also defines the `SEMANTIC` / `TEMPORAL` / `STRUCTURAL`
relationship families the ranking service relies on.

**`backend/app/services/graph_extraction_service.py`** — the plug-in seam.
`Detection` (dataclass), `GraphExtractor` (Protocol), `NullGraphExtractor`
(default, returns `[]`), `register_extractor`, `get_graph_extractor`, and
`cosine_similarity_matrix` — the single definition of "similarity" in the
codebase, used for `SIMILAR_TO` edges. An unknown `GRAPH_EXTRACTOR` is a hard
error, never a silent fallback. `extract_for_frames` contains a per-frame
extractor failure so one bad frame cannot abort indexing.

**`backend/app/services/graph_service.py`** — construction and traversal.
`build_graph_for_video` creates the VIDEO node, one FRAME node per retained
keyframe, SCENE nodes from timestamp clustering, then structural, temporal and
thresholded-similarity edges, then whatever a real extractor reported. It never
commits. `expand_from_seeds` is the bounded traversal: level-sized batches,
`1 + 2 × depth` queries, hard stops at both caps, recording a `reached_by`
evidence chain per node. `serialize_graph` returns a whole video's graph in two
queries with a response-size guard.

**`backend/app/services/graph_ranking_service.py`** — the scoring formula, with
the full derivation in the module docstring. `RankingConfig` is a frozen
dataclass, so the four ablation conditions are values of the *same* code path
production uses. `ablation_configs()` returns A/B/C/D. `rank_candidates`
produces `ScoredCandidate` objects carrying all four sub-scores, the additive
`contributions` breakdown, `related_nodes`, and `relationships`.

**`backend/app/services/graph_search_service.py`** — the retrieval pipeline. The
candidate SQL is the same shape as the baseline's, differing only in the LIMIT
and the `embedding IS NOT NULL` guard, so any difference in output is
attributable to the graph rather than to a different candidate generator. Times
each stage with `perf_counter` and returns the per-stage breakdown, which is
where the Phase 3 latency numbers come from.

**`backend/app/schemas/graph.py`**, **`controllers/graph_controller.py`**,
**`api/v1/routers/graph.py`** — the two additive endpoints.

**`backend/alembic/`** — the migration mechanism. `0001_graph_layer.py` creates
only `graph_nodes` and `graph_edges`; `downgrade()` drops only those. Neither
`videos` nor `video_frames` is touched, so the migration is additive and
forward-only. Verified with `alembic upgrade head --sql`.

### 5.3 Frontend

**`src/api.ts`** — the whole API boundary. `search`, `SearchResult` and
`SearchResponse` are unchanged. Appended: `RelatedNode`,
`RelationshipEvidence`, `GraphSearchResult`, `GraphSearchResponse`, `GraphNode`,
`GraphEdge`, `VideoGraphResponse`, and the `searchGraph()` / `getVideoGraph()`
functions.

**`src/components/GraphResultCard.tsx`** — renders a graph result: the
component scores, a stacked bar of the additive `contributions`, and a
collapsible "Why this result?" panel listing the related nodes and the concrete
edge chain, including each edge's recorded `basis` translated into English
("measured cosine similarity between frame embeddings"). The bar is a rendering
of the API's own `contributions`, which sum back to `final_score` — the browser
does not form a second opinion.

**`src/components/GraphSearchResults.tsx`** — the list plus diagnostics. When
`graph_available` is false it says so explicitly, because "no graph content for
this video" and "a graph run that found nothing" are different situations.

**`src/hooks/useGraphSearch.ts`** — parallel to `useVideoSearch`, same shape, so
switching modes is one line in `App.tsx`.

**`src/App.tsx`** — a two-button mode toggle and a graph-depth selector
(0–3, where 0 is the evaluation baseline). Both search paths remain available
at all times, which makes the effect of graph re-ranking directly observable.

**`src/vite-env.d.ts`** — added. It was missing, which made `npm run build`
fail on `import.meta.env` with TS2339 *before* any graph work existed. Standard
Vite scaffold file.

---

## 6. The three flows

### 6.1 Upload / indexing flow

```
  1. POST /api/v1/videos  (multipart)
       └─► validate extension {mp4,mov,mkv,webm,avi}
       └─► save to MEDIA_ROOT/<uuid><ext>, check MAX_UPLOAD_MB and non-emptiness
       └─► get_video_metadata() via OpenCV  -> duration, fps, frame_count
       └─► INSERT videos (status='queued');  commit
       └─► BackgroundTasks.add_task(process_video_background, id)

  2. process_video_background            [own SessionLocal]
       ├─► status='processing';  commit
       ├─► extract_adaptive_keyframes(path)
       │       └─► decode every frame, 64x36 grayscale mean-abs-diff,
       │           keep on threshold or max-interval
       ├─► (no frames?)  status='failed', 'No frames could be extracted'
       ├─► process_and_store_embeddings(...)
       │       └─► per batch of BATCH_SIZE: save 320x180 q85 thumbnail,
       │           encode_images(), INSERT video_frames, commit
       ├─► build_graph_for_video(db, video_id)          [NEW]
       │       ├─► VIDEO node
       │       ├─► FRAME node per retained keyframe (frame_id FK)
       │       ├─► SCENE nodes by timestamp clustering + CONTAINS / BELONGS_TO
       │       ├─► FOLLOWED_BY / PRECEDES from real timestamp order
       │       ├─► SIMILAR_TO where measured cosine >= 0.80
       │       ├─► extractor output, or nothing (NullGraphExtractor)
       │       └─► commit            <-- ONE transaction for the whole graph
       ├─► backfill duration/fps/frame_count, status='ready', error=NULL
       └─► on ANY exception: rollback, status='failed', error_message
```

### 6.2 Search flow (unchanged, the baseline)

```
  1. POST /api/v1/search  { query, video_id?, top_k, min_score }
  2. encode_text(query) -> 1152-d L2-normalised vector
  3. SELECT ... 1 - (f.embedding <=> :emb::vector) AS score
       FROM video_frames f JOIN videos v ON v.id = f.video_id
       WHERE v.status='ready' AND score >= :min_score
       ORDER BY f.embedding <=> :emb::vector ASC
       LIMIT top_k * 3                       <-- over-fetch for grouping
  4. _temporal_group(results, TEMPORAL_GROUP_WINDOW)
       sort by timestamp, group where gap <= window,
       keep the highest-scoring frame per group
  5. truncate to top_k
  6. { query, results, total, latency_ms }
  7. POST /api/v1/search/clip/{frame_id} -> 5s FFmpeg clip centred on the timestamp
```

### 6.3 Graph flow (new, opt-in via a new route)

```
  1. POST /api/v1/search/graph  { query, video_id?, top_k, graph_depth }
  2. encode_text(query)                         [timed: embedding]
  3. same pgvector candidate query, LIMIT = top_k * GRAPH_CANDIDATE_MULTIPLIER
                                                 [timed: vector_retrieval]
  4. expand_from_seeds(seed_frame_ids,
                        max_depth = min(graph_depth, GRAPH_MAX_DEPTH),
                        max_nodes = GRAPH_MAX_NODES)
       level 0 : SELECT nodes WHERE frame_id IN (...)      -- 1 query
       level d : SELECT edges touching the frontier        -- 1 query
                 SELECT the discovered nodes                -- 1 query
       stop at max_depth AND at max_nodes
                                                 [timed: graph_expansion]
  5. rank_candidates(...)
       vector_score     = clip01(1 - cosine_distance)
       graph_score      = 4-term convex combination (see section 7)
       temporal_score   = max(0, 1 - nearest_gap / W)
       confidence_score = geometric mean of node confidences
       final_score      = weighted sum / weight sum
       sort by (-final_score, -vector_score, timestamp, frame_id)
                                                 [timed: graph_reranking]
  6. response carries the full breakdown + evidence chain
```

**`graph_depth = 0` disables expansion entirely and reproduces the vector-only
ordering** — that is the ablation's control condition, reached through the same
code path rather than a separate implementation.

### 6.4 Worked example — what a result's evidence actually looks like

For a frame at `t = 42.5 s` that pgvector ranked 3rd for
*"speaker drawing a diagram on a whiteboard"*:

```
Seed (vector candidate)          Frame @ 42.5s
     │
     ├── SIMILAR_TO ──────────►  Frame @ 44.1s      weight 0.91   basis: measured_cosine_similarity
     │                          (a different shot, near-identical content)
     ├── CONTAINS ◄──────────── SCENE @ 41.0–45.2s            basis: deterministic_containment
     ├── BELONGS_TO ──────────► SCENE @ 41.0–45.2s            basis: deterministic_containment
     ├── FOLLOWED_BY ─────────► Frame @ 44.1s      gap 1.6s    basis: timestamp_ordering
     └── PRECEDES ◄──────────── Frame @ 40.2s      gap 2.3s    basis: timestamp_ordering
```

The API returns exactly this as `related_nodes` and `relationships`, and the
"Concept Relations" column would carry `DEPICTS → CONCEPT "whiteboard"` **only
if a real extractor were registered**. It is not, so no `DEPICTS` edge exists
today, and the UI says so rather than inventing one.

---

## 7. Scoring, in full

```
FinalScore = ( α·VectorScore + β·GraphScore + γ·TemporalScore + δ·ConfidenceScore )
             ─────────────────────────────────────────────────────────────────────
                                  α + β + γ + δ
```

Defaults sum to 1.00, so the division is a no-op at the shipped configuration —
but it makes any non-negative weight set valid and keeps the score in `[0, 1]`.

| Weight | Env var | Default |
|---|---|---|
| α | `VECTOR_WEIGHT` | 0.65 |
| β | `GRAPH_WEIGHT` | 0.20 |
| γ | `TEMPORAL_WEIGHT` | 0.10 |
| δ | `CONFIDENCE_WEIGHT` | 0.05 |

### VectorScore

```
VectorScore = clip01( 1 − cosine_distance(query, frame) )
```

Identical to what `/search` ranks on, read from pgvector's `<=>`. Not
renormalised, so the baseline and the graph pipeline stay comparable. Clipped
at 0: a negative cosine contributes nothing rather than fighting the other terms.

### GraphScore

A convex combination of four sub-scores, each traceable to an edge the bounded
traversal actually traversed:

```
                        a_d · DirectSimilarity + a_s · SceneOverlap
GraphScore = ─────────────────────────────────────────────────────────────
                        + a_c · EdgeConfidence + a_t · TemporalAdjacency
                        a_d + a_s + a_c + a_t
```

| Coefficient | Env var | Default |
|---|---|---|
| `a_d` | `GRAPH_DIRECT_COEFF` | 0.40 |
| `a_s` | `GRAPH_SCENE_COEFF` | 0.25 |
| `a_c` | `GRAPH_EDGE_CONF_COEFF` | 0.20 |
| `a_t` | `GRAPH_TEMPORAL_COEFF` | 0.15 |

**`DirectSimilarity(F, S)`** — strongest measured semantic link from `F` to any
vector seed:

```
max over SIMILAR_TO edges (F ↔ G), G ∈ S, of edge.weight        (0 if none)
```

`edge.weight` is the measured cosine similarity stored at build time, so this
term can never exceed its evidence.

**`SceneOverlap(F, S)`** — is `F` inside a shot that a seed is also in?

```
min( 1, |SCENE(F) ∩ SCENE(S)| / max(1, |SCENE(S)|) )
```

**`EdgeConfidence(F)`** — how much the producing stage trusted the evidence that
reached `F`:

```
mean( edge.confidence for each traversed edge )                 (0 for a depth-0 seed)
```

**`TemporalAdjacency(F, S)`** — is `F` adjacent in time to a seed, *and* is
there a real temporal edge saying so?

```
max over temporal edges (F ↔ G), G ∈ S, of  1 / (1 + |t(F) − t(G)| / W)
                                                              (0 if none)
```

`1/(1 + d/W)` is monotone decreasing in the gap, equals 1 at zero gap and 0.5 at
exactly one window. **A chosen convention, not a fitted or optimal value.**

### TemporalScore

Independent of the graph — whether a result sits in a *dense region of the
timeline* rather than being an isolated spike:

```
TemporalScore(r) = max( 0, 1 − min_{r' ≠ r} |t(r) − t(r')| / W )
                 = 0  when r is the only candidate
```

Rationale: a contiguous run of relevant frames is more likely to be a real
coherent scene than one isolated frame that happens to score well. This is a
**design choice, stated as such, not a validated claim** — the ablation exists
precisely to measure whether it actually helps. Note `r' ≠ r`: the candidate
excludes itself, or every candidate would score 1.0 and the term would be a
useless constant.

### ConfidenceScore

```
ConfidenceScore(r) = geometric mean of node.confidence over r and its related nodes
```

Geometric, so one unconfident node drags the score down.

> **Honest limitation.** With the default `NullGraphExtractor` there is no
> extractor, every node is built deterministically with `confidence = 1.0`, and
> this term is identically 1.0 for every candidate. It is therefore
> **ranking-neutral in the shipped configuration**, and
> `ranking.confidence_term_diagnostic` reports that as a measured fact. The
> ablation deliberately sets `CONFIDENCE_WEIGHT = 0` for the same reason:
> including a constant would make the tables look like the term is contributing
> when it demonstrably is not.

### Tie-breaking

```
key = ( −final_score, −vector_score, timestamp_seconds, frame_id )
```

Total and deterministic: ties fall back to the raw vector score, then to the
earlier timestamp, then to the frame UUID. Two runs over the same data produce
byte-identical orderings, which is what makes the ablation reproducible.

---

## 8. Evaluation

Full reference with every formula and a worked example for each:
**[`docs/EVALUATION.md`](docs/EVALUATION.md)**.
Reproducing the experiments: **[`experiments/README.md`](experiments/README.md)**.

### 8.1 Metrics

| Metric | K values | Definition |
|---|---|---|
| Precision@K | 1, 5, 10, 20 | `\|relevant in top-K\| / K` — denominator is `K`, not `len(ranked)`, so short result lists are penalised |
| Recall@K | 1, 5, 10, 20 | `\|relevant in top-K\| / \|relevant\|` |
| HitRate@K | 1, 5, 10 | `1.0` if any top-K item is relevant, else `0.0` |
| MRR | – | `mean(1 / rank_of_first_relevant)` |
| MAP | – | `mean( (1/R) Σ P@i · relevant(i) )`, over the **full** ranked list |
| NDCG@K | 1, 5, 10, 20 | `DCG@K / IDCG@K`, **binary** relevance |

**Matching rule for single-ground-truth-timestamp datasets.** A retrieved frame
is relevant if its id matches a ground-truth id, *or* its timestamp is within
`tolerance_seconds` (default **2.0 s**, inclusive) of a ground-truth timestamp.
With exactly one annotated timestamp, at most one retrieved frame is relevant,
so Recall@K ∈ {0, 1} and Precision@K peaks at `1/K`. Recall@K is therefore the
metric to read on such a dataset; a low Precision@10 mostly means "the system
returned distractors", not "the system is bad".

**NDCG uses binary relevance** unless a dataset supplies real grades, because
none of this project's datasets carry grades and inventing them would fabricate
information. `ndcg_at_k` accepts a `grades` mapping for a future graded dataset.

**Percentiles** use linear interpolation between closest ranks (NumPy default),
`pos = (n−1)·p/100`. Stated explicitly because mixing conventions with
nearest-rank tools makes P95 values incomparable.

### 8.2 Indexing and storage metrics

```
Frame Reduction % = (1 − retained / original) × 100
```

`original` is `videos.frame_count` (every decoded frame), `retained` is the
number of rows in `video_frames`. Storage: raw video bytes, thumbnail bytes,
clip bytes, embedding bytes (`rows × 1152 × 4`, labelled as an arithmetic
estimate), total, and `derived / raw` overhead ratio.

### 8.3 Latency, per stage

`embedding`, `vector_retrieval`, `graph_expansion`, `graph_reranking`, `total` —
each with avg, median, P95, P99, min, max, all from real `time.perf_counter`
measurements taken in `graph_search_service.graph_search`.

### 8.4 Experiments

| Study | Conditions | What it isolates |
|---|---|---|
| **Ablation** | A vector only · B vector+temporal · C vector+graph · D vector+graph+temporal | Which signals actually contribute |
| **Graph depth** | 0, 1, 2, 3 | Whether deeper helps — plus nodes reached and latency at each depth. **No assumption is made that deeper is better**; the report states the measured direction |
| **Model comparison** | SigLIP vs SigLIP-2, identical frames/queries/ground truth/top-K/metrics | The backbone contribution |

Conditions are `RankingConfig` values from the production module, so the
published ablation necessarily describes the code that runs in production.

### 8.5 Current status — implemented vs. measured

| Metric family | Implemented | Measured |
|---|---|---|
| Recall@K, Precision@K, MRR, MAP, NDCG@K, HitRate@K | ✅ unit-tested | ❌ **N/A — no ground truth annotated** |
| Latency per stage | ✅ | ❌ **N/A — no indexed video** |
| Frame reduction | ✅ | ❌ **N/A — no indexed video** |
| Storage | ✅ | ❌ **N/A — no indexed video** |
| Graph expansion counts | ✅ | ❌ **N/A — no indexed video** |
| SigLIP vs SigLIP-2 | ✅ | ❌ **N/A — one checkpoint** |

**No metric value appears anywhere in this repository.** The formulas are real,
tested code; the numbers are not yet real, and the harness emits `N/A` with a
recorded reason rather than a plausible-looking figure. See
[§13](#13-known-limitations-and-current-status) for what unblocks them.

---

## 9. Visualization

Eight figures, each generated **only** from the CSV/JSON in
`experiments/results/`. No figure script contains a hard-coded metric, latency
or frame count. Run them all with `make plots`, or individually:

```bash
python -m experiments.visualization.plot_recall          # figures/recall_at_k.png
python -m experiments.visualization.plot_precision       # figures/precision_at_k.png
python -m experiments.visualization.plot_mrr             # figures/mrr.png
python -m experiments.visualization.plot_ndcg            # figures/ndcg_at_k.png
python -m experiments.visualization.plot_latency         # figures/latency_breakdown.png
python -m experiments.visualization.plot_frame_reduction # figures/frame_reduction.png
python -m experiments.visualization.plot_ablation        # figures/ablation.png
python -m experiments.visualization.plot_comparison      # figures/model_comparison.png
```

| # | Script | Input | Figure |
|---|---|---|---|
| 1 | `plot_recall.py` | `retrieval_metrics.csv` | Recall@K, vector vs vector+graph |
| 2 | `plot_precision.py` | `retrieval_metrics.csv` | Precision@K, vector vs vector+graph |
| 3 | `plot_mrr.py` | `retrieval_metrics.csv` | MRR per configuration |
| 4 | `plot_ndcg.py` | `retrieval_metrics.csv` | NDCG@K, vector vs vector+graph |
| 5 | `plot_latency.py` | `latency_results.csv` | Per-stage mean with P95 overlay |
| 6 | `plot_frame_reduction.py` | `indexing_results.csv` | original / candidate / retained + reduction % |
| 7 | `plot_ablation.py` | `ablation_results.csv` | A/B/C/D quality and latency, two panels |
| 8 | `plot_comparison.py` | `model_comparison.csv` | SigLIP vs SigLIP-2 |

**When there is nothing to plot, a figure is still written — captioned `N/A`
with the reason**, and its axes are cleared. An empty-looking chart is
indistinguishable from a real measurement of nothing, so the figure says so in
words instead. `N/A` is never rendered as `0`.

Every script accepts `--results-dir`, which is how
`backend/tests/test_reporting_and_plots.py` proves all eight render from real
CSV data without touching the real results directory.

---

## 10. Proposed contribution

> **VideoCtrl-F is an engineering integration and evaluation contribution.**
>
> The project combines adaptive keyframe extraction, SigLIP-2 embeddings,
> pgvector similarity search, temporal context, and relational-graph
> re-ranking into one working system, and evaluates that system experimentally
> against a vector-only baseline on identical data with identical metrics.
>
> **It does not claim that graph-based retrieval, knowledge graphs, semantic
> video search, or SigLIP-2 are novel in themselves.** Adaptive keyframe
> selection, contrastive image-text pretraining, approximate nearest-neighbour
> search, temporal post-processing of retrieval results, and graph-enhanced
> re-ranking are each well-established lines of work. The contribution is the
> *integration* — specifically, a dual-path design where the vector baseline is
> preserved byte-for-byte so that the graph layer's effect can be isolated as a
> controlled experiment rather than argued from intuition.
>
> **What is claimed:**
> - A working, reproducible implementation of the full pipeline.
> - A graph layer whose every edge has a recorded, reproducible basis (a
>   measured cosine similarity, a real timestamp ordering, or a real extractor's
>   output) — with no synthetic or randomly-assigned relationships anywhere in
>   a production code path.
> - An evaluation harness in which absent measurements are reported as `N/A`
>   with a stated reason, and in which the Markdown summary is generated from
>   the same records as the CSVs, so a number cannot reach a report without also
>   reaching the data.
>
> **What is explicitly not claimed:** any quantitative improvement. The
> ablation and comparison tables are currently unpopulated because no ground
> truth has been annotated and no video has been indexed. Until those
> measurements exist, the honest statement is that the architecture is
> implemented, testable, and instrumented — not that it outperforms anything.
>
> The honest current state of the concept layer makes the framing concrete: the
> "Concept Relations" branch of the architecture is an **interface with a
> documented no-op default**, because no detector, OCR engine, ASR model or
> captioner exists in this repository. Populating it with invented labels to
> make the graph look richer would have been the single easiest way to
> fabricate a result, and the code is structured specifically to make that
> impossible.

---

## 11. Configuration reference

All settings live in `backend/app/core/config.py`; `.env.example` documents all
of them. Original 14 first, then the 18 added by this work.

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg2://videocontrol:videocontrol@localhost:5432/videocontrol` | SQLAlchemy URL |
| `MODEL_NAME` | `google/siglip2-so400m-patch14-384` | HF checkpoint for both towers |
| `MEDIA_ROOT` | `./data/media` | uploaded videos |
| `FRAME_ROOT` | `./data/frames` | keyframe thumbnails |
| `CLIP_ROOT` | `./data/clips` | FFmpeg clips |
| `BATCH_SIZE` | `8` | frames per forward pass |
| `FRAME_DIFF_THRESHOLD` | `0.08` | normalised mean-abs-diff keep threshold |
| `MIN_FRAME_INTERVAL_SECONDS` | `0.5` | minimum gap between retained frames |
| `MAX_FRAME_INTERVAL_SECONDS` | `3.0` | forced-keep interval |
| `MAX_UPLOAD_MB` | `2048` | post-write upload cap |
| `TEMPORAL_GROUP_WINDOW` | `3.0` | seconds; result de-duplication in `/search` |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | allow-list |
| `LOG_LEVEL` | `INFO` | root log level |
| `HF_HOME` | `./data/models` | HF cache |
| **Graph & ranking (new)** | | |
| `GRAPH_ENABLED` | `true` | master switch; `false` = original behaviour exactly |
| `GRAPH_EXTRACTOR` | `null` | extraction backend. `null` is the only one registered; an unknown value is a hard error |
| `VECTOR_WEIGHT` | `0.65` | α |
| `GRAPH_WEIGHT` | `0.20` | β |
| `TEMPORAL_WEIGHT` | `0.10` | γ |
| `CONFIDENCE_WEIGHT` | `0.05` | δ |
| `GRAPH_MAX_DEPTH` | `2` | hard hop cap |
| `GRAPH_MAX_NODES` | `200` | hard node cap |
| `GRAPH_SIMILARITY_THRESHOLD` | `0.80` | minimum measured cosine for a `SIMILAR_TO` edge |
| `TEMPORAL_WINDOW_SECONDS` | `5.0` | temporal scale γ. Distinct from `TEMPORAL_GROUP_WINDOW` |
| `GRAPH_SCENE_WINDOW_SECONDS` | `3.0` | scene clustering window |
| `GRAPH_SIMILAR_TOP_K` | `5` | max `SIMILAR_TO` edges per frame |
| `GRAPH_MAX_EDGES_PER_VIDEO` | `20000` | per-video edge cap; truncation is logged |
| `GRAPH_CANDIDATE_MULTIPLIER` | `3` | candidate-pool multiplier over `top_k` |
| `GRAPH_DIRECT_COEFF` | `0.40` | `a_d` |
| `GRAPH_SCENE_COEFF` | `0.25` | `a_s` |
| `GRAPH_EDGE_CONF_COEFF` | `0.20` | `a_c` |
| `GRAPH_TEMPORAL_COEFF` | `0.15` | `a_t` |

---

## Quick Start (TL;DR)

### 🐳 Docker (Recommended — includes pgvector)

```bash
# 1. Configure
cp .env.example .env

# 2. Start all services
docker compose up --build

# 3. Apply graph migration (creates graph_nodes, graph_edges tables)
docker compose exec backend alembic upgrade head

# ✅ Ready!
# Frontend:  http://localhost:5173
# API:       http://localhost:8000
# API Docs:  http://localhost:8000/docs
```

### 💻 Native (No Docker)

```bash
# 1. Install pgvector (one-time)
# Ubuntu: sudo apt install postgresql-16-pgvector
# From source: git clone https://github.com/pgvector/pgvector && cd pgvector && make && sudo make install

# 2. Create database
sudo -u postgres psql -c "CREATE USER videocontrol WITH PASSWORD 'videocontrol';"
sudo -u postgres psql -c "CREATE DATABASE videocontrol OWNER videocontrol;"
sudo -u postgres psql -d videocontrol -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 3. Setup project
cp .env.example .env
mkdir -p data/media data/frames data/clips data/models

# 4. Backend
cd backend
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
psql -U videocontrol -d videocontrol -h localhost -f app/db/init.sql
alembic upgrade head
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 5. Frontend (new terminal)
cd frontend
npm install
export VITE_API_URL=http://localhost:8000   # Windows: $env:VITE_API_URL="http://localhost:8000"
npm run dev
```

### 🧪 Smoke Test

1. Open http://localhost:5173
2. Upload a short `.mp4` (30–60s)
3. Wait for status **ready** (refresh library)
4. Type a query → click result → video jumps to timestamp

---

## 12. Running it

### 12.1 Docker (Recommended)

```bash
# Configure environment
cp .env.example .env          # adjust DATABASE_URL if needed

# Start stack (PostgreSQL + pgvector, Backend, Frontend)
docker compose up --build

# Apply graph tables migration
docker compose exec backend alembic upgrade head
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| API | http://localhost:8000 |
| API Docs (OpenAPI) | http://localhost:8000/docs |
| Postgres | `pgvector/pgvector:pg16` (bundled extension) |

`init.sql` bootstraps base tables on a **fresh** volume; `alembic upgrade head` adds graph tables to existing database.

### 12.2 Native (No Docker)

Prerequisites: [`NATIVE_SETUP.md`](NATIVE_SETUP.md) — requires **PostgreSQL 14+ with pgvector extension**.

```bash
# Project setup
cp .env.example .env
mkdir -p data/media data/frames data/clips data/models

# Backend
cd backend
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
psql -U videocontrol -d videocontrol -h localhost -f app/db/init.sql  # base tables
alembic upgrade head                 # graph tables
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (separate terminal)
cd frontend
npm install
export VITE_API_URL=http://localhost:8000
npm run dev
```

Bootstrap helpers: [`scripts/README.md`](scripts/README.md).

### 12.3 Tests

```bash
# All 178 tests
make test
# or
cd backend && pytest -v

# Database-backed tests (opt-in)
VIDEOCTRLF_RUN_DB_TESTS=1 pytest

# Frontend build verification
cd frontend && npm run build
```

### 12.4 Evaluation Commands

#### Dry Run (Validate Annotations Only — No DB/Model)

```bash
make eval-dry
# or
python -m experiments.evaluation.evaluate_retrieval \
    --dataset experiments/datasets/evaluation.csv \
    --dry-run
```

#### Full Evaluation Study

**Prerequisites:** Indexed video(s) + annotated ground truth in `evaluation.csv`

```bash
make eval
# or
python -m experiments.evaluation.evaluate_retrieval \
    --dataset experiments/datasets/evaluation.csv \
    --model siglip2 \
    --top-k 10 \
    --graph --graph-depth 2 \
    --ablation --depth-sweep --index-metrics \
    --output experiments/results/
```

| Flag | Purpose |
|------|---------|
| `--graph` | Enable graph-aware retrieval (condition D) |
| `--graph-depth` | Graph expansion depth (default: 2) |
| `--ablation` | Run all four conditions A/B/C/D |
| `--depth-sweep` | Sweep graph depths 0,1,2,3 |
| `--index-metrics` | Collect frame-reduction, storage, graph sizes |
| `--dry-run` | Validate annotations only |

#### Evaluation Outputs

All written to `experiments/results/`:

| File | Contents |
|------|----------|
| `retrieval_metrics.csv` / `.json` | Vector vs vector+graph, all metrics @ K=1,5,10,20 |
| `ablation_results.csv` / `.json` | Conditions A/B/C/D |
| `latency_results.csv` | Avg/median/P95/P99 per pipeline stage |
| `indexing_results.csv` | Frame counts, reduction %, graph sizes, storage |
| `graph_results.csv` | Expanded nodes, traversed edges, depth sweep |
| `model_comparison.csv` / `.json` | SigLIP vs SigLIP-2 (when both indexed) |
| `SUMMARY.md` | **Generated** from same rows as CSVs |
| `figures/*.png` | 8 figures (see below) |

---

## 13. Visualization Commands

### Generate All Figures

```bash
make plots
# or individually:
for s in recall precision mrr ndcg latency frame_reduction ablation comparison; do
    python -m experiments.visualization.plot_$s
done
```

### Individual Figures

| Command | Output Figure | Input CSV |
|---------|---------------|-----------|
| `python -m experiments.visualization.plot_recall` | `recall_at_k.png` | `retrieval_metrics.csv` |
| `python -m experiments.visualization.plot_precision` | `precision_at_k.png` | `retrieval_metrics.csv` |
| `python -m experiments.visualization.plot_mrr` | `mrr.png` | `retrieval_metrics.csv` |
| `python -m experiments.visualization.plot_ndcg` | `ndcg_at_k.png` | `retrieval_metrics.csv` |
| `python -m experiments.visualization.plot_latency` | `latency_breakdown.png` | `latency_results.csv` |
| `python -m experiments.visualization.plot_frame_reduction` | `frame_reduction.png` | `indexing_results.csv` |
| `python -m experiments.visualization.plot_ablation` | `ablation.png` | `ablation_results.csv` |
| `python -m experiments.visualization.plot_comparison` | `model_comparison.png` | `model_comparison.csv` |

### From Alternative Results Directory

```bash
python -m experiments.visualization.plot_recall --results-dir /path/to/results
```

### Figure Behavior

- **Measured data** → plotted normally
- **Missing/unmeasured data** → figure written with explicit `N/A` caption (never rendered as zero)
- All scripts read **only** from CSV/JSON — no hard-coded values

---

## 14. Production Readiness Checklist

### ✅ Verified & Complete

| Item | Verification |
|------|--------------|
| Backend imports cleanly | `import app.main` — all routes load |
| Route table integrity | Unchanged + 2 new additive routes |
| Migration SQL | `alembic upgrade head --sql` verified additive |
| Test suite | 178 tests pass (incl. 7 pre-existing) |
| Metrics correctness | 47 tests vs hand-computed values |
| Bounded traversal | 43 tests, incl. query-count assertions |
| Scoring/ranking/tie-breaking | 42 tests |
| Frontend build | `npm run build` succeeds |
| Figures render from CSV | 8 scripts tested end-to-end in CI |
| N/A semantics | 20 tests: `None`→`N/A`, `0.0` stays `0.0` |

Four real bugs found and fixed by tests:
1. Extra traversal query breaking batching contract
2. Unenforced `max_nodes` cap
3. `temporal_score` including candidate itself (constant term)
4. `issubclass` on `Protocol` with non-method member

### ⚠️ Required for Production Deployment

| Item | Status | Action |
|------|--------|--------|
| pgvector extension | ❌ Not installed locally | Use Docker or install via build tools |
| Ground truth annotations | ❌ Empty in `evaluation.csv` | Annotate 5 queries with `video_id` + timestamps |
| Indexed videos | ❌ None in database | Upload + index at least one video |
| Model warm-up | ❌ Disabled in `main.py` | Uncomment `get_model_service()` in startup |
| SSL/TLS | ❌ Not configured | Add reverse proxy (nginx/Caddy) with certs |
| Authentication | ❌ Not implemented | Add auth middleware if multi-user |
| Rate limiting | ❌ Not implemented | Add `slowapi` or similar |
| Monitoring/logging | ⚠️ Basic only | Add structured logging, metrics export |
| Backup strategy | ❌ Not defined | Document `pg_dump` + volume backup |

### 🔧 Key Production Configuration (`.env`)

```bash
# Database — use managed PostgreSQL with pgvector
DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/db

# Model — consider smaller variant if VRAM limited
MODEL_NAME=google/siglip2-so400m-patch14-384

# Storage — use persistent volumes
MEDIA_ROOT=/data/media
FRAME_ROOT=/data/frames
CLIP_ROOT=/data/clips

# Performance tuning
BATCH_SIZE=8                    # Lower if OOM
GRAPH_MAX_DEPTH=2               # Hard cap
GRAPH_MAX_NODES=200             # Hard cap
GRAPH_SIMILARITY_THRESHOLD=0.80 # Strict for precision

# Security
CORS_ORIGINS=https://yourdomain.com
LOG_LEVEL=INFO
```

---

## 15. Known Limitations (Honest Assessment)

1. **Concept layer is empty** — No detector/OCR/ASR/captioner. `DEPICTS` edges never written. Graph signal limited to similarity + temporal + scene structure.

2. **ConfidenceScore is ranking-neutral** — All nodes have `confidence=1.0` with `NullGraphExtractor`. Ablation sets weight=0 intentionally.

3. **Weights are defaults, not tuned** — α=0.65, β=0.20, etc. are engineering choices. No claim of optimality.

4. **Five queries is not a benchmark** — Even with annotations, sample is tiny. Report as single observations.

5. **One video ≠ corpus** — Frame-reduction/storage from single clip doesn't generalize.

6. **Pre-existing inconsistencies** — `.env` paths vs Docker mounts; `test_adaptive_extraction.py` doesn't test extraction; FFmpeg only in Docker.

---

## 16. Document Map

| Document | Contents |
|----------|----------|
| `docs/AUDIT_REPORT.md` | Phase 0 repository audit |
| `docs/EVALUATION.md` | Every metric with exact formula, matching rule, worked example |
| `experiments/README.md` | How to reproduce experiments; measured-vs-implemented status |
| `experiments/datasets/README.md` | Column reference + how to build labelled set |
| `scripts/README.md` | Environment bootstrap scripts + pgvector blocker |
| `NATIVE_SETUP.md` | Native (non-Docker) setup |
| `AGENT.md` | This file — project state, commands, architecture reference |

---

**Team:** Vignesh K (23AM069) · Yogaazhaki S (23AM072) · Deepika S (23AM016)

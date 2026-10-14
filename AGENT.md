# VideoCtrl-F - Agent Documentation

## Project Overview

**VideoCtrl-F** is an adaptive keyframe and SigLIP-2 based semantic video retrieval framework, extended with a relational knowledge-graph layer and graph-aware re-ranking.

A multimodal video search engine: upload a video, it is decoded adaptively into keyframes, each keyframe is embedded with SigLIP-2, and those embeddings are indexed in PostgreSQL + pgvector for fast cosine-similarity search. A text query is embedded in the same space, and the closest frames are returned as timestamps and clips. On top of that, each video also gets a **relational graph** — scenes, frames, and measured frame-to-frame similarity — which is used to re-rank results and to explain why each result was returned.

---

## Current Implementation Status (as of 2026-10-13)

### ✅ Completed & Verified

| Component | Status | Verification |
|-----------|--------|--------------|
| Backend API (FastAPI) | Complete | 178 tests pass |
| Adaptive Keyframe Extraction | Complete | Unit tested |
| SigLIP-2 Embeddings (Image + Text) | Complete | Model singleton verified |
| pgvector Integration | Complete | HNSW index, cosine similarity |
| Relational Graph Layer | Complete | Schema, construction, traversal |
| Bounded Graph Traversal | Complete | 43 tests, query-count assertions |
| Graph-Aware Re-ranking | Complete | 42 tests, ablation configs A/B/C/D |
| Evaluation Metrics | Complete | 47 tests vs hand-computed values |
| Visualization (8 figures) | Complete | End-to-end tested in CI |
| Frontend (React + Vite + Tailwind) | Complete | Builds successfully |
| Docker Compose Stack | Complete | Postgres + pgvector, Backend, Frontend |
| Alembic Migrations | Complete | Additive, verified via --sql |

### 🔄 Partially Complete / Blocked

| Component | Status | Blocker |
|-----------|--------|---------|
| End-to-end indexing | Code ready | pgvector not installed locally |
| Retrieval evaluation | Code ready | No indexed videos + no ground truth |
| SigLIP vs SigLIP-2 comparison | Code ready | Only one checkpoint indexed |
| Concept extraction layer | Interface ready | No detector/OCR/ASR/captioner implemented |

### ❌ Not Yet Measured (No Data)

| Metric Family | Implementation | Measured |
|---------------|----------------|----------|
| Recall@K, Precision@K, MRR, MAP, NDCG@K | ✅ | ❌ N/A — no ground truth |
| Latency per stage | ✅ | ❌ N/A — no indexed video |
| Frame reduction % | ✅ | ❌ N/A — no indexed video |
| Storage overhead | ✅ | ❌ N/A — no indexed video |
| Graph expansion counts | ✅ | ❌ N/A — no indexed video |

---

## Architecture Summary

### Three Flows

1. **Upload/Indexing Flow** — Video → Adaptive Frames → SigLIP-2 Embeddings → pgvector + Relational Graph
2. **Search Flow (Baseline)** — Text Query → SigLIP-2 Text Embedding → pgvector Top-K → Temporal Grouping → Results
3. **Graph Flow (New)** — Text Query → pgvector Candidates → Bounded Graph Expansion → 4-Term Re-ranking → Results + Evidence

### Relational Graph Schema

```
VIDEO → CONTAINS → SCENE → BELONGS_TO → FRAME
FRAME → FOLLOWED_BY / PRECEDES → FRAME (temporal ordering)
FRAME → SIMILAR_TO → FRAME (measured cosine ≥ 0.80)
FRAME → DEPICTS → CONCEPT/OBJECT/TEXT (empty - no extractor)
```

### Scoring Formula

```
FinalScore = (α·VectorScore + β·GraphScore + γ·TemporalScore + δ·ConfidenceScore) / (α+β+γ+δ)

GraphScore = (a_d·DirectSimilarity + a_s·SceneOverlap + a_c·EdgeConfidence + a_t·TemporalAdjacency) / (a_d+a_s+a_c+a_t)
```

Defaults: α=0.65, β=0.20, γ=0.10, δ=0.05 | a_d=0.40, a_s=0.25, a_c=0.20, a_t=0.15

---

## Key Files Reference

### Backend Core

| File | Purpose |
|------|---------|
| `backend/app/main.py` | FastAPI app, CORS, static mounts, /health |
| `backend/app/core/config.py` | All 32 settings (14 original + 18 graph) |
| `backend/app/db/init.sql` | Base schema bootstrap (fresh volume) |
| `backend/alembic/versions/0001_graph_layer.py` | Graph tables migration |

### Services (Backend)

| File | Purpose |
|------|---------|
| `model_service.py` | SigLIP-2 singleton (images + text) |
| `embedding_service.py` | Adaptive keyframes, thumbnails, embeddings |
| `video_service.py` | Upload, background indexing, clip pipeline |
| `search_service.py` | **FROZEN** - vector-only baseline |
| `graph_extraction_service.py` | GraphExtractor protocol + NullGraphExtractor |
| `graph_service.py` | Graph construction + bounded traversal |
| `graph_ranking_service.py` | Four-term scoring formula + ablation configs |
| `graph_search_service.py` | Graph-aware retrieval pipeline |

### API Routes

| Route | Purpose |
|-------|---------|
| `POST /api/v1/videos` | Upload video |
| `GET /api/v1/videos` | List videos |
| `POST /api/v1/search` | **FROZEN** - vector-only search |
| `POST /api/v1/search/graph` | **NEW** - graph-aware search |
| `GET /api/v1/videos/{id}/graph` | **NEW** - fetch full video graph |
| `POST /api/v1/search/clip/{frame_id}` | Generate 5s FFmpeg clip |

### Frontend

| File | Purpose |
|------|---------|
| `src/App.tsx` | Mode toggle (vector/graph), depth selector |
| `src/api.ts` | Complete API boundary (search + graph) |
| `src/hooks/useVideoSearch.ts` | **FROZEN** - vector search hook |
| `src/hooks/useGraphSearch.ts` | **NEW** - graph search hook |
| `src/components/GraphResultCard.tsx` | "Why this result?" panel + score breakdown |
| `src/components/GraphSearchResults.tsx` | Graph results list + diagnostics |

### Evaluation & Visualization

| File | Purpose |
|------|---------|
| `experiments/evaluation/evaluate_retrieval.py` | Main CLI entrypoint |
| `experiments/evaluation/metrics.py` | All IR metrics implementations |
| `experiments/evaluation/ranking.py` | Ablation conditions A/B/C/D |
| `experiments/evaluation/graph_metrics.py` | Expansion observations |
| `experiments/evaluation/reporting.py` | CSV/JSON/Markdown output |
| `experiments/visualization/plot_*.py` | 8 figure scripts |
| `experiments/visualization/_common.py` | Shared plotting utilities |

---

## Environment Requirements

### Hard Requirements

- **PostgreSQL 14+ with pgvector extension** — *Required for vector column*
- **Python 3.11+** — Backend dependencies pinned for 3.11/3.12
- **Node.js 18+** — Frontend build
- **FFmpeg** — Clip generation

### GPU (Optional but Recommended)

- CUDA-capable GPU with 8GB+ VRAM for SigLIP-2
- CPU works but is significantly slower

---

## Running Commands

### Docker (Recommended - Includes pgvector)

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env if needed (DATABASE_URL, etc.)

# 2. Start stack
docker compose up --build

# 3. Apply graph migration (graph tables)
docker compose exec backend alembic upgrade head

# Access:
# Frontend: http://localhost:5173
# API:      http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Native (No Docker)

```bash
# 1. Install pgvector (one-time)
# Ubuntu: sudo apt install postgresql-16-pgvector
# From source: git clone pgvector && make && sudo make install

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
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
psql -U videocontrol -d videocontrol -h localhost -f app/db/init.sql
cd backend && alembic upgrade head
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 5. Frontend (new terminal)
cd frontend
npm install
export VITE_API_URL=http://localhost:8000
npm run dev
```

### Quick Test (Smoke Test)

```bash
# 1. Open http://localhost:5173
# 2. Upload a short .mp4 (30-60s)
# 3. Wait for status "ready"
# 4. Type a query → click result → video jumps to timestamp
```

---

## Evaluation Commands

### Dry Run (Validate Annotations Only)

```bash
# No database, no model - just validates CSV
make eval-dry
# or
python -m experiments.evaluation.evaluate_retrieval \
    --dataset experiments/datasets/evaluation.csv \
    --dry-run
```

### Full Evaluation Study

```bash
# Requires: indexed video(s) + annotated ground truth in evaluation.csv
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

### Evaluation Outputs

| File | Contents |
|------|----------|
| `experiments/results/retrieval_metrics.csv` | Vector vs vector+graph, all metrics @ K=1,5,10,20 |
| `experiments/results/ablation_results.csv` | Conditions A/B/C/D |
| `experiments/results/latency_results.csv` | Avg/median/P95/P99 per pipeline stage |
| `experiments/results/indexing_results.csv` | Frame counts, reduction %, graph sizes, storage |
| `experiments/results/graph_results.csv` | Expanded nodes, traversed edges, depth sweep |
| `experiments/results/model_comparison.csv` | SigLIP vs SigLIP-2 (when both indexed) |
| `experiments/results/SUMMARY.md` | Generated from same rows as CSVs |
| `experiments/results/figures/*.png` | 8 figures (see below) |

---

## Visualization Commands

### Generate All Figures

```bash
make plots
# or manually:
for s in recall precision mrr ndcg latency frame_reduction ablation comparison; do
    python -m experiments.visualization.plot_$s
done
```

### Individual Figures

| Command | Output | Input CSV |
|---------|--------|-----------|
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

---

## Testing

```bash
# All tests (178)
make test
# or
cd backend && pytest -v

# Database-backed tests (opt-in)
VIDEOCTRLF_RUN_DB_TESTS=1 pytest

# Frontend build test
cd frontend && npm run build
```

---

## Production Readiness Checklist

### ✅ Completed

- [x] Backend imports cleanly, all routes load
- [x] Route table unchanged + 2 new additive routes
- [x] Migration SQL correct and additive (verified via alembic --sql)
- [x] 178 backend tests pass (including 7 pre-existing)
- [x] All metrics unit-tested against hand-computed values (47 tests)
- [x] Bounded traversal verified (no N+1, hard caps enforced) - 43 tests
- [x] Scoring/ranking/tie-breaking verified - 42 tests
- [x] Frontend builds successfully (`npm run build`)
- [x] All 8 figures render from CSV (tested in `test_reporting_and_plots.py`)
- [x] N/A semantics verified - 20 tests asserting None→N/A, 0.0 stays 0.0
- [x] Four real bugs found and fixed by tests

### ⚠️ Needs Attention for Production

| Item | Status | Action Needed |
|------|--------|---------------|
| pgvector extension | ❌ Not installed locally | Install via Docker or build tools |
| Ground truth annotations | ❌ Empty in evaluation.csv | Annotate 5 queries with video_id + timestamps |
| Indexed videos | ❌ None in database | Upload and index at least one video |
| Model warm-up | ❌ Disabled in main.py | Uncomment model warm-up on startup |
| SSL/TLS | ❌ Not configured | Add reverse proxy (nginx/Caddy) with certs |
| Authentication | ❌ Not implemented | Add auth middleware if multi-user |
| Rate limiting | ❌ Not implemented | Add slowapi or similar |
| Monitoring/logging | ⚠️ Basic only | Add structured logging, metrics export |
| Backup strategy | ❌ Not defined | Document pg_dump + volume backup |

### 🔧 Configuration for Production

Key settings in `.env` to adjust:

```bash
# Database (use managed PostgreSQL with pgvector)
DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/db

# Model (consider smaller variant if VRAM limited)
MODEL_NAME=google/siglip2-so400m-patch14-384

# Storage (use persistent volumes)
MEDIA_ROOT=/data/media
FRAME_ROOT=/data/frames
CLIP_ROOT=/data/clips

# Performance
BATCH_SIZE=8                    # Lower if OOM
GRAPH_MAX_DEPTH=2               # Hard cap
GRAPH_MAX_NODES=200             # Hard cap
GRAPH_SIMILARITY_THRESHOLD=0.80 # Strict for precision

# Security
CORS_ORIGINS=https://yourdomain.com
LOG_LEVEL=INFO
```

---

## Known Limitations (Honest Assessment)

1. **Concept layer is empty** — No detector/OCR/ASR/captioner. `DEPICTS` edges never written. Graph signal limited to similarity + temporal + scene structure.

2. **ConfidenceScore is ranking-neutral** — All nodes have confidence=1.0 with NullGraphExtractor. Ablation sets weight=0 intentionally.

3. **Weights are defaults, not tuned** — α=0.65, β=0.20, etc. are engineering choices. No claim of optimality.

4. **Five queries is not a benchmark** — Even with annotations, sample is tiny. Report as single observations.

5. **One video ≠ corpus** — Frame-reduction/storage from single clip doesn't generalize.

6. **Pre-existing inconsistencies** — `.env` paths vs Docker mounts; `test_adaptive_extraction.py` doesn't test extraction; FFmpeg only in Docker.

---

## Team

- Vignesh K — 23AM069
- Yogaazhaki S — 23AM072
- Deepika S — 23AM016

Project duration: 8 Months
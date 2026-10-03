# Phase 0 — Repository Audit Report

**Repository:** VideoCtrl-F (`D:\VideoCtrlF`)
**Audit date:** 2026-09-27
**Auditor scope:** full read-only inspection of every tracked source file, plus live probing of the local Python, Node, PostgreSQL and Docker environments.
**Method:** every statement below was produced by opening and reading the referenced file, or by executing a read-only probe. Nothing is inferred from the prompt's description of the system.

---

## 0. Executive summary

The repository is a **complete, working, single-service semantic video retrieval system** — considerably more mature than the prompt's "Context" section implies (it already has a real vector index, HNSW, real embedding generation, real adaptive keyframe extraction, real clip generation, a working React UI, and a Docker Compose stack).

It is **also structurally different** from what the prompt assumes, in five ways that materially change the plan:

| Prompt assumption | Reality in repo | Consequence |
|---|---|---|
| Alembic migrations | **No Alembic.** Schema is raw SQL in `backend/app/db/init.sql`, auto-applied only on first Docker DB init | A versioned migration mechanism must be *introduced*, not merely extended |
| `services/`-heavy FastAPI with repositories | Flat 4-layer: `api/v1/routers` → `controllers` → `services` → `models` | Follow the existing layer boundaries exactly; do not invent a DAO layer |
| An evaluation framework exists | `experiments/` is **scaffolding only**; `evaluate.py` scores *supplied prediction files* and never runs the retrieval system | The evaluation framework must be built essentially from scratch |
| Graph extraction has inputs (OCR/object detection) | **No OCR, no object detector, no ASR, no captioner, no concept extractor exists anywhere in the repo** | `graph_extraction_service.py` must be a documented no-op interface, per prompt §0 |
| Ground truth exists for evaluation | `experiments/evaluation_dataset.csv` has 5 queries with **empty `video_id` and empty `ground_truth_timestamp`** | Zero usable ground truth. Metrics must be reported `N/A` until annotations exist, or annotations must be created and labelled as such |

**Nothing in this repository is a stub or a placeholder.** Every line in `services/`, `models/`, `api/`, and `frontend/src/` is real, working code. This is a strong foundation to extend additively.

---

## 1. Frontend

**Location:** `frontend/`

| Aspect | Finding |
|---|---|
| Framework | React 18.3.1 + TypeScript 5.5, built by Vite 5.4 (`frontend/vite.config.ts`) |
| Styling | Tailwind CSS 3.4 + PostCSS/autoprefixer (`tailwind.config.js`, `postcss.config.js`) |
| Icons | `lucide-react` 0.441 |
| HTTP | `axios` 1.7, single pre-configured client in `frontend/src/api.ts` with `baseURL = import.meta.env.VITE_API_URL` and `timeout: 120000` |
| **Routing** | **There is no router.** `react-router-dom` is not a dependency. The app is a single page; "navigation" is React state (`selectedId`, `seekTo`) in `App.tsx`. |
| **State management** | **No Redux / Zustand / Jotai / Context store.** Plain `useState` + `useCallback` in `App.tsx`, plus one custom hook `useVideoSearch.ts`. |
| Tests | **None.** No Vitest/Jest config, no test script in `package.json`. |
| Build | `npm run build` = `tsc -b && vite build` (type-checked) |
| Dockerfile | `frontend/Dockerfile` exists; compose maps `5173:5173` and injects `VITE_API_URL=http://localhost:8000` |

**File-by-file:**

- `src/main.tsx` — React root mount only.
- `src/App.tsx` (113 lines) — the entire application shell. Polls `GET /api/v1/videos` every 5 s via `setInterval`; holds `videos`, `selectedId`, `seekTo`, `currentTime`; renders a 3-column Tailwind grid: left sidebar (`VideoUploader`, `VideoLibrary`), main column (`SearchBar`, `VideoPlayer`, `TimelineHeatmap`, `SearchResults`). `handleJump(ts, videoId)` selects the video and seeks.
- `src/api.ts` (89 lines) — the **entire** typed API surface. Exports `Video`, `SearchResult`, `SearchResponse` interfaces and functions `uploadVideo`, `getVideos`, `getVideo`, `search`, `generateClip`, `mediaUrl`. **This is the only file that must change to surface graph results.**
- `src/hooks/useVideoSearch.ts` — wraps `search()` with `results`, `loading`, `error`, `latency`, `lastQuery` state. Calls `POST /api/v1/search` with `{query, video_id, top_k, min_score: 0.0}`.
- `src/components/VideoUploader.tsx` — drag-and-drop + file input, progress bar from axios `onUploadProgress`.
- `src/components/VideoLibrary.tsx` — list with per-status icons (`ready`/`processing`/`queued`/`failed`) and `error_message` display.
- `src/components/SearchBar.tsx` — controlled input, disabled unless a video with `status === "ready"` is selected.
- `src/components/VideoPlayer.tsx` — plain `<video>`; seeks when `seekTo` changes.
- `src/components/TimelineHeatmap.tsx` — maps results to horizontal markers; marker opacity is `clamp(0.3, 1, r.score)`.
- `src/components/SearchResults.tsx` — list wrapper, shows `latency.toFixed(0) ms`.
- `src/components/ResultCard.tsx` — thumbnail, timestamp, `Similarity {(score*100).toFixed(1)}%`, Jump + Clip buttons. **This is where graph explainability should be rendered.**
- `src/styles/globals.css` (413 bytes) — Tailwind directives plus a `brand` colour ramp used throughout.

---

## 2. Backend

**Location:** `backend/`, package root `app/`, `PYTHONPATH=/app` in the Dockerfile.

### 2.1 Structure (actual, complete file list)

```
backend/
├── Dockerfile
├── requirements.txt
├── test_gpu.py                 # standalone GPU probe, not part of the app
├── test.jpg                    # stray asset
├── data/media/e11a89c8-…mp4    # 92,123,729 bytes, git-tracked (see §9)
├── app/
│   ├── main.py                 # FastAPI app, CORS, static mounts, /health
│   ├── core/config.py          # pydantic-settings Settings + lru_cache
│   ├── core/logging.py         # setup_logging() → "videocontrolf" logger
│   ├── db/session.py           # engine, SessionLocal, Base, get_db
│   ├── db/init.sql             # THE schema (no migrations tool)
│   ├── models/{__init__,video,frame}.py
│   ├── schemas/{__init__,search,video}.py
│   ├── controllers/{__init__,search_controller,video_controller}.py
│   ├── services/{__init__,model_service,embedding_service,video_service,search_service,clip_service}.py
│   └── api/v1/
│       ├── router.py           # APIRouter(prefix="/api/v1")
│       └── routers/{__init__,videos,search}.py
└── tests/{__init__,test_health,test_schemas,test_adaptive_extraction,test_evaluate}.py
```

Layering is strictly downward: **routers → controllers → services → models → db**. Controllers contain only HTTP error mapping (`ValueError → 400/404`, `Exception → 500`). Services contain all logic and import models directly. There is **no repository/DAO layer**, and adding one would be inconsistent with the existing style.

### 2.2 `app/main.py`
- `FastAPI(title="VideoCtrl-F", version="1.0.0", docs_url="/docs", redoc_url="/redoc")`.
- CORS from `settings.cors_origins_list`, `allow_credentials=True`, all methods/headers.
- Creates `MEDIA_ROOT`, `FRAME_ROOT`, `CLIP_ROOT` at import time.
- Static mounts: `/media/frames` → `FRAME_ROOT`, `/media/clips` → `CLIP_ROOT`, `/media/videos` → `MEDIA_ROOT`.
- `GET /health` → `{"status": "ok"}`.
- `@app.on_event("startup")` logs only; model warm-up is commented out (lines 52–53).
- Catch-all `Exception` handler → 500 `{"detail": "Internal server error"}`.

### 2.3 `app/core/config.py`
`Settings(BaseSettings)` with `env_file = (".env", "../.env")`, `case_sensitive=True`, `extra="ignore"`, `get_settings()` wrapped in `@lru_cache()`. Fourteen fields — see §8 for the full inventory.

### 2.4 Controllers
- `video_controller.upload_video` → `video_service.create_video_record`; `list_videos`; `get_video` (404 if missing).
- `search_controller.perform_search` → `search_service.search`; `get_clip` → `search_service.generate_result_clip` (404 on `ValueError`).

### 2.5 `app/api/v1/`
`router.py` includes `videos.router` (tags `["Videos"]`) then `search.router` (tags `["Search"]`) under prefix `/api/v1`.

**Complete existing endpoint inventory (must remain byte-compatible):**

| Method | Path | Handler | Request | Response |
|---|---|---|---|---|
| GET | `/health` | `main.health` | – | `{"status":"ok"}` |
| POST | `/api/v1/videos` | `videos.upload_video` | multipart `file` | `VideoResponse` |
| GET | `/api/v1/videos?skip=&limit=` | `videos.list_videos` | – | `VideoListResponse` |
| GET | `/api/v1/videos/{video_id}` | `videos.get_video` | – | `VideoResponse` |
| POST | `/api/v1/search` | `search.search` | `SearchRequest` | `SearchResponse` |
| POST | `/api/v1/search/clip/{frame_id}` | `search.generate_clip` | – | `{"clip_url": str}` |

---

## 3. Database

**Engine:** `postgresql+psycopg2://…`, `pool_pre_ping=True`, `SessionLocal(autocommit=False, autoflush=False)`, `Base = declarative_base()`, `get_db()` generator dependency.

### 3.1 Complete current schema — `backend/app/db/init.sql`

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS videos (
    id UUID PRIMARY KEY,
    filename VARCHAR(512) NOT NULL,
    original_filename VARCHAR(512) NOT NULL,
    path VARCHAR(1024) NOT NULL,
    duration_seconds FLOAT,
    fps FLOAT,
    frame_count INTEGER,
    status VARCHAR(32) NOT NULL DEFAULT 'queued',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS video_frames (
    id UUID PRIMARY KEY,
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    timestamp_seconds FLOAT NOT NULL,
    frame_number INTEGER NOT NULL,
    thumbnail_path VARCHAR(1024) NOT NULL,
    embedding vector(1152),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_video_frames_video_id ON video_frames(video_id);
CREATE INDEX IF NOT EXISTS idx_video_frames_embedding
    ON video_frames USING hnsw (embedding vector_cosine_ops);
```

**Existing indexes:** exactly two. A btree on `video_frames(video_id)` and an HNSW index on `video_frames.embedding` with `vector_cosine_ops`.

**Existing cascade:** `video_frames.video_id → videos.id ON DELETE CASCADE`. Deleting a video removes its frames.

### 3.2 SQLAlchemy models
- `app/models/video.py` → `Video`, 10 columns, mirrors `videos`.
- `app/models/frame.py` → `VideoFrame`, `embedding = Column(Vector(1152), nullable=True)`, `video_id` FK with `ondelete="CASCADE"`, `index=True`.
- `app/models/__init__.py` re-exports `Video`, `VideoFrame`. **Any new model must be added here.**

### 3.3 Migration tooling — **GAP**

**There is no Alembic, no `alembic.ini`, no `versions/` directory, no `migrations/` package.** Schema is applied exclusively by mounting `init.sql` into `/docker-entrypoint-initdb.d/`, which PostgreSQL only executes **when the data volume is empty**. Consequences:

1. `CREATE TABLE IF NOT EXISTS` means a schema *change* to an existing table is silently ignored on any already-initialised database.
2. There is no schema version, no upgrade path, and no way to add the graph tables to a running deployment.
3. The prompt requires "a real migration … do not hand-edit the DB". With `init.sql` alone that requirement **cannot be met**.

**Resolution taken (flagged for confirmation):** introduce Alembic as a *new, additive* dev dependency, with `init.sql` retained unchanged as the fresh-Docker bootstrap. New graph tables get real Alembic revisions. Existing `videos`/`video_frames` are **not** touched by any revision, so the migration is non-destructive and forward-only.

### 3.4 JSONB — **GAP**
`JSONB` is **not used anywhere** in the current schema. The `metadata JSONB` columns the prompt requests for `graph_nodes`/`graph_edges` would be the first use. This is additive and safe.

---

## 4. AI / model services

**File:** `backend/app/services/model_service.py` (68 lines). Singleton via `__new__` + `threading.Lock`, lazy via module-level `_model_service` and `get_model_service()`.

| Property | Value |
|---|---|
| Model id | `settings.MODEL_NAME`, default **`google/siglip2-so400m-patch14-384`** |
| Loader | `transformers.AutoProcessor.from_pretrained` + `transformers.AutoModel.from_pretrained` |
| **Embedding dimension** | **1152** (matches `Vector(1152)` in the schema) |
| Device | `"cuda" if torch.cuda.is_available() else "cpu"`, chosen at `__init__` |
| Image path | `processor(images=…)` → `model.get_image_features(**inputs)` → `torch.nn.functional.normalize(emb, p=2, dim=-1)` → `emb.cpu().numpy()` |
| Text path | `processor(text=…, padding=True)` → `model.get_text_features(**inputs)` → same L2 normalisation |
| Batching | Handled by the *caller* (`embedding_service`), not the model service |
| Precision | Full fp32. **No** `torch.float16`, **no** autocast, **no** `torch.inference_mode()` |
| Preprocessing | **Fully delegated to the HF `AutoProcessor`.** No custom resize/normalize/augment code exists. |

**Gap:** no batched text encoding beyond what the caller passes; `encode_text` accepts `str | list[str]` and always returns a 2-D array.

---

## 5. Video pipeline

**File:** `backend/app/services/embedding_service.py` (142 lines).

### 5.1 `extract_adaptive_keyframes(video_path) -> List[(frame_number, ts_seconds, bgr_frame)]`

The actual heuristic, read line by line:

1. `cv2.VideoCapture`; raises `ValueError` if not opened.
2. Reads `CAP_PROP_FPS` (default 25.0 if falsy) and `CAP_PROP_FRAME_COUNT` (default 0).
3. Sequentially decodes **every** frame. For each frame at index `i`, `ts = i / fps`.
4. **Frame 0 is always kept.**
5. Otherwise: grayscale → `cv2.resize(gray, (64, 36))` → `diff = mean(|small − prev_small|) / 255.0`.
6. Keep if `(diff >= FRAME_DIFF_THRESHOLD and (ts − last_kept_ts) >= MIN_FRAME_INTERVAL_SECONDS)` **or** `(ts − last_kept_ts) >= MAX_FRAME_INTERVAL_SECONDS`.
7. On keep: re-grayscale, store the new `prev_gray_small`, update `last_kept_ts`, append a `.copy()` of the BGR frame.
8. Logs candidate/retained counts and the retained ratio.

Note: the comparison baseline `prev_gray_small` is the last **retained** frame, not the previous decoded frame.

### 5.2 `process_and_store_embeddings(db, video_id, video_path, retained) -> int`

- `FRAME_ROOT/<video_id>/` created.
- Iterates in `BATCH_SIZE` (=8) slices.
- Per frame: BGR→RGB→`PIL.Image`, thumbnail saved as `{frame_number:08d}.jpg` at **320×180, JPEG quality 85**.
- `model.encode_images(pil_images)` for the batch.
- One `VideoFrame` per frame with `embedding=emb.tolist()`.
- **`db.commit()` after every batch** — i.e. indexing is *not* a single transaction today. This matters for the prompt's "all graph writes in one transaction" requirement (see §10).

### 5.3 Thumbnails / clips
- Thumbnails: `FRAME_ROOT/<video_id>/<8-digit frame number>.jpg`, served at `/media/frames/<video_id>/<name>`.
- Clips: `services/clip_service.generate_clip(video_path, timestamp_seconds, duration=5.0, video_id)` → FFmpeg, argument list (no shell), `-ss (ts − 2.5)` clamped at 0, `-t 5`, `libx264 -preset fast -crf 23`, `aac`, `+faststart`, 120 s timeout, random UUID filename, returns the **relative** name. API returns `/media/clips/<name>`.

### 5.4 Orchestration — `services/video_service.py`
`create_video_record` → validate extension (`{.mp4,.mov,.mkv,.webm,.avi}`) → `save_upload` to `MEDIA_ROOT/<video_id><ext>` with post-write size check (`MAX_UPLOAD_MB`) and an empty-file check → `get_video_metadata` via OpenCV → insert `Video(status="queued")` → `background_tasks.add_task(process_video_background, video_id)`.

`process_video_background` opens its **own** `SessionLocal()` and runs: `status="processing"` → `extract_adaptive_keyframes` → (`status="failed"`, `"No frames could be extracted"` if empty) → `process_and_store_embeddings` → backfill duration/fps/frame_count → `status="ready"`. On any exception: log, set `status="failed"` + `error_message=str(e)[:500]`. `finally: db.close()`.

**The `failed` status and the error-message convention already exist**, so the prompt's "roll back and set `video.status = failed`" requirement maps onto existing behaviour with no schema change.

---

## 6. Search pipeline

**File:** `backend/app/services/search_service.py` (135 lines).

### 6.1 `search(db, request) -> SearchResponse` — end to end

1. `t0 = time.perf_counter()`.
2. `model.encode_text(request.query)[0].tolist()` → query vector.
3. Raw SQL via `db.execute(text(sql), params).mappings().all()`:

```sql
SELECT f.id AS frame_id, f.video_id, v.original_filename AS video_name,
       f.timestamp_seconds, f.thumbnail_path,
       1 - (f.embedding <=> :emb::vector) AS score
FROM video_frames f
JOIN videos v ON v.id = f.video_id
WHERE v.status = 'ready'
  AND 1 - (f.embedding <=> :emb::vector) >= :min_score
  -- + "AND f.video_id = :video_id" when request.video_id is set
ORDER BY f.embedding <=> :emb::vector ASC
LIMIT :limit            -- = request.top_k * 3   (over-fetch for grouping)
```

The query vector is passed as `str(query_emb)`, relying on PostgreSQL's text→`vector` cast. Similarity is `1 − cosine_distance`; the same expression appears in `SELECT`, `WHERE` and `ORDER BY`.

4. Each row → dict; `thumbnail_url` is rebuilt as `/media/frames/{video_id}/{basename}` (path is **discarded and reconstructed from the video_id + filename**).
5. `_temporal_group(raw_results, TEMPORAL_GROUP_WINDOW)`: sorts by timestamp, greedily groups consecutive results whose gap ≤ window, keeps only the **highest-scoring** frame per group, assigns `group_id`, re-sorts by score descending.
6. Truncate to `request.top_k`.
7. `latency_ms = round((time.perf_counter() − t0) * 1000, 2)`.
8. Returns `SearchResponse(query, results, total, latency_ms)`.

`generate_result_clip(db, frame_id)` looks up frame → video, calls `generate_clip(..., duration=5.0)`, returns `/media/clips/{name}`.

### 6.2 Request/response schemas — `app/schemas/search.py` (**must not change**)

```python
SearchRequest:  query: str (min_length=2, max_length=500, non-blank validator),
                video_id: Optional[UUID] = None,
                top_k: int = 10 (ge=1, le=100),
                min_score: float = 0.0 (ge=0.0, le=1.0)
SearchResult:   frame_id: UUID, video_id: UUID, video_name: str,
                timestamp_seconds: float, score: float, thumbnail_url: str,
                clip_url: Optional[str] = None, group_id: Optional[int] = None
SearchResponse: query: str, results: List[SearchResult], total: int, latency_ms: float
```

**Note for Phase 2:** the prompt's proposed graph response uses `final_score` and drops `score`. Those must be **new fields on new response models**, not edits to `SearchResult`/`SearchResponse`, or every existing client breaks.

---

## 7. Docker

`docker-compose.yml` (`version: "3.9"`) — three services, five named volumes:

| Service | Image / build | Key wiring |
|---|---|---|
| `postgres` | `pgvector/pgvector:pg16` | `POSTGRES_USER/PASSWORD/DB = videocontrol`; volume `postgres_data` + `./backend/app/db/init.sql:/docker-entrypoint-initdb.d/init.sql`; port `5432:5432`; `pg_isready` healthcheck (5 s interval, 10 retries) |
| `backend` | build `./backend` | `env_file: .env`; volumes `media_data:/data/media`, `frame_data:/data/frames`, `clip_data:/data/clips`, `model_cache:/models`; port `8000:8000`; `depends_on: postgres (service_healthy)`; `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload` |
| `frontend` | build `./frontend` | port `5173:5173`; `VITE_API_URL=http://localhost:8000`; `depends_on: backend` |

**Inconsistency found (pre-existing, not introduced by me):** `.env` ships `MEDIA_ROOT=./data/media`, `FRAME_ROOT=./data/frames`, `CLIP_ROOT=./data/clips` — **project-root-relative** — while `docker-compose.yml` mounts those directories at **`/data/...`** (absolute). Inside the container the configured relative paths resolve to `/data/media` etc. only because the process CWD is `/app`… which is *not* where they are. Docker-based media serving is therefore inconsistent with the native setup. Flagged, not fixed, since fixing it would change existing behaviour.

`backend/Dockerfile`: `python:3.11-slim`; apt `ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1`; `pip install -r requirements.txt`; `ENV PYTHONPATH=/app`, `ENV HF_HOME=/models`; `mkdir -p /data/media /data/frames /data/clips /models`.

`backend/requirements.txt` (pinned): `fastapi==0.115.0`, `uvicorn[standard]==0.30.6`, `pydantic==2.9.2`, `pydantic-settings==2.5.2`, `sqlalchemy==2.0.35`, `psycopg2-binary==2.9.9`, `pgvector==0.3.4`, `python-multipart==0.0.9`, `torch>=2.2.0`, `transformers>=4.45.0`, `Pillow==10.4.0`, `opencv-python-headless==4.10.0.84`, `numpy==1.26.4`, `python-dotenv==1.0.1`, `httpx==0.27.2`, `pytest==8.3.3`, `pytest-asyncio==0.24.0`, `aiofiles==24.1.0`.

**Not present:** `matplotlib`, `pandas`, `alembic` — all required by Phases 1 and 3.

---

## 8. Environment variables — complete inventory

From `app/core/config.py` (14 fields; no others exist anywhere):

| Var | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg2://videocontrol:videocontrol@localhost:5432/videocontrol` | SQLAlchemy DB URL |
| `MODEL_NAME` | `google/siglip2-so400m-patch14-384` | HF checkpoint for both image and text towers |
| `MEDIA_ROOT` | `./data/media` | uploaded source videos |
| `FRAME_ROOT` | `./data/frames` | extracted frame thumbnails |
| `CLIP_ROOT` | `./data/clips` | FFmpeg-generated clips |
| `BATCH_SIZE` | `8` | frames per SigLIP-2 forward pass |
| `FRAME_DIFF_THRESHOLD` | `0.08` | normalised mean-abs-diff keep threshold |
| `MIN_FRAME_INTERVAL_SECONDS` | `0.5` | minimum gap between retained frames |
| `MAX_FRAME_INTERVAL_SECONDS` | `3.0` | forced-keep interval |
| `MAX_UPLOAD_MB` | `2048` | post-write upload size cap |
| `TEMPORAL_GROUP_WINDOW` | `3.0` | seconds; grouping window in `search_service._temporal_group` |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | comma-separated allow-list |
| `LOG_LEVEL` | `INFO` | root log level |
| `HF_HOME` | `./data/models` | HF cache dir |

The on-disk `.env` overrides `DATABASE_URL` to `postgresql+psycopg2://postgres:viki2006@localhost:5432/ctrlfinder` and `HF_HOME=./data/models`; the rest match the defaults. `.env` is git-ignored.

**All eight `GRAPH_*` / weighting variables named in the prompt are new.** `TEMPORAL_GROUP_WINDOW` already exists and is the natural neighbour of `TEMPORAL_WINDOW_SECONDS`, but they are semantically different (result-grouping window vs. temporal-proximity scoring window) and must remain separate settings.

---

## 9. Tests

| File | What it actually tests | Honest assessment |
|---|---|---|
| `tests/__init__.py` | empty | makes `tests` a package so pytest inserts `backend/` on `sys.path` |
| `test_health.py` | `TestClient(app).get("/health")` → 200, `{"status":"ok"}` | Real, but transitively imports **the entire app**, i.e. `pgvector`, `torch`, `transformers` |
| `test_schemas.py` | `SearchRequest` min-length and `top_k` bounds | Real unit test, no I/O |
| `test_adaptive_extraction.py` | **only** `test_config_defaults` — asserts 4 settings are positive/ordered | **Misnamed.** It never calls `extract_adaptive_keyframes`. There is **zero** test coverage of the keyframe heuristic. |
| `test_evaluate.py` (untracked, modified working tree) | loads `experiments/evaluate.py` by `importlib` path and checks `compare_systems` metrics | Real, stdlib-only, no app import |

- **No `conftest.py`. No fixtures. No DB integration tests. No mocking layer.** Adding a `conftest.py` is new but harmless and needed for Phase 4.
- Runner: `pytest`; `Makefile` target `test:` → `cd backend && . .venv/bin/activate 2>/dev/null; pytest -v`. Note `backend/.venv` **does not exist**.
- `.pytest_cache/v/cache/nodeids` records 7 passing tests under `backend/tests/…` paths, so the suite has been run before — but not in the current environment (see §10).

---

## 10. Environment audit — **the critical section**

Every row below was measured, not assumed.

### 10.1 Python

| Finding | Evidence |
|---|---|
| `python` on `PATH` is **broken** | `D:\Scripts\python.exe` → `failed to locate pyvenv.cfg` |
| Two interpreters exist | `py -0p` → `-V:3.13 *  D:\python.exe`, `-V:3.12  C:\Users\vikik\AppData\Local\Programs\Python\Python312\python.exe` |
| `D:\python.exe` reports a **malformed `site-packages`** | `site.getsitepackages()` → `['D:', 'D:Lib\\site-packages']` (missing separator) |
| Consequence: **no third-party package is importable by default** | with no `PYTHONPATH`, `import sqlalchemy` / `fastapi` / `numpy` / `psycopg2` all raise `ModuleNotFoundError`, despite the packages existing under `D:\Lib\site-packages` |
| Workaround verified | `$env:PYTHONPATH="D:\Lib\site-packages"` → `numpy`, `sqlalchemy`, `fastapi`, `pydantic_settings`, `psycopg2`, `uvicorn`, `httpx`, `starlette` all import **OK** |
| `D:\python.exe` has no `pip` module of its own | `python -m pip --version` → `No module named pip` (pip *is* present in `D:\Lib\site-packages`; it resolves once `PYTHONPATH` is set — confirmed by a successful `pip download`) |
| **Python 3.12 install is empty** | `numpy`, `sqlalchemy`, `fastapi`, `pgvector`, `torch`, `cv2`, `matplotlib`, `pandas`, `pytest` all missing |
| **Network access to PyPI works** | `pip download pytest` → `Saved … pytest-9.1.1-py3-none-any.whl` |

### 10.2 Missing Python packages required by this project

| Package | Needed by | In `D:\Lib\site-packages`? | Installable? |
|---|---|---|---|
| `pgvector` | `app/models/frame.py` (module-level import) | **No** | yes (pure Python) |
| `torch` | `app/services/model_service.py` | **No** | yes, ~2.5 GB |
| `transformers` | `app/services/model_service.py` | **No** | yes |
| `opencv-python-headless` | `embedding_service`, `video_service` | **No** | yes |
| `Pillow` | `embedding_service` | **No** | yes |
| `pytest` | test suite | **No** | yes |
| `matplotlib` | Phase 3 figures | **No** | yes |
| `pandas` | Phase 3 CSV/JSON handling | **No** | yes |
| `alembic` | Phase 1 migrations | **No** | yes |

**Net effect: the backend cannot be imported at all in this environment.** `import app.main` → `api.v1.router` → `routers.search` → `controllers.search_controller` → `services.search_service` → `models.frame` → `import pgvector.sqlalchemy` → **ModuleNotFoundError**. This means `tests/test_health.py` cannot currently run locally.

### 10.3 PostgreSQL

| Finding | Evidence |
|---|---|
| Server **is running** | `Get-Service postgresql-x64-18` → `Running`; TCP 5432 reachable |
| Version | `PostgreSQL 18.3 on x86_64-pc-windows-msvc, compiled by msvc-19.44.35223, 64-bit` |
| Credentials in `.env` **work** | `PGPASSWORD=viki2006 psql -h localhost -U postgres` connects |
| **pgvector is NOT installed** | `SELECT … FROM pg_available_extensions WHERE name ILIKE '%vector%'` → **0 rows**. No `vector.control` anywhere under `C:\Program Files\PostgreSQL\18` |
| Therefore `CREATE EXTENSION vector` **will fail** | and `video_frames.embedding vector(1152)` cannot be created |
| The `.env` target database is **completely empty** | `\dt` in `ctrlfinder` → *"Did not find any tables."* Only `plpgsql` is installed. |
| MSVC toolchain is present | `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools` (would make a source build of pgvector possible) |
| `ffmpeg` / `ffprobe` **not on PATH** | not found. Present in `backend/Dockerfile`, absent natively. → `POST /api/v1/search/clip/{id}` cannot succeed natively. |

**Consequence:** *no end-to-end run is currently possible in this environment.* No frame can be embedded (no torch/transformers), and no table can be created (no pgvector).

### 10.4 Real data that *is* available

| Asset | Detail |
|---|---|
| **A real 92 MB video** | `backend/data/media/e11a89c8-37bf-41b6-be8b-bceb9b0096da.mp4`, 92,123,729 bytes, **git-tracked** (`git ls-files` confirms) |
| Note | `MEDIA_ROOT` is `./data/media` (project root), so this file sits **outside** the configured media root. `data/media`, `data/frames`, `data/clips`, `data/models` at the root are all **empty**. |
| `data/models` | empty — SigLIP-2 has **not** been downloaded |
| Frontend | `node` v24.15.0 available; `frontend/node_modules` **not installed** |

### 10.5 Git state

`main`, 4 commits (`0b02cba` "Edited the model" is HEAD). Working tree:
```
 M README.md
 M experiments/evaluate.py
?? backend/tests/test_evaluate.py
```
(Note: the tracked path is `README.md`; the on-disk name reads `Readme.md` — a Windows case-insensitivity artifact. `README.md` is **not** in `.gitignore`, and `experiments/results/*.csv|json|png` are **not** ignored either.)

---

## 11. What already exists and will be reused **as-is**

| # | Asset | Why it is reused unchanged |
|---|---|---|
| 1 | `app/services/model_service.py` | Working SigLIP-2 singleton, correct 1152-d L2-normalised image **and** text towers. Graph work needs both. |
| 2 | `app/services/embedding_service.extract_adaptive_keyframes` | The adaptive heuristic *is* the project's named contribution. Reused verbatim; the graph builder consumes its output. |
| 3 | `app/services/embedding_service.process_and_store_embeddings` | Thumbnail naming (`{frame_number:08d}.jpg`, 320×180 q85) and `VideoFrame` insertion are reused; the graph builder is bolted on *after* it. |
| 4 | `app/models/video.py`, `app/models/frame.py` | `VideoFrame.id` becomes `graph_nodes.frame_id`; `Video.status` already has a `failed` state. |
| 5 | `app/db/session.py` `get_db` / `SessionLocal` | Same session/transaction machinery the graph writer will use. |
| 6 | `app/schemas/search.py` | `SearchRequest` / `SearchResult` / `SearchResponse` are frozen inputs to the ablation baseline. |
| 7 | `app/services/search_service.search` | The **vector-only baseline** for the ablation. It must keep working byte-for-byte. |
| 8 | `app/services/clip_service.generate_clip` | Downstream of every result, graph or not. |
| 9 | `app/api/v1/router.py` include pattern | One added `include_router` line is the only required integration edit. |
| 10 | `docker-compose.yml` | Untouched; new tables arrive via migration, not a new service. |
| 11 | `frontend/src/api.ts` axios client + `mediaUrl` | New functions appended; existing exports untouched. |
| 12 | `app/core/config.py` pattern | New settings follow the same `BaseSettings` style. |

## 12. What existing code will be **modified**, and exactly why

Only five files, each for a stated integration reason. No behavioural change to any existing contract.

| # | File | Minimal change | Why unavoidable |
|---|---|---|---|
| 1 | `app/core/config.py` | **Append** 8–10 new fields with defaults | Central config surface; the prompt mandates env-driven weights. Additive only — no existing field touched. |
| 2 | `app/models/__init__.py` | **Append** `GraphNode`, `GraphEdge` to imports and `__all__` | This is the declared model registry; new models must be exported here or `Base.metadata` misses them. |
| 3 | `app/api/v1/router.py` | **Append** `include_router(graph.router, tags=["Graph"])` | The only place routers are registered. Purely additive — existing include order untouched. |
| 4 | `app/services/video_service.py` | Call the graph builder **after** `process_and_store_embeddings` succeeds, inside the existing background task | The prompt requires graph construction to be *part of* the existing indexing run, not a separate job. Insertion point is after frames exist, before `status="ready"`. Existing status transitions and error handling reused verbatim. |
| 5 | `frontend/src/api.ts` | **Append** `searchGraph()`, `getVideoGraph()` + 3 new interfaces | Sole API boundary; additive functions and types only, so `SearchResult`/`SearchResponse` stay byte-identical. |

`experiments/` is **replaced by a package** (new files) rather than edited: the existing `evaluate.py` solves a different problem (scoring supplied prediction files) and `test_evaluate.py` depends on it. `test_evaluate.py` must keep passing, so `evaluate.py` is left byte-identical and the new framework lives in `experiments/evaluation/`.

## 13. API contracts that must remain untouched

Frozen, byte-for-byte:

1. `GET /health` → `{"status": "ok"}`
2. `POST /api/v1/videos` → `VideoResponse`
3. `GET /api/v1/videos` → `VideoListResponse`
4. `GET /api/v1/videos/{video_id}` → `VideoResponse`
5. **`POST /api/v1/search`** → `SearchRequest` in, `SearchResponse` out, including `score`/`group_id`/`thumbnail_url`/`clip_url` field names and the `_temporal_group` behaviour
6. `POST /api/v1/search/clip/{frame_id}` → `{"clip_url": str}`
7. `videos` and `video_frames` **table shapes** — no migration may alter them
8. `VideoStatus` enum values `queued | processing | ready | failed`
9. All fourteen `Settings` field names and defaults
10. `frontend/src/api.ts` existing exports and the `SearchResult` interface

## 14. Features with no existing implementation → interface only, never a fake

| Requested capability | Status in repo | Build as |
|---|---|---|
| Object detection (YOLO / Grounding DINO) | **Absent** | `GraphExtractor` protocol + `NullGraphExtractor` default; returns `[]`. Never fabricated. |
| OCR / on-screen text | **Absent** | Same. |
| ASR / transcript nodes | **Absent** | Same. |
| Image captioning (BLIP / Florence) | **Absent** | Same. |
| VLM concept/concept extraction | **Absent** | Same. |
| Named-entity / concept linking | **Absent** | Same. |
| Embedding dimension other than 1152 | **Absent** | `MODEL_NAME` is configurable, so a SigLIP-1 checkpoint would be dimension 1152 too; but **no** second checkpoint has ever been run here, so the SigLIP-vs-SigLIP-2 table cannot be populated without downloading and running both. |

Consequently `graph_nodes.node_type` is limited to what real extraction supports: `VIDEO`, `SCENE`, `FRAME` — with `TEXT`/`CONCEPT`/`OBJECT` declared in the enum but **only ever written by a real extractor**. `PERSON`, `EVENT`, `ACTION`, `LOCATION` are **not** added, exactly as the prompt requires.

## 15. Full list of new files required

**Phase 1 — graph layer**
| File | Purpose |
|---|---|
| `backend/app/models/graph.py` | `GraphNode`, `GraphEdge` SQLAlchemy models |
| `backend/app/services/graph_extraction_service.py` | `GraphExtractor` protocol, `Detection` dataclass, `NullGraphExtractor` (documented no-op), `get_graph_extractor()` |
| `backend/app/services/graph_service.py` | Node/edge builders, scene grouping, similarity + temporal edges, bounded traversal, batched fetch |
| `backend/app/services/graph_ranking_service.py` | `vector_score`, `graph_score`, `temporal_score`, `confidence_score`, weighted combination, tie-breaking |
| `backend/app/schemas/graph.py` | `GraphSearchRequest/Response`, `GraphNodeOut`, `GraphEdgeOut`, `VideoGraphResponse` |
| `backend/app/controllers/graph_controller.py` | HTTP mapping for the two new endpoints |
| `backend/app/api/v1/routers/graph.py` | Route definitions |
| `backend/alembic.ini`, `backend/alembic/env.py`, `backend/alembic/script.py.mako`, `backend/alembic/versions/0001_graph_layer.py` | Real, versioned, non-destructive migration |

**Phase 2 — graph-aware retrieval** (folded into the above; no extra files beyond schemas/controller/router)

**Phase 3 — evaluation**
| File | Purpose |
|---|---|
| `experiments/__init__.py`, `experiments/evaluation/__init__.py`, `experiments/visualization/__init__.py`, `experiments/datasets/__init__.py` | Make `python -m experiments.evaluation.evaluate_retrieval` work |
| `experiments/evaluation/metrics.py` | Recall@K, Precision@K, MRR, MAP, NDCG@K, HitRate@K — documented formulas |
| `experiments/evaluation/graph_metrics.py` | Expanded-node counts, depth effects, edge-type distribution |
| `experiments/evaluation/ranking.py` | Ablation configurations A/B/C/D, depth sweeps |
| `experiments/evaluation/evaluate_retrieval.py` | CLI entrypoint |
| `experiments/evaluation/reporting.py` | CSV + JSON writers; Markdown table generated *from* the CSV/JSON |
| `experiments/evaluation/dataset.py` | Dataset loading + ground-truth parsing; strict about missing ground truth |
| `experiments/visualization/plot_recall.py`, `plot_precision.py`, `plot_mrr.py`, `plot_latency.py`, `plot_comparison.py`, `plot_ablation.py`, `plot_frame_reduction.py`, `plot_ndcg.py` | Eight figures, CSV/JSON-only inputs |
| `experiments/visualization/_common.py` | Shared matplotlib setup (`Agg` backend) and CSV readers |
| `experiments/results/README.md` | Explains which CSVs are measured vs. absent |
| `experiments/README.md` | Reproduction instructions |

**Phase 4 — tests**
| File | Purpose |
|---|---|
| `backend/tests/conftest.py` | `sys.path` + settings isolation; no DB required for unit tests |
| `backend/tests/test_graph_models.py` | Node/edge construction, cascade/FK metadata assertions |
| `backend/tests/test_graph_traversal.py` | `max_depth` / `max_nodes` caps, no-N+1 batch shape (**synthetic data, labelled**) |
| `backend/tests/test_graph_ranking.py` | Scoring components, combination, tie-breaking (**synthetic**) |
| `backend/tests/test_metrics.py` | Every metric vs. hand-computed expected values (**synthetic**) |
| `backend/tests/test_graph_extraction.py` | `NullGraphExtractor` returns `[]`; protocol conformance |

**Phase 5 — docs**
`README.md` (rewrite) · `docs/AUDIT_REPORT.md` (this file) · `docs/EVALUATION.md`

## 16. Phase 0 blocking findings — decisions required before Phase 1

| # | Finding | Impact | Recommendation |
|---|---|---|---|
| **B1** | **No `pgvector` extension in the local PostgreSQL 18** | `CREATE TABLE video_frames` is impossible → no indexing, no search, no real end-to-end evaluation, no real metric values | Build pgvector from source with the installed MSVC Build Tools, **or** run the stack in Docker (`pgvector/pgvector:pg16`, which bundles it) |
| **B2** | **No torch / transformers / opencv / Pillow / pgvector Python packages** | `import app.main` fails; the entire backend is unimportable and untestable locally | `pip install` them (network confirmed working). torch ≈ 2.5 GB, SigLIP-2 ≈ 1.5 GB more |
| **B3** | `python` on PATH is broken; `D:\python.exe` mis-resolves `site-packages` | Every command needs `$env:PYTHONPATH="D:\Lib\site-packages"` or a proper venv | Create a real venv for the project |
| **B4** | **No usable ground truth** — `evaluation_dataset.csv` has 5 queries with empty `video_id` *and* empty `ground_truth_timestamp` | Recall@K / NDCG@K / MAP are **undefined** until annotations exist. Fabricating them is forbidden | Ship the metrics with a strict validator that reports `N/A`, and provide the annotation template; real values only if a labelled set is produced |
| **B5** | No second embedding checkpoint has ever been run | SigLIP vs SigLIP-2 table is empty | Report `N/A — not evaluated` unless both checkpoints are actually indexed and scored |
| **B6** | No matplotlib / pandas | Phase 3 figures cannot render | `pip install` |
| **B7** | `ffmpeg` absent natively | Clip generation untestable locally (works in Docker) | Use Docker, or install FFmpeg |
| **B8** | `experiments/` is not a Python package | The prompt's CLI form `python -m experiments.evaluation.evaluate_retrieval` fails | Add `__init__.py` files (listed in §15) |
| **B9** | `init.sql` cannot alter an already-initialised DB; no migration tool | Graph tables would never reach an existing deployment | Add Alembic, additive only (§3.3) |
| **B10** | `experiments/results/` and `README.md` are not git-ignored, but `data/` is not ignored either while a 92 MB mp4 is tracked | Committing generated CSV/JSON/PNG could bloat the repo | Ignore generated artefacts explicitly; keep `.gitkeep` |

**Per prompt §0, Phases 1–5 are not started until B1–B10 are acknowledged.** B1/B2/B3 are environment decisions, not code decisions: the *code* can be written and unit-tested without them, but every "real measured number" in the acceptance checklist depends on B1+B2 being resolved.

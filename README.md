# VideoCtrl-F: An Adaptive Keyframe and SigLIP-2 Based Semantic Video Retrieval Framework

**VideoCtrl-F** is a multimodal semantic video search engine. Upload long videos and search their visual content with natural-language queries such as:

- “person wearing a yellow jacket”
- “person walking past a red car”
- “speaker drawing a diagram on a whiteboard”
- “a presentation slide showing a bar chart”
- “person writing mathematical equations”

The system returns ranked timestamps, thumbnails, and short contextual video clips.

---

## Features

- Adaptive keyframe extraction (removes visually redundant frames)
- SigLIP-2 vision + text encoders (shared multimodal embedding space)
- L2-normalized 1152-D embeddings
- PostgreSQL + pgvector with HNSW index
- Cosine-similarity search
- Temporal result grouping
- 5-second FFmpeg contextual clips
- Modern React + Tailwind search UI with timeline heatmap
- Background processing (FastAPI BackgroundTasks, migration-ready for Celery/RQ)
- Full Docker Compose stack
- Research evaluation scaffolding (Recall@K, MRR, MAP, latency, storage reduction)

---

## Architecture

```
                 ┌────────────────────────┐
                 │     React Frontend     │
                 │ Upload / Search /      │
                 │ Player / Results       │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │      FastAPI API       │
                 └───────────┬────────────┘
                             │
                 ┌───────────┴────────────┐
                 │                        │
                 ▼                        ▼
       ┌──────────────────┐     ┌──────────────────┐
       │ Video Processing │     │ Semantic Search  │
       └────────┬─────────┘     └────────┬─────────┘
                │                        │
                ▼                        ▼
       ┌──────────────────┐     ┌──────────────────┐
       │ Adaptive Frame   │     │ SigLIP-2 Text    │
       │ Extraction       │     │ Encoder          │
       └────────┬─────────┘     └────────┬─────────┘
                │                        │
                ▼                        │
       ┌──────────────────┐               │
       │ SigLIP-2 Vision  │               │
       │ Encoder          │               │
       └────────┬─────────┘               │
                │                        │
                └────────────┬───────────┘
                             ▼
                 ┌────────────────────────┐
                 │ PostgreSQL + pgvector  │
                 │ 1152-D Embeddings      │
                 │ HNSW Index             │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │ Ranked Search Results  │
                 │ + FFmpeg Clips         │
                 └────────────────────────┘
```

Backend layers strictly follow:

```
Router → Controller → Service → Database / AI / Processing
```

---

## Technology Stack

| Layer        | Technologies                                      |
|--------------|---------------------------------------------------|
| Frontend     | React, Vite, TypeScript, Tailwind CSS, Axios, Lucide |
| Backend      | Python, FastAPI, Uvicorn, Pydantic, SQLAlchemy    |
| Database     | PostgreSQL 16 + pgvector (HNSW)                   |
| AI           | Hugging Face Transformers, SigLIP-2 (`google/siglip2-so400m-patch14-384`) |
| Vision       | OpenCV (adaptive keyframes), Pillow               |
| Clips        | FFmpeg                                            |
| Deployment   | Docker Compose                                    |

---

## Installation

### Option A – Native (no Docker) ← **use this if you don’t have Docker**

Full step-by-step guide: **[NATIVE_SETUP.md](NATIVE_SETUP.md)**

Quick summary:

1. Install Python 3.11+, Node 18+, FFmpeg, PostgreSQL + **pgvector**
2. Create DB user/database and enable the `vector` extension
3. `cp .env.example .env`  (already configured for localhost)
4. Backend:
   ```bash
   cd backend
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   psql -U videocontrol -d videocontrol -h localhost -f app/db/init.sql
   uvicorn app.main:app --reload --port 8000
   ```
5. Frontend (new terminal):
   ```bash
   cd frontend
   npm install
   VITE_API_URL=http://localhost:8000 npm run dev
   ```
6. Open http://localhost:5173

### Option B – Docker

```bash
cp .env.example .env
docker compose up --build
```

| Service   | URL                          |
|-----------|------------------------------|
| Frontend  | http://localhost:5173        |
| Backend   | http://localhost:8000        |
| API Docs  | http://localhost:8000/docs   |
| ReDoc     | http://localhost:8000/redoc  |
| Health    | http://localhost:8000/health |

Volumes persist: PostgreSQL data, uploaded media, frames, clips, and Hugging Face model cache.

---

## Environment Variables

See `.env.example`. Key variables:

| Variable                     | Description                          | Default                                      |
|------------------------------|--------------------------------------|----------------------------------------------|
| `DATABASE_URL`               | PostgreSQL connection string         | —                                            |
| `MODEL_NAME`                 | Hugging Face model ID                | `google/siglip2-so400m-patch14-384`          |
| `BATCH_SIZE`                 | Embedding batch size                 | 8                                            |
| `FRAME_DIFF_THRESHOLD`       | Adaptive keyframe difference         | 0.08                                         |
| `MIN_FRAME_INTERVAL_SECONDS` | Min seconds between kept frames      | 0.5                                          |
| `MAX_FRAME_INTERVAL_SECONDS` | Max seconds before forcing a frame   | 3.0                                          |
| `TEMPORAL_GROUP_WINDOW`      | Result grouping window (seconds)     | 3.0                                          |
| `MAX_UPLOAD_MB`              | Upload size limit                    | 2048                                         |
| `CORS_ORIGINS`               | Allowed origins                      | `http://localhost:5173`                      |

Never commit secrets.

---

## API Documentation

Interactive docs are available at `/docs` (Swagger) and `/redoc`.

### Main endpoints

| Method | Path                        | Description                          |
|--------|-----------------------------|--------------------------------------|
| POST   | `/api/v1/videos`            | Upload video (multipart)             |
| GET    | `/api/v1/videos`            | List videos                          |
| GET    | `/api/v1/videos/{id}`       | Get video metadata                   |
| POST   | `/api/v1/search`            | Semantic search                      |
| POST   | `/api/v1/search/clip/{id}`  | Generate 5 s contextual clip         |
| GET    | `/health`                   | Health check                         |

**Search request example**

```json
{
  "query": "person writing on a whiteboard",
  "video_id": null,
  "top_k": 10,
  "min_score": 0.0
}
```

---

## Video Processing Pipeline

1. Upload → validation (extension, size, integrity)
2. Create DB record (`status=queued`)
3. Background task starts (`status=processing`)
4. Adaptive keyframe extraction (OpenCV)
5. Batch SigLIP-2 image encoding + L2 normalization
6. Store embeddings + thumbnails in PostgreSQL / filesystem
7. `status=ready`

### Adaptive Keyframe Extraction

Frames are retained when:

- visual difference ≥ `FRAME_DIFF_THRESHOLD`, **or**
- maximum temporal interval (`MAX_FRAME_INTERVAL_SECONDS`) is reached.

This reduces redundant frames while guaranteeing coverage of long static segments.

---

## SigLIP-2 Pipeline

- Model loaded once (singleton) and reused.
- Automatic device selection: CUDA if available, else CPU.
- Official Hugging Face processor for both images and text.
- Embeddings are L2-normalized before storage / search.
- Vector dimension: **1152**.

---

## PostgreSQL / pgvector Setup

`init.sql` enables the `vector` extension and creates an HNSW index:

```sql
CREATE INDEX ... ON video_frames
USING hnsw (embedding vector_cosine_ops);
```

Search uses cosine distance (`<=>`).

---

## Search Algorithm

1. Encode query with SigLIP-2 text encoder + L2 normalize.
2. HNSW approximate nearest-neighbor search.
3. Convert distance → similarity (`1 - distance`).
4. Temporal grouping of nearby timestamps.
5. Return top-K ranked results with thumbnail URLs.

---

## Evaluation & Research Reproducibility

```
experiments/
├── evaluation_dataset.csv   # query, video_id, ground_truth_timestamp
├── evaluate.py              # Recall@K, MRR, MAP, latency
├── compare_siglip.py        # SigLIP vs SigLIP-2 fair comparison
├── plot_results.py          # Matplotlib figures from measured CSVs
└── results/                 # empty – no fabricated numbers
```

**Metrics**

- Recall@1 / @5 / @10
- Precision@K
- MRR, MAP
- Query latency
- Indexing time
- Frames retained ratio
- Storage reduction

Populate `evaluation_dataset.csv` with real ground truth, run the live system, then execute the evaluation scripts. **No results are invented.**

---

## Testing

```bash
cd backend
pytest -v
```

Tests cover health endpoint, schema validation, and configuration of adaptive parameters.

---

## Troubleshooting

| Issue                        | Suggestion                                              |
|------------------------------|---------------------------------------------------------|
| Model download slow          | First run caches under `/models` volume                 |
| CUDA OOM                     | Lower `BATCH_SIZE`                                      |
| No search results            | Ensure video status is `ready`; try broader query       |
| FFmpeg errors                | Confirm FFmpeg is installed in the backend image        |
| CORS errors                  | Check `CORS_ORIGINS` in `.env`                          |

---

## Production Deployment Considerations

- Replace local filesystem with S3 / MinIO for media.
- Move background jobs to Celery + Redis or RQ.
- Add authentication / rate limiting.
- Use a production ASGI server (Gunicorn + Uvicorn workers).
- Monitor GPU memory and index size.
- Consider object detection / OCR / audio embeddings as future modalities.

---

## Limitations

- Frame-level embeddings do not provide full temporal reasoning.
- Visually similar scenes can produce false positives.
- Fine-grained actions may need temporal models.
- Audio and speech transcripts are not indexed.
- OCR is not independently indexed.
- Local filesystem storage is suitable for development; object storage is preferable at scale.

---

## Future Scope

Designed for easy extension:

- Audio embeddings / Whisper speech-to-text
- OCR
- CLIP / SigLIP re-ranking
- Temporal Transformers / video-language models
- Object detection, face/person tracking, scene segmentation
- Hybrid text + visual search
- Distributed GPU inference
- Redis / Celery workers
- S3 / MinIO storage

---

## Research Novelty

Semantic video retrieval itself is not new. The contribution of this framework is the integrated pipeline:

```
Adaptive Keyframe Selection
+ SigLIP-2 Multimodal Representation
+ L2 Normalization
+ Vector Database Retrieval (pgvector HNSW)
+ Temporal Grouping
+ Timestamp-Aware Clip Generation
```

Adaptive selection demonstrably reduces the number of processed frames, inference load, embedding storage, and index size while aiming to preserve retrieval quality.

---

## Team

| Name              | ID       |
|-------------------|----------|
| Vignesh K         | 23AM069  |
| Yogaazhaki S      | 23AM072  |
| Deepika S         | 23AM016  |

**Project duration:** 8 Months

---

## License

This repository is provided for academic and research use.

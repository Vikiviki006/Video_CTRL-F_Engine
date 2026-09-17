# VideoCtrl-F – Native Setup (No Docker)

Follow these steps if you do **not** have Docker.

---

## 1. System Requirements

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.11+ | 3.10 may work, 3.11 recommended |
| Node.js | 18 or 20 | for the React frontend |
| FFmpeg | any recent | required for clip generation |
| PostgreSQL | 14+ | must have the **pgvector** extension |
| GPU (optional) | CUDA | speeds up SigLIP-2; CPU works but is slower |

### Install system packages (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip \
  nodejs npm ffmpeg postgresql postgresql-contrib \
  libpq-dev build-essential
```

### Install system packages (macOS with Homebrew)

```bash
brew install python@3.11 node ffmpeg postgresql@16
brew services start postgresql@16
```

---

## 2. Install pgvector

pgvector is required for the vector index.

**Ubuntu / Debian (if package available):**
```bash
sudo apt install postgresql-16-pgvector   # adjust version number
```

**From source (works everywhere):**
```bash
git clone --branch v0.7.4 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

Then enable it in your database (see step 3).

---

## 3. Create PostgreSQL database

```bash
# Switch to postgres user (Linux) or use your local superuser
sudo -u postgres psql
```

Inside `psql`:

```sql
CREATE USER videocontrol WITH PASSWORD 'videocontrol';
CREATE DATABASE videocontrol OWNER videocontrol;
\c videocontrol
CREATE EXTENSION IF NOT EXISTS vector;
GRANT ALL ON SCHEMA public TO videocontrol;
\q
```

(On macOS you can usually run `psql postgres` directly as your user.)

---

## 4. Project setup

```bash
# Unpack the zip if you haven't already
cd VideoCtrlF

# Create local .env from the example
cp .env.example .env

# Create data directories
mkdir -p data/media data/frames data/clips data/models
```

The default `.env` already points to:
- `DATABASE_URL=...@localhost:5432/videocontrol`
- local `./data/...` folders

---

## 5. Backend

```bash
cd backend

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies (this will download PyTorch + transformers)
pip install --upgrade pip
pip install -r requirements.txt

# Initialise the database schema
# (the init.sql is also run automatically by Docker; for native we run it once)
psql -U videocontrol -d videocontrol -h localhost -f app/db/init.sql

# Start the API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

First request that needs the model will download **SigLIP-2** (~1–2 GB) into `./data/models`.  
This can take several minutes depending on your internet.

Check health: http://localhost:8000/health  
API docs: http://localhost:8000/docs

---

## 6. Frontend (new terminal)

```bash
cd frontend

npm install
# Tell the frontend where the backend lives
export VITE_API_URL=http://localhost:8000

npm run dev
```

Open: http://localhost:5173

---

## 7. Optional – Makefile helpers

If you still want short commands:

```bash
# from project root
make backend    # starts uvicorn (after you activate venv)
make frontend   # starts Vite
```

(You can add these targets yourself; they are not required.)

---

## 8. Common issues

| Problem | Fix |
|---------|-----|
| `psycopg2` / `libpq` errors | `sudo apt install libpq-dev` then reinstall `psycopg2-binary` |
| `ModuleNotFoundError: torch` | Make sure the venv is activated and `pip install -r requirements.txt` finished |
| Model download fails / slow | Set `HF_HOME` to a fast disk; or pre-download with `huggingface-cli download google/siglip2-so400m-patch14-384` |
| FFmpeg not found | Install FFmpeg and ensure it is on `PATH` (`ffmpeg -version`) |
| CORS errors in browser | Confirm `CORS_ORIGINS` in `.env` contains `http://localhost:5173` |
| CUDA OOM | Lower `BATCH_SIZE=2` or `4` in `.env` |
| “relation video_frames does not exist” | Re-run `psql ... -f app/db/init.sql` |

---

## 9. Minimal smoke test

1. Open http://localhost:5173
2. Upload a short `.mp4` (30–60 s is enough for testing)
3. Wait until status becomes **ready** (refresh the library)
4. Type a query related to the video content
5. Click a result → video should jump to that timestamp

---

## Hardware note

- **CPU only**: works, but embedding a long video can take many minutes.
- **GPU (CUDA)**: strongly recommended for anything longer than a few minutes.

Enjoy searching your videos!

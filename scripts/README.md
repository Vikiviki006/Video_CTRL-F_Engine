# Phase-0 environment bootstrap scripts

Helper scripts written during the repository audit. They are idempotent and
document what they do and why.

| Script | Purpose | Status |
|---|---|---|
| `bootstrap_env.bat` | Creates/uses `D:\VideoCtrlF\.venv` and installs every dependency needed by Phases 1-5. | **Works.** Needs Python 3.12 (the repo's pins, e.g. `numpy==1.26.4`, have no 3.13 wheels). |
| `bootstrap_env2.bat` | Installs `torch` + `transformers`. | **Works, with a caveat** - see below. |
| `build_pgvector.bat` | Builds and installs the `pgvector` PostgreSQL extension from source. | **Blocked on this machine** - see below. |

## torch index caveat

`bootstrap_env.bat` first tries `download.pytorch.org` for the CUDA wheel. On
this network that host does not resolve, so it falls back to PyPI, which serves
a **CPU-only** build (`torch 2.14.0+cpu`). The installed runtime is therefore
CPU-only and `torch.cuda.is_available()` is `False`, even though an RTX 3050 is
present. That is acceptable: the GPU has 4 GB of VRAM, which is not enough for
`google/siglip2-so400m-patch14-384` at fp32 anyway.

## pgvector is the hard blocker

`video_frames.embedding` is a `vector(1152)` column, so the schema cannot be
created without the `pgvector` extension. On this machine:

* `SELECT * FROM pg_available_extensions WHERE name ILIKE '%vector%'` returns
  **0 rows** for the local PostgreSQL 18.3 install.
* `C:\Program Files\PostgreSQL\18\share\extension` is **not writable** (the
  shell runs without administrator rights).
* **No C compiler is present** - no `cl.exe`, no `gcc`, no `clang`; Visual
  Studio Build Tools 18 is installed but without the C++ workload.
* **Docker is not installed.**
* WSL2 Ubuntu 24.04 is present, but `sudo` requires a password that is not
  available non-interactively.

Consequently `build_pgvector.bat` cannot complete here. Any one of the
following unblocks it, and nothing else in the project needs to change:

1. Install the **C++ build tools** workload for Visual Studio Build Tools 18,
   run this shell as Administrator, then:
   ```bat
   scripts\build_pgvector.bat
   ```
2. Install **Docker Desktop**, then bring the stack up with
   `docker compose up --build` - the `pgvector/pgvector:pg16` image already
   bundles the extension.
3. Provide the WSL password so the extension can be installed with
   `sudo apt install postgresql-16-pgvector` inside Ubuntu.
4. Point `DATABASE_URL` at any PostgreSQL instance that already has pgvector.

Once the extension exists:

```bat
psql -h localhost -U postgres -d ctrlfinder -f backend\app\db\init.sql
cd backend && alembic upgrade head
```

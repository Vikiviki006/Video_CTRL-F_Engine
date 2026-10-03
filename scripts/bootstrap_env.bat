@echo off
REM Phase-0 environment bootstrap for VideoCtrl-F.
REM Creates a working venv and installs every dependency needed by Phases 1-5.
REM Idempotent: safe to re-run.

set VENV=D:\VideoCtrlF\.venv
set PY=%VENV%\Scripts\python.exe

echo === [1/4] core web + db stack ===
%PY% -m pip install --upgrade pip setuptools wheel
%PY% -m pip install ^
  "fastapi==0.115.0" ^
  "uvicorn[standard]==0.30.6" ^
  "pydantic==2.9.2" ^
  "pydantic-settings==2.5.2" ^
  "sqlalchemy==2.0.35" ^
  "psycopg2-binary==2.9.9" ^
  "pgvector==0.3.4" ^
  "python-multipart==0.0.9" ^
  "python-dotenv==1.0.1" ^
  "httpx==0.27.2" ^
  "aiofiles==24.1.0" ^
  "numpy==1.26.4" ^
  "Pillow==10.4.0" ^
  "opencv-python-headless==4.10.0.84" ^
  "alembic==1.13.3" ^
  "matplotlib==3.9.2" ^
  "pandas==2.2.3" ^
  "pytest==8.3.3" ^
  "pytest-asyncio==0.24.0"
if errorlevel 1 exit /b 1

echo === [2/4] torch (CUDA 12.4 wheel) ===
%PY% -m pip install torch --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
  echo CUDA torch install failed - falling back to CPU wheel
  %PY% -m pip install torch --index-url https://download.pytorch.org/whl/cpu
)
if errorlevel 1 exit /b 1

echo === [3/4] transformers ===
%PY% -m pip install "transformers>=4.45.0" "accelerate>=0.34.0" "safetensors>=0.4.5"
if errorlevel 1 exit /b 1

echo === [4/4] verify ===
%PY% -c "import importlib;[print(('OK  ' if importlib.util.find_spec(m) else 'MISS'),m) for m in ['fastapi','sqlalchemy','pgvector','psycopg2','torch','transformers','cv2','PIL','numpy','matplotlib','pandas','alembic','pytest']]"
echo === BOOTSTRAP COMPLETE ===

@echo off
REM Phase-0 environment bootstrap, part 2: torch + transformers.
REM download.pytorch.org does not resolve in this environment, so torch is
REM taken from PyPI instead (which does resolve). The PyPI Windows wheel is
REM the CUDA build and is large, but it is the only reachable source.

set PY=D:\VideoCtrlF\.venv\Scripts\python.exe

echo === transformers (small) ===
%PY% -m pip install "transformers>=4.45.0" "safetensors>=0.4.5" "huggingface-hub>=0.25.0"
if errorlevel 1 exit /b 1

echo === torch from PyPI (large, ~2.5GB) ===
%PY% -m pip install --retries 10 --timeout 120 torch
if errorlevel 1 (
  echo PyPI torch install failed
  exit /b 1
)

echo === verify ===
%PY% -c "import torch; print('torch', torch.__version__, 'cuda_available=', torch.cuda.is_available())"
%PY% -c "import transformers; print('transformers', transformers.__version__)"
echo === BOOTSTRAP PART 2 COMPLETE ===

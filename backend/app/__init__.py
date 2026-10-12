# VideoCtrl-F Backend Application

import os
from pathlib import Path

# huggingface_hub resolves HF_HOME / HF_HUB_CACHE once, at import time, which
# happens long before pydantic builds Settings. Exporting them here - before
# anything imports transformers - is what makes the HF_HOME from .env actually
# take effect. Without it the model cache silently falls back to
# ~/.cache/huggingface, which is fatal on a machine whose system drive is full.
_HF_ENV_KEYS = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "HF_HUB_DISABLE_XET",
    "HF_TOKEN",
)


def _export_hf_env() -> None:
    """Copy the HF_* cache settings from .env into os.environ."""
    for env_file in (Path(".env"), Path("../.env")):
        if not env_file.is_file():
            continue
        try:
            lines = env_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key in _HF_ENV_KEYS:
                os.environ[key] = value.strip()
        break


_export_hf_env()

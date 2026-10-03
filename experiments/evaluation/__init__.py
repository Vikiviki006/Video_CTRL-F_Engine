"""Evaluation framework for VideoCtrl-F.

Imports the application packages so that ``python -m experiments.evaluation.*``
works from the repository root. The backend is added to ``sys.path`` because
``app`` lives under ``backend/`` and is not installed as a package.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _REPO_ROOT / "backend"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

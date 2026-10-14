"""Shared pytest fixtures for VideoCtrl-F.

Two jobs:

1. **Make the package importable.** ``app`` lives in ``backend/`` and tests
   live in ``backend/tests/``. pytest's rootdir insertion normally handles
   this because ``tests/`` is a package, but being explicit means the suite
   runs identically from the repo root and from ``backend/``.

2. **Isolate settings.** The suite must not silently depend on a developer's
   local ``.env`` (a real ``DATABASE_URL``, a real model name). Every test
   runs against explicitly constructed settings, and any test that would touch
   a database is skipped unless it opts in via ``--run-db-tests``.

Database-backed tests
---------------------
The graph layer's tables reference ``videos`` and ``video_frames``, and
``video_frames.embedding`` is a ``vector(1152)`` column, so creating the schema
requires the pgvector extension. Where that is unavailable those tests skip
with an explicit reason rather than failing or silently passing. The pure-logic
tests - traversal bounds, scoring, metrics - need no database at all and always
run.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# ``tests/test_metrics.py`` and ``tests/test_reporting_and_plots.py`` import
# ``experiments.*``, which lives in the *repo root*, not in ``backend/``. Running
# pytest from ``backend/`` therefore could not resolve it, so the two evaluation
# modules failed at collection with ModuleNotFoundError. Adding the repo root is
# the same fix as above, one level up, and keeps the "runs identically from the
# repo root and from backend/" promise this module already makes for ``app``.
REPO_ROOT = BACKEND_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Settings isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    """Point every Settings-dependent module at deterministic test values.

    ``get_settings`` is ``lru_cache``d and most services capture ``settings`` at
    import time, so this clears the cache, patches the env, and re-imports the
    affected modules' ``settings`` attribute. It restores everything after.
    """
    from app.core import config as config_module

    defaults = {
        "GRAPH_MAX_DEPTH": "2",
        "GRAPH_MAX_NODES": "200",
        "GRAPH_SIMILARITY_THRESHOLD": "0.80",
        "TEMPORAL_WINDOW_SECONDS": "5.0",
        "GRAPH_SCENE_WINDOW_SECONDS": "3.0",
        "GRAPH_SIMILAR_TOP_K": "5",
        "GRAPH_MAX_EDGES_PER_VIDEO": "20000",
        "VECTOR_WEIGHT": "0.65",
        "GRAPH_WEIGHT": "0.20",
        "TEMPORAL_WEIGHT": "0.10",
        "CONFIDENCE_WEIGHT": "0.05",
        "GRAPH_ENABLED": "True",
        "GRAPH_EXTRACTOR": "null",
        "LOG_LEVEL": "WARNING",
    }
    for key, value in defaults.items():
        monkeypatch.setenv(key, value)

    config_module.get_settings.cache_clear()

    # Re-bind the module-level ``settings`` object in every service that
    # captured it at import time, so the patched env actually takes effect.
    import importlib

    for module_name in (
        "app.services.graph_service",
        "app.services.graph_ranking_service",
        "app.services.graph_extraction_service",
    ):
        module = importlib.import_module(module_name)
        monkeypatch.setattr(module, "settings", config_module.get_settings(), raising=False)

    yield

    config_module.get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Database opt-in
# ---------------------------------------------------------------------------


def _pgvector_available() -> bool:
    try:
        import psycopg2  # noqa: F401
        from pgvector.sqlalchemy import Vector  # noqa: F401
    except Exception:
        return False
    from app.core.config import get_settings

    try:
        import sqlalchemy

        engine = sqlalchemy.create_engine(
            get_settings().DATABASE_URL, connect_args={"connect_timeout": 3}
        )
        with engine.connect() as conn:
            from sqlalchemy import text

            conn.execute(text("SELECT 1"))
            has_vector = conn.execute(
                text(
                    "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
                )
            ).fetchone()
        engine.dispose()
        return has_vector is not None
    except Exception:
        return False


@pytest.fixture(scope="session")
def db_available() -> bool:
    """True only when a live database with the pgvector extension is reachable."""
    import os

    if os.environ.get("VIDEOCTRLF_RUN_DB_TESTS", "0") != "1":
        return False
    return _pgvector_available()


@pytest.fixture
def db_session(db_available):
    """A real Session, skipped with a reason when the database is unusable."""
    if not db_available:
        pytest.skip(
            "Database tests disabled. Enable with VIDEOCTRLF_RUN_DB_TESTS=1 and a "
            "reachable PostgreSQL that has the pgvector extension installed."
        )
    from app.db.session import SessionLocal

    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


# ---------------------------------------------------------------------------
# Synthetic graph objects
# ---------------------------------------------------------------------------
#
# Everything below builds clearly-labelled SYNTHETIC fixtures for unit tests.
# This is the only place in the project where made-up graph data is permitted,
# and it never reaches a production code path: the fixtures are assembled
# in-memory and never written to a database or to a results file.


def make_node(
    node_id: Optional[uuid.UUID] = None,
    node_type: str = "FRAME",
    video_id: Optional[uuid.UUID] = None,
    entity_key: Optional[str] = None,
    frame_id: Optional[uuid.UUID] = None,
    timestamp: Optional[float] = None,
    confidence: float = 1.0,
) -> "Any":
    """Build an unsaved, synthetic ``GraphNode``."""
    from app.models.graph import GraphNode

    node_id = node_id or uuid.uuid4()
    return GraphNode(
        id=node_id,
        video_id=video_id or uuid.uuid4(),
        node_type=node_type,
        entity_key=entity_key or f"synthetic:{node_id}",
        label=f"synthetic {node_type}",
        timestamp_start=timestamp,
        timestamp_end=timestamp,
        frame_id=frame_id,
        confidence=confidence,
        node_metadata={"synthetic": True},
    )


def make_edge(
    source: "Any",
    target: "Any",
    relationship: str = "SIMILAR_TO",
    weight: float = 1.0,
    confidence: float = 1.0,
) -> "Any":
    """Build an unsaved, synthetic ``GraphEdge`` between two nodes."""
    from app.models.graph import GraphEdge

    return GraphEdge(
        id=uuid.uuid4(),
        source_node_id=source.id,
        target_node_id=target.id,
        relationship=relationship,
        weight=weight,
        confidence=confidence,
        edge_metadata={"synthetic": True, "basis": "unit_test"},
    )


class ScriptedSession:
    """A Session stand-in that serves canned results and counts queries.

    ``graph_service.expand_from_seeds`` issues queries in a fixed, documented
    order: one to resolve seeds, then two per expansion level (edges, then
    nodes). Scripting that order lets the tests assert the properties that
    actually matter - that traversal is *bounded* and that it is *batched* -
    without needing a live database.

    ``execute_calls`` records every call, which is what the N+1 test asserts
    against.
    """

    def __init__(self, responses: Sequence[Any]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.execute_calls = 0

    def _next(self) -> Any:
        if self._index >= len(self._responses):
            raise AssertionError(
                f"ScriptedSession ran out of responses: asked for #{self._index + 1} "
                f"but only {len(self._responses)} were scripted. The traversal "
                f"issued more queries than the batching contract allows."
            )
        value = self._responses[self._index]
        self._index += 1
        return value

    def execute(self, statement: Any, *args: Any, **kwargs: Any) -> "_Result":
        self.execute_calls += 1
        return _Result(self._next())

    @property
    def unused_responses(self) -> int:
        return len(self._responses) - self._index


class _Result:
    def __init__(self, value: Any) -> None:
        self._value = value

    def mappings(self) -> "_Result":
        return self

    def scalars(self) -> "_Result":
        return self

    def all(self) -> List[Any]:
        if self._value is None:
            return []
        return list(self._value)


@pytest.fixture
def scripted_session_factory():
    """Factory for :class:`ScriptedSession` instances."""

    def _factory(responses: Sequence[Any]) -> ScriptedSession:
        return ScriptedSession(responses)

    return _factory

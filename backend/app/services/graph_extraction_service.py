"""Pluggable concept/object/text extraction for the relational graph.

Why this module exists
----------------------
The specification calls for ``DEPICTS`` / ``CONTAINS`` edges that come from a
real object detector, OCR engine, ASR model or captioner. **None of those
exist in this repository.** The audit (``docs/AUDIT_REPORT.md`` §14) verified
that there is no detector, no OCR, no speech recognition, no captioning model
and no VLM anywhere in the codebase, and the only embedding model present is
SigLIP-2, which produces vectors and not labels.

The correct engineering response is therefore an *extensible interface with an
honest default*, not a fabricated detector. This module defines:

* :class:`Detection`   - what an extractor must return.
* :class:`GraphExtractor` - the protocol every backend implements.
* :class:`NullGraphExtractor` - the default. Returns ``[]``. Always.
* :func:`get_graph_extractor` - selection hook driven by ``GRAPH_EXTRACTOR``.

The invariant this module enforces
----------------------------------
**No production code path may invent detection output.** ``NullGraphExtractor``
returns an empty list, so with the default configuration the graph contains
only nodes and edges derivable from real timestamps and real measured cosine
similarities. ``CONCEPT`` / ``OBJECT`` / ``TEXT`` node types exist in the
schema and stay empty until someone supplies a genuine extractor. There is no
code path that populates them with placeholder labels to make the interface
look exercised.

How to plug in a real backend
-----------------------------
Subclass :class:`GraphExtractor`, implement :meth:`GraphExtractor.extract`, and
register it in :func:`_EXTRACTOR_REGISTRY` under a new name, then set
``GRAPH_EXTRACTOR=<name>`` in the environment. No caller changes: the graph
builder only ever calls :meth:`GraphExtractor.extract` and handles an empty
result correctly. Candidate real backends, all of which fit the same
signature:

* **Object detection** - YOLO, Grounding DINO. Emit one ``Detection`` per box
  with ``node_type=OBJECT`` and the box in ``metadata["bbox"]``.
* **OCR** - Tesseract, PaddleOCR, EasyOCR. Emit ``node_type=TEXT`` with the
  recognised string as ``label``.
* **ASR / subtitles** - Whisper, or an SRT/VTT sidecar. Emit
  ``node_type=TEXT`` with ``timestamp_start``/``timestamp_end`` set to the
  spoken interval.
* **Captioning / VLM** - BLIP, Florence-2, Qwen-VL. Emit either an ``OBJECT``
  or a ``CONCEPT`` node with the caption as ``label``.

Because :class:`Detection` carries ``node_type``, ``label``, ``confidence``,
``timestamp_start``, ``timestamp_end`` and free-form ``metadata``, every one of
those backends slots in without a signature change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.models.graph import NodeType

logger = setup_logging()


@dataclass
class Detection:
    """One extracted entity observed in one frame.

    Attributes
    ----------
    node_type:
        Which :class:`~app.models.graph.NodeType` this observation becomes.
    label:
        Human-readable identity, e.g. ``"person"`` or ``"EXIT 4"``. For
        ``FRAME``/``SCENE`` rows this is usually left ``None``.
    confidence:
        The extractor's own score in ``[0, 1]``. Stored verbatim on the node.
    timestamp_start, timestamp_end:
        Temporal extent within the video. For a single-frame observation these
        equal that frame's timestamp.
    metadata:
        Free-form provenance. Real backends should record enough here to
        reproduce the observation, e.g. ``{"bbox": [x1, y1, x2, y2],
        "detector": "yolov8n", "detector_version": "..."}``.
    """

    node_type: NodeType
    label: Optional[str] = None
    description: Optional[str] = None
    confidence: float = 1.0
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.node_type, NodeType):
            # Accept the plain string form so callers need not import the enum.
            self.node_type = NodeType(self.node_type)
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(
                f"Detection.confidence must be in [0, 1], got {self.confidence}"
            )
        if (
            self.timestamp_start is not None
            and self.timestamp_end is not None
            and self.timestamp_end < self.timestamp_start
        ):
            raise ValueError(
                "Detection.timestamp_end must be >= timestamp_start "
                f"(got {self.timestamp_start} -> {self.timestamp_end})"
            )


@runtime_checkable
class GraphExtractor(Protocol):
    """Interface every extraction backend must satisfy.

    A backend is constructed with the resolved
    :class:`~app.core.config.Settings` and is then called once per retained
    keyframe during indexing. Implementations are expected to be expensive
    (model load, warm-up), so construction happens once per indexing run.
    """

    name: str

    def extract(
        self,
        image: Image.Image,
        timestamp_seconds: float,
        video_id: Any = None,
        frame_id: Any = None,
    ) -> List[Detection]:
        """Return every entity observed in ``image``.

        Parameters
        ----------
        image:
            RGB PIL image of the retained keyframe. The caller guarantees it
            is the exact image that was embedded, so detections and the frame
            embedding describe the same visual content.
        timestamp_seconds:
            Position of the frame in the source video. Backends that detect
            events over time may use it as an initial hint.
        video_id, frame_id:
            Optional context for provenance metadata.

        Returns
        -------
        list of :class:`Detection`
            Empty when the backend has nothing to report. Returning ``[]`` is
            always valid and is what the default backend always does.
        """
        ...


class NullGraphExtractor:
    """The default backend: reports nothing, honestly.

    This is not a stub pretending to work. It is the explicit statement that
    *this repository contains no object detector, OCR engine, ASR model or
    captioner*, so no ``OBJECT`` / ``TEXT`` / ``CONCEPT`` observations can be
    produced. It implements the full protocol, so callers exercise the same
    code path they would with a real backend, including the empty-result
    branch.

    Set ``GRAPH_EXTRACTOR`` to the name of a registered backend to change it.
    """

    name = "null"

    def __init__(self, settings=None) -> None:
        self.settings = settings or get_settings()
        logger.info(
            "GraphExtractor='null' - no object/OCR/ASR/caption model is "
            "available in this repository, so no OBJECT/TEXT/CONCEPT nodes or "
            "DEPICTS edges will be produced. Graph content is limited to VIDEO, "
            "SCENE and FRAME nodes with CONTAINS/BELONGS_TO/FOLLOWED_BY/"
            "PRECEDES/SIMILAR_TO edges derived from real timestamps and real "
            "measured cosine similarities."
        )

    def extract(
        self,
        image: Image.Image,
        timestamp_seconds: float,
        video_id: Any = None,
        frame_id: Any = None,
    ) -> List[Detection]:
        """Always returns ``[]``. See the class docstring for why."""
        return []


# Registry of available backends. The default entry is deliberately the honest
# no-op. Real backends are added here by whoever implements them; nothing else
# in the codebase needs to change.
_EXTRACTOR_REGISTRY: Dict[str, type] = {
    NullGraphExtractor.name: NullGraphExtractor,
}


def register_extractor(name: str, cls: type) -> None:
    """Register a backend so ``GRAPH_EXTRACTOR=<name>`` selects it.

    Intended for real implementations and for tests that need a deterministic
    stand-in. Tests must use obviously-synthetic detections; production code
    must not register a backend that fabricates output.

    Conformance is checked by structural inspection rather than
    ``issubclass(cls, GraphExtractor)``. ``GraphExtractor`` declares a ``name``
    attribute as well as a method, and a ``Protocol`` with non-method members
    cannot be used with ``issubclass`` at all - it raises ``TypeError``. Checking
    the two members directly is both valid and clearer about what is required.
    """
    if not isinstance(cls, type):
        raise TypeError(f"extractor backend must be a class, got {cls!r}")
    missing = [
        member
        for member in ("name", "extract")
        if not hasattr(cls, member)
    ]
    if missing or not callable(getattr(cls, "extract", None)):
        raise TypeError(
            f"{cls.__name__} does not satisfy the GraphExtractor protocol; "
            f"missing or non-callable: {', '.join(missing) or 'extract'}. "
            f"A backend must expose a `name` string and an `extract(...)` method."
        )
    _EXTRACTOR_REGISTRY[name] = cls
    logger.info(f"Registered GraphExtractor backend '{name}' -> {cls.__name__}")


def available_extractors() -> List[str]:
    """Names of every registered backend, for diagnostics and /health output."""
    return sorted(_EXTRACTOR_REGISTRY)


def get_graph_extractor(settings=None) -> GraphExtractor:
    """Return the configured extractor backend.

    Selection is by the ``GRAPH_EXTRACTOR`` setting. An unknown name is a hard
    error rather than a silent fallback: quietly degrading to the no-op
    extractor would hide a configuration mistake and make the graph look
    emptier than intended.
    """
    settings = settings or get_settings()
    name = getattr(settings, "GRAPH_EXTRACTOR", NullGraphExtractor.name)
    if name not in _EXTRACTOR_REGISTRY:
        raise ValueError(
            f"Unknown GRAPH_EXTRACTOR '{name}'. "
            f"Available: {', '.join(available_extractors())}. "
            f"Register a backend with graph_extraction_service.register_extractor()."
        )
    return _EXTRACTOR_REGISTRY[name](settings)


def extract_for_frames(
    extractor: GraphExtractor,
    frames: List[Dict[str, Any]],
) -> Dict[str, List[Detection]]:
    """Run ``extractor`` over a list of frame descriptors.

    Convenience wrapper used by the graph builder so that per-frame extraction
    errors are contained: a detector that fails on one frame yields no
    detections for that frame rather than aborting the whole indexing run.
    The failure is logged, never silently swallowed.

    Parameters
    ----------
    extractor:
        Any :class:`GraphExtractor`.
    frames:
        Dicts with at least ``image`` (PIL), ``timestamp_seconds``,
        ``frame_id``; ``video_id`` is optional.

    Returns
    -------
    dict
        Maps ``frame_id`` to the detections found for that frame. Frames with
        no detections, or whose extraction raised, are present with an empty
        list.
    """
    out: Dict[str, List[Detection]] = {}
    for spec in frames:
        frame_id = spec.get("frame_id")
        try:
            out[frame_id] = list(
                extractor.extract(
                    image=spec["image"],
                    timestamp_seconds=spec["timestamp_seconds"],
                    video_id=spec.get("video_id"),
                    frame_id=frame_id,
                )
                or []
            )
        except Exception as exc:  # noqa: BLE001 - one bad frame must not kill indexing
            logger.warning(
                f"Graph extraction failed for frame {frame_id} "
                f"(t={spec.get('timestamp_seconds')}): {exc}. "
                f"Continuing with no detections for this frame."
            )
            out[frame_id] = []
    return out


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity for a matrix of embeddings.

    Exposed here (rather than inlined in ``graph_service``) so that
    ``SIMILAR_TO`` edges have exactly one definition of "similarity" in the
    codebase: the same L2-normalised, inner-product identity that
    ``model_service`` relies on and that pgvector's ``vector_cosine_ops``
    operator implements server-side.

    Because SigLIP-2 embeddings are L2-normalised before storage, the dot
    product *is* the cosine similarity, and the result lies in ``[-1, 1]``.

    Parameters
    ----------
    embeddings:
        Shape ``(n, d)``. Rows need not be normalised; they are normalised
        defensively here so the function is correct for any input.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, n)``, symmetric, with a zeroed diagonal.
    """
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D (n, d) array, got shape {arr.shape}")
    if arr.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    # Guard against a zero row (an all-zero embedding is meaningless but should
    # not raise); a zero row becomes all-zeros and therefore never similar.
    norms[norms == 0.0] = 1.0
    unit = arr / norms
    sim = unit @ unit.T
    np.fill_diagonal(sim, 0.0)
    return sim

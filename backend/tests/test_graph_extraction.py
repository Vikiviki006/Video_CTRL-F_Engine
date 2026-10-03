"""Unit tests for the pluggable extraction interface.

The central claim these tests defend: **with no extractor registered, the system
reports nothing, and nothing in this codebase can populate OBJECT / TEXT /
CONCEPT nodes or DEPICTS edges.** The tests below assert that absence
explicitly, because a silent fabrication here would be the single worst failure
mode of the whole graph layer - it would look like a working concept graph.

A deliberately-synthetic extractor is registered inside one test to prove the
interface is genuinely pluggable, and it is unregistered in a finally block so
it cannot leak into any other test.
"""

import numpy as np
import pytest
from PIL import Image

from app.models.graph import NodeType
from app.services.graph_extraction_service import (
    Detection,
    GraphExtractor,
    NullGraphExtractor,
    available_extractors,
    cosine_similarity_matrix,
    extract_for_frames,
    get_graph_extractor,
    register_extractor,
)


@pytest.fixture
def dummy_image():
    return Image.new("RGB", (64, 64), color=(10, 20, 30))


# ---------------------------------------------------------------------------
# The default extractor reports nothing, honestly
# ---------------------------------------------------------------------------


def test_null_extractor_is_the_default():
    assert get_graph_extractor().name == "null"


def test_null_extractor_returns_no_detections(dummy_image):
    # Not an exception, not a placeholder: an empty list.
    assert NullGraphExtractor().extract(dummy_image, 1.0) == []


def test_null_extractor_satisfies_the_protocol(dummy_image):
    assert isinstance(NullGraphExtractor(), GraphExtractor)


def test_null_extractor_is_the_only_registered_backend_by_default():
    assert available_extractors() == ["null"]


def test_unknown_extractor_name_is_a_hard_error_not_a_silent_fallback():
    """A typo must not quietly degrade to the no-op and hide the mistake."""

    class FakeSettings:
        GRAPH_EXTRACTOR = "yolo-does-not-exist"

    with pytest.raises(ValueError, match="Unknown GRAPH_EXTRACTOR"):
        get_graph_extractor(FakeSettings())


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_detection_accepts_a_plain_string_node_type():
    det = Detection(node_type="CONCEPT", label="whiteboard")
    assert det.node_type is NodeType.CONCEPT


def test_detection_rejects_out_of_range_confidence():
    with pytest.raises(ValueError, match=r"confidence must be in \[0, 1\]"):
        Detection(node_type=NodeType.CONCEPT, confidence=1.5)


def test_detection_rejects_inverted_time_span():
    with pytest.raises(ValueError, match="timestamp_end must be >="):
        Detection(
            node_type=NodeType.TEXT,
            timestamp_start=10.0,
            timestamp_end=5.0,
        )


def test_detection_metadata_defaults_to_an_independent_dict():
    a = Detection(node_type=NodeType.CONCEPT)
    b = Detection(node_type=NodeType.CONCEPT)
    a.metadata["x"] = 1
    assert b.metadata == {}


# ---------------------------------------------------------------------------
# extract_for_frames error containment
# ---------------------------------------------------------------------------


class _ExplodingExtractor:
    name = "exploding"

    def extract(self, image, timestamp_seconds, video_id=None, frame_id=None):
        if timestamp_seconds == 2.0:
            raise RuntimeError("detector blew up on this frame")
        return [Detection(node_type=NodeType.CONCEPT, label="ok")]


def test_extract_for_frames_contains_a_single_frame_failure(dummy_image):
    """One bad frame must not abort indexing; it yields no detections instead."""
    frames = [
        {"image": dummy_image, "timestamp_seconds": 1.0, "frame_id": "a"},
        {"image": dummy_image, "timestamp_seconds": 2.0, "frame_id": "b"},
        {"image": dummy_image, "timestamp_seconds": 3.0, "frame_id": "c"},
    ]
    out = extract_for_frames(_ExplodingExtractor(), frames)
    assert len(out["a"]) == 1
    assert out["b"] == []  # the failure is contained, not propagated
    assert len(out["c"]) == 1


def test_extract_for_frames_handles_a_none_result(dummy_image):
    class NoneExtractor:
        name = "none"

        def extract(self, *a, **k):
            return None

    out = extract_for_frames(
        NoneExtractor(), [{"image": dummy_image, "timestamp_seconds": 1.0, "frame_id": "a"}]
    )
    assert out["a"] == []


# ---------------------------------------------------------------------------
# The interface really is pluggable
# ---------------------------------------------------------------------------


def test_a_custom_extractor_can_be_registered_and_selected(dummy_image):
    """Proves a real detector can be added without touching any caller."""

    class SyntheticConceptExtractor:
        """A clearly-synthetic stand-in used ONLY to test the plugin seam."""

        name = "synthetic-test-extractor"

        def __init__(self, settings=None):
            self.settings = settings

        def extract(self, image, timestamp_seconds, video_id=None, frame_id=None):
            return [
                Detection(
                    node_type=NodeType.CONCEPT,
                    label="whiteboard",
                    confidence=0.9,
                    timestamp_start=timestamp_seconds,
                    timestamp_end=timestamp_seconds,
                    metadata={"synthetic": True},
                )
            ]

    class Settings:
        GRAPH_EXTRACTOR = "synthetic-test-extractor"

    register_extractor("synthetic-test-extractor", SyntheticConceptExtractor)
    try:
        assert "synthetic-test-extractor" in available_extractors()
        extractor = get_graph_extractor(Settings())
        assert extractor.name == "synthetic-test-extractor"
        dets = extractor.extract(dummy_image, 3.0)
        assert len(dets) == 1
        assert dets[0].node_type is NodeType.CONCEPT
        assert dets[0].metadata["synthetic"] is True
    finally:
        from app.services.graph_extraction_service import _EXTRACTOR_REGISTRY

        _EXTRACTOR_REGISTRY.pop("synthetic-test-extractor", None)


def test_register_extractor_rejects_a_non_conforming_class():
    class NotAnExtractor:
        pass

    with pytest.raises(TypeError, match="GraphExtractor protocol"):
        register_extractor("bogus", NotAnExtractor)


# ---------------------------------------------------------------------------
# Cosine similarity - the single definition used for SIMILAR_TO edges
# ---------------------------------------------------------------------------


def test_cosine_similarity_matrix_of_orthogonal_vectors_is_zero():
    mat = cosine_similarity_matrix(np.eye(3, dtype=np.float32))
    assert np.allclose(mat, np.zeros((3, 3)))


def test_cosine_similarity_matrix_has_a_zeroed_diagonal():
    mat = cosine_similarity_matrix(np.ones((4, 4), dtype=np.float32))
    # A frame is excluded from being "similar to itself".
    assert np.allclose(np.diag(mat), 0.0)


def test_cosine_similarity_matrix_is_symmetric():
    rng = np.random.default_rng(0)
    mat = cosine_similarity_matrix(rng.normal(size=(6, 8)).astype(np.float32))
    assert np.allclose(mat, mat.T)


def test_cosine_similarity_matrix_normalises_unnormalised_input():
    """Defensive normalisation: a scaled vector has the same direction."""
    a = np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    mat = cosine_similarity_matrix(a)
    # Still orthogonal after normalisation.
    assert mat[0, 1] == pytest.approx(0.0)


def test_cosine_similarity_matrix_handles_empty_input():
    assert cosine_similarity_matrix(np.zeros((0, 0), dtype=np.float32)).shape == (0, 0)


def test_cosine_similarity_matrix_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2-D"):
        cosine_similarity_matrix(np.zeros(5, dtype=np.float32))

"""
Unit tests for adaptive keyframe logic.
These tests do not require a real video; they validate the parameter interface.
"""
from app.core.config import get_settings


def test_config_defaults():
    s = get_settings()
    assert s.FRAME_DIFF_THRESHOLD > 0
    assert s.MIN_FRAME_INTERVAL_SECONDS > 0
    assert s.MAX_FRAME_INTERVAL_SECONDS >= s.MIN_FRAME_INTERVAL_SECONDS
    assert s.BATCH_SIZE >= 1

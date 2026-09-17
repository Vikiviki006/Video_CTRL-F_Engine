from typing import List, Tuple
from pathlib import Path
import uuid
import numpy as np
from PIL import Image
import cv2
from sqlalchemy.orm import Session
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.models.frame import VideoFrame
from app.services.model_service import get_model_service

logger = setup_logging()
settings = get_settings()


def _frame_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR frame to RGB PIL Image."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def extract_adaptive_keyframes(
    video_path: str,
) -> List[Tuple[int, float, np.ndarray]]:
    """
    Adaptive keyframe extraction.
    Retains a frame when visual difference >= threshold OR max temporal interval is reached.
    Returns list of (frame_number, timestamp_seconds, bgr_frame).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps > 0 else 0.0

    logger.info(
        f"Adaptive extraction started | fps={fps:.2f} | frames={total_frames} | duration={duration:.1f}s"
    )

    min_interval = settings.MIN_FRAME_INTERVAL_SECONDS
    max_interval = settings.MAX_FRAME_INTERVAL_SECONDS
    threshold = settings.FRAME_DIFF_THRESHOLD

    retained: List[Tuple[int, float, np.ndarray]] = []
    prev_gray_small = None
    last_kept_ts = -999.0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts = frame_idx / fps
        keep = False

        # Always keep first frame
        if prev_gray_small is None:
            keep = True
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (64, 36))
            # Structural difference (mean absolute difference normalized)
            diff = np.mean(np.abs(small.astype(np.float32) - prev_gray_small.astype(np.float32))) / 255.0
            time_since = ts - last_kept_ts

            if diff >= threshold and time_since >= min_interval:
                keep = True
            elif time_since >= max_interval:
                keep = True

        if keep:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_gray_small = cv2.resize(gray, (64, 36))
            last_kept_ts = ts
            # Store a copy of the frame
            retained.append((frame_idx, ts, frame.copy()))

        frame_idx += 1

    cap.release()
    logger.info(
        f"Adaptive extraction finished | candidate_frames={total_frames} | retained={len(retained)} | ratio={len(retained)/max(total_frames,1):.3f}"
    )
    return retained


def process_and_store_embeddings(
    db: Session,
    video_id: uuid.UUID,
    video_path: str,
    retained_frames: List[Tuple[int, float, np.ndarray]],
) -> int:
    """
    Batch-encode retained frames with SigLIP-2 and insert into database.
    Also writes thumbnails to disk.
    """
    model = get_model_service()
    Path(settings.FRAME_ROOT).mkdir(parents=True, exist_ok=True)
    video_frame_dir = Path(settings.FRAME_ROOT) / str(video_id)
    video_frame_dir.mkdir(parents=True, exist_ok=True)

    batch_size = settings.BATCH_SIZE
    total_inserted = 0

    for i in range(0, len(retained_frames), batch_size):
        batch = retained_frames[i : i + batch_size]
        pil_images = []
        meta = []

        for frame_num, ts, bgr in batch:
            pil = _frame_to_pil(bgr)
            pil_images.append(pil)

            # Save thumbnail
            thumb_name = f"{frame_num:08d}.jpg"
            thumb_path = video_frame_dir / thumb_name
            pil.resize((320, 180)).save(str(thumb_path), quality=85)
            meta.append((frame_num, ts, str(thumb_path)))

        # Encode batch
        embeddings = model.encode_images(pil_images)

        for j, (frame_num, ts, thumb_path) in enumerate(meta):
            emb = embeddings[j].tolist()
            frame = VideoFrame(
                video_id=video_id,
                timestamp_seconds=ts,
                frame_number=frame_num,
                thumbnail_path=thumb_path,
                embedding=emb,
            )
            db.add(frame)
            total_inserted += 1

        db.commit()
        logger.info(f"Embedded batch {i // batch_size + 1} | frames={len(batch)}")

    return total_inserted

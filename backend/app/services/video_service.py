import os
import uuid
import shutil
from pathlib import Path
from typing import Optional
from fastapi import UploadFile, BackgroundTasks
from sqlalchemy.orm import Session
import cv2
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.models.video import Video
from app.services.embedding_service import extract_adaptive_keyframes, process_and_store_embeddings

logger = setup_logging()
settings = get_settings()

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def _sanitize_filename(name: str) -> str:
    """Basic sanitization – keep only safe characters."""
    base = os.path.basename(name)
    safe = "".join(c for c in base if c.isalnum() or c in "._- ")
    return safe[:200] or "video"


def validate_upload(file: UploadFile) -> None:
    if not file.filename:
        raise ValueError("Empty filename")
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    # Size check is done by reading content length if available; FastAPI middleware or content-length header preferred.


def save_upload(file: UploadFile, video_id: uuid.UUID) -> str:
    """Save uploaded file to MEDIA_ROOT using internal ID. Returns full path."""
    Path(settings.MEDIA_ROOT).mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    filename = f"{video_id}{ext}"
    path = os.path.join(settings.MEDIA_ROOT, filename)

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Basic size validation after write
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_MB:
        os.remove(path)
        raise ValueError(f"File exceeds maximum upload size of {settings.MAX_UPLOAD_MB} MB")

    if size_mb < 0.001:
        os.remove(path)
        raise ValueError("Uploaded file is empty or too small")

    return path


def get_video_metadata(path: str) -> dict:
    """Extract duration, fps, frame_count using OpenCV."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Invalid or unsupported video file")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()
    return {"duration_seconds": duration, "fps": fps, "frame_count": frame_count}


def process_video_background(video_id: uuid.UUID, db_url: str | None = None) -> None:
    """
    Background task: adaptive keyframe extraction → embedding → DB insert.
    Designed so it can later be moved to Celery/RQ without rewriting callers.
    """
    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            logger.error(f"Video {video_id} not found for processing")
            return

        video.status = "processing"
        db.commit()
        logger.info(f"Processing started for video {video_id}")

        # Adaptive extraction
        retained = extract_adaptive_keyframes(video.path)

        if not retained:
            video.status = "failed"
            video.error_message = "No frames could be extracted"
            db.commit()
            return

        # Embed + store
        inserted = process_and_store_embeddings(db, video_id, video.path, retained)

        # Update metadata if missing
        meta = get_video_metadata(video.path)
        video.duration_seconds = meta["duration_seconds"]
        video.fps = meta["fps"]
        video.frame_count = meta["frame_count"]
        video.status = "ready"
        video.error_message = None
        db.commit()
        logger.info(f"Processing completed for video {video_id} | frames_stored={inserted}")

    except Exception as e:
        logger.exception(f"Processing failed for video {video_id}: {e}")
        try:
            video = db.query(Video).filter(Video.id == video_id).first()
            if video:
                video.status = "failed"
                video.error_message = str(e)[:500]
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


def create_video_record(
    db: Session,
    file: UploadFile,
    background_tasks: BackgroundTasks,
) -> Video:
    """Create DB record, save file, enqueue background processing."""
    validate_upload(file)
    video_id = uuid.uuid4()
    original = _sanitize_filename(file.filename or "video")

    path = save_upload(file, video_id)

    try:
        meta = get_video_metadata(path)
    except Exception as e:
        os.remove(path)
        raise ValueError(f"Invalid video: {e}")

    video = Video(
        id=video_id,
        filename=os.path.basename(path),
        original_filename=original,
        path=path,
        duration_seconds=meta["duration_seconds"],
        fps=meta["fps"],
        frame_count=meta["frame_count"],
        status="queued",
    )
    db.add(video)
    db.commit()
    db.refresh(video)

    logger.info(f"Upload completed | id={video_id} | original={original}")

    # Enqueue background processing
    background_tasks.add_task(process_video_background, video_id)

    return video


def list_videos(db: Session, skip: int = 0, limit: int = 50) -> list[Video]:
    return db.query(Video).order_by(Video.created_at.desc()).offset(skip).limit(limit).all()


def get_video(db: Session, video_id: uuid.UUID) -> Optional[Video]:
    return db.query(Video).filter(Video.id == video_id).first()

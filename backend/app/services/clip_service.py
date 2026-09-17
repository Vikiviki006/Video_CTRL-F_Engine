import os
import subprocess
import uuid
from pathlib import Path
from app.core.config import get_settings
from app.core.logging import setup_logging

logger = setup_logging()
settings = get_settings()


def generate_clip(
    video_path: str,
    timestamp_seconds: float,
    duration: float = 5.0,
    video_id: str | None = None,
) -> str:
    """
    Generate a contextual 5-second clip centered around the given timestamp.
    Uses FFmpeg with argument list (no shell injection).
    Returns the relative path to the generated clip.
    """
    Path(settings.CLIP_ROOT).mkdir(parents=True, exist_ok=True)

    half = duration / 2.0
    start = max(0.0, timestamp_seconds - half)
    # FFmpeg will clamp automatically if beyond end

    clip_id = str(uuid.uuid4())
    out_name = f"{video_id or 'clip'}_{clip_id}.mp4"
    out_path = os.path.join(settings.CLIP_ROOT, out_name)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-movflags", "+faststart",
        out_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            logger.error(f"FFmpeg failed: {result.stderr[-500:]}")
            raise RuntimeError(f"Clip generation failed: {result.stderr[-200:]}")
        logger.info(f"Generated clip: {out_path}")
        return out_name
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg timed out")
        raise RuntimeError("Clip generation timed out")
    except FileNotFoundError:
        logger.error("FFmpeg binary not found")
        raise RuntimeError("FFmpeg is not installed")

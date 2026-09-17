from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List
from pathlib import Path


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+psycopg2://videocontrol:videocontrol@localhost:5432/videocontrol"
    MODEL_NAME: str = "google/siglip2-so400m-patch14-384"
    MEDIA_ROOT: str = "./data/media"
    FRAME_ROOT: str = "./data/frames"
    CLIP_ROOT: str = "./data/clips"
    BATCH_SIZE: int = 8
    FRAME_DIFF_THRESHOLD: float = 0.08
    MIN_FRAME_INTERVAL_SECONDS: float = 0.5
    MAX_FRAME_INTERVAL_SECONDS: float = 3.0
    MAX_UPLOAD_MB: int = 2048
    TEMPORAL_GROUP_WINDOW: float = 3.0
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"
    LOG_LEVEL: str = "INFO"
    HF_HOME: str = "./data/models"

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    class Config:
        # Look for .env in current dir and parent (project root)
        env_file = (".env", "../.env")
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()

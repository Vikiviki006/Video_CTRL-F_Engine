import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID
from app.db.session import Base


class Video(Base):
    __tablename__ = "videos"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(512), nullable=False)
    original_filename = Column(String(512), nullable=False)
    path = Column(String(1024), nullable=False)
    duration_seconds = Column(Float, nullable=True)
    fps = Column(Float, nullable=True)
    frame_count = Column(Integer, nullable=True)
    status = Column(String(32), nullable=False, default="queued")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

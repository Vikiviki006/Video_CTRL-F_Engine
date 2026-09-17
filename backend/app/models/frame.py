import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from app.db.session import Base


class VideoFrame(Base):
    __tablename__ = "video_frames"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(
        UUID(as_uuid=True),
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    timestamp_seconds = Column(Float, nullable=False)
    frame_number = Column(Integer, nullable=False)
    thumbnail_path = Column(String(1024), nullable=False)
    embedding = Column(Vector(1152), nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

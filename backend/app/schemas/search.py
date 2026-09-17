from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from uuid import UUID


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    video_id: Optional[UUID] = None
    top_k: int = Field(default=10, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()


class SearchResult(BaseModel):
    frame_id: UUID
    video_id: UUID
    video_name: str
    timestamp_seconds: float
    score: float
    thumbnail_url: str
    clip_url: Optional[str] = None
    group_id: Optional[int] = None


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total: int
    latency_ms: float

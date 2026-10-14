from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from uuid import UUID


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    video_id: Optional[UUID] = None
    top_k: int = Field(default=10, ge=1, le=100)
    min_score: float = Field(
        default=-1.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Cosine-similarity floor for the candidate pool. The default is "
            "the lowest possible cosine, i.e. no floor. It must NOT default to "
            "0.0: SigLIP text/image cosines cluster near zero "
            "(measured range about -0.06..0.06 on this corpus), so a 0.0 floor "
            "silently emptied the pool for 6 of 25 evaluation queries and cut "
            "hitrate@10 from 0.64 to 0.48. Raise it only to prune, never to filter."
        ),
    )

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

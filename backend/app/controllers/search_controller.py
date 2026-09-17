from uuid import UUID
from fastapi import HTTPException
from sqlalchemy.orm import Session
from app.schemas.search import SearchRequest, SearchResponse
from app.services import search_service


def perform_search(db: Session, request: SearchRequest) -> SearchResponse:
    try:
        return search_service.search(db, request)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Search failed")


def get_clip(db: Session, frame_id: UUID) -> dict:
    try:
        url = search_service.generate_result_clip(db, frame_id)
        return {"clip_url": url}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Clip generation failed")

from uuid import UUID
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.controllers import search_controller
from app.schemas.search import SearchRequest, SearchResponse

router = APIRouter(prefix="/search")


@router.post(
    "",
    response_model=SearchResponse,
    summary="Semantic video search",
    description="Search video content using natural language. Returns ranked timestamps with thumbnails.",
)
def search(request: SearchRequest, db: Session = Depends(get_db)):
    return search_controller.perform_search(db, request)


@router.post(
    "/clip/{frame_id}",
    summary="Generate a 5-second contextual clip for a result frame",
)
def generate_clip(frame_id: UUID, db: Session = Depends(get_db)):
    return search_controller.get_clip(db, frame_id)

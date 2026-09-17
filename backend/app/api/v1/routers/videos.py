from uuid import UUID
from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, Query
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.controllers import video_controller
from app.schemas.video import VideoResponse, VideoListResponse

router = APIRouter(prefix="/videos")


@router.post(
    "",
    response_model=VideoResponse,
    summary="Upload a video",
    description="Upload a video file (mp4, mov, mkv, webm, avi). Processing starts in background.",
)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    return video_controller.upload_video(db, file, background_tasks)


@router.get(
    "",
    response_model=VideoListResponse,
    summary="List uploaded videos",
)
def list_videos(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    return video_controller.list_videos(db, skip=skip, limit=limit)


@router.get(
    "/{video_id}",
    response_model=VideoResponse,
    summary="Get video by ID",
)
def get_video(video_id: UUID, db: Session = Depends(get_db)):
    return video_controller.get_video(db, video_id)

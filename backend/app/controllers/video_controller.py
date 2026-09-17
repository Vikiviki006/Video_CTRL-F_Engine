from uuid import UUID
from fastapi import UploadFile, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session
from app.schemas.video import VideoResponse, VideoListResponse
from app.services import video_service


def upload_video(
    db: Session,
    file: UploadFile,
    background_tasks: BackgroundTasks,
) -> VideoResponse:
    try:
        video = video_service.create_video_record(db, file, background_tasks)
        return VideoResponse.model_validate(video)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Upload failed")


def list_videos(db: Session, skip: int = 0, limit: int = 50) -> VideoListResponse:
    videos = video_service.list_videos(db, skip=skip, limit=limit)
    return VideoListResponse(
        videos=[VideoResponse.model_validate(v) for v in videos],
        total=len(videos),
    )


def get_video(db: Session, video_id: UUID) -> VideoResponse:
    video = video_service.get_video(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return VideoResponse.model_validate(video)

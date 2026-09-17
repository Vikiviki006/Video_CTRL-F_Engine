import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.api.v1.router import api_router

settings = get_settings()
logger = setup_logging()

app = FastAPI(
    title="VideoCtrl-F",
    description="Adaptive Keyframe and SigLIP-2 Based Semantic Video Retrieval Framework",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure media directories exist
for d in [settings.MEDIA_ROOT, settings.FRAME_ROOT, settings.CLIP_ROOT]:
    Path(d).mkdir(parents=True, exist_ok=True)

# Mount static media (thumbnails + clips)
# Frames are under FRAME_ROOT/<video_id>/<frame>.jpg
app.mount("/media/frames", StaticFiles(directory=settings.FRAME_ROOT), name="frames")
app.mount("/media/clips", StaticFiles(directory=settings.CLIP_ROOT), name="clips")
app.mount("/media/videos", StaticFiles(directory=settings.MEDIA_ROOT), name="videos")

app.include_router(api_router)


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


@app.on_event("startup")
def on_startup():
    logger.info("VideoCtrl-F backend started")
    # Optionally warm up model (can be slow on first request otherwise)
    # from app.services.model_service import get_model_service
    # get_model_service()


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

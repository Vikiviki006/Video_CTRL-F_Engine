from fastapi import APIRouter
from app.api.v1.routers import videos, search

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(videos.router, tags=["Videos"])
api_router.include_router(search.router, tags=["Search"])

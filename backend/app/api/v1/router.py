from fastapi import APIRouter
from app.api.v1.routers import videos, search, graph

api_router = APIRouter(prefix="/api/v1")
# Existing routers are included first and in their original order, so their
# route table is byte-identical to before. The graph router is appended.
api_router.include_router(videos.router, tags=["Videos"])
api_router.include_router(search.router, tags=["Search"])
api_router.include_router(graph.router, tags=["Graph"])

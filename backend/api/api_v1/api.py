from fastapi import APIRouter

from .tryon import tryon_router
from .user import user_router
from .compatibility import compatibility_router
from .upload import upload_router

api_router = APIRouter()

api_router.include_router(user_router, prefix="/user", tags=["user"])
api_router.include_router(
    compatibility_router, prefix="/compatibility", tags=["compatibility"]
)
api_router.include_router(upload_router, prefix="/upload", tags=["upload"])
api_router.include_router(tryon_router, prefix="/tryon", tags=["tryon"])
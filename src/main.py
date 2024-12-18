from fastapi import FastAPI
from src.core.config import settings
from src.api.endpoints import video

app = FastAPI(title=settings.PROJECT_NAME)

# Add video processing routes
app.include_router(
    video.router,
    prefix=f"{settings.API_V1_STR}/video",
    tags=["video"]
) 
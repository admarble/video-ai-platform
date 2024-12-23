from fastapi import FastAPI
from src.core.config import settings
from src.api.endpoints import video
from src.core.middleware import SecurityHeadersMiddleware
from src.core.security import SecurityLevel

app = FastAPI(title=settings.PROJECT_NAME)

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware, security_level=SecurityLevel.HIGH)

# Add video processing routes
app.include_router(
    video.router,
    prefix=f"{settings.API_V1_STR}/video",
    tags=["video"]
) 
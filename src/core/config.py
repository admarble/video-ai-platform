from typing import List
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Video AI Platform"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []
    
    # Database
    DATABASE_URL: str
    
    # Redis
    REDIS_URL: str
    
    # Model Settings
    SCENE_MODEL: str = "MCG-NJU/videomae-base"
    OBJECT_MODEL: str = "facebook/detr-resnet-50"
    AUDIO_MODEL: str = "facebook/wav2vec2-base-960h"
    CLIP_MODEL: str = "openai/clip-vit-base-patch32"
    
    # Processing
    BATCH_SIZE: int = 32
    MAX_VIDEO_LENGTH: int = 3600  # Maximum video length in seconds
    
    # Storage paths
    UPLOAD_DIR: str = "data/uploads"
    PROCESSING_DIR: str = "data/processing"
    
    # Processing settings
    FRAME_SAMPLING_RATE: int = 2
    MIN_SEGMENT_LENGTH: int = 30
    CONFIDENCE_THRESHOLD: float = 0.5
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()

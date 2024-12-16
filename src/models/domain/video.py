from pydantic import BaseModel, HttpUrl
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class VideoUpload(BaseModel):
    url: HttpUrl
    title: Optional[str] = None
    description: Optional[str] = None

class SceneInfo(BaseModel):
    start_time: float
    end_time: float
    description: str
    confidence: float
    objects: List[str]

class ObjectDetection(BaseModel):
    label: str
    confidence: float
    bbox: List[float]
    timestamp: float

class VideoMetadata(BaseModel):
    id: str
    url: HttpUrl
    status: ProcessingStatus
    duration: Optional[float] = None
    frame_count: Optional[int] = None
    fps: Optional[float] = None
    resolution: Optional[tuple] = None
    created_at: datetime
    updated_at: datetime

class ProcessingResult(BaseModel):
    video_id: str
    scenes: List[SceneInfo]
    objects: List[ObjectDetection]
    transcript: Dict[float, str]  # timestamp: text
    embeddings: Optional[Dict[str, List[float]]] = None  # For CLIP features
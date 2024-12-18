from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional, Dict
from dataclasses import dataclass

class VideoMetadata(BaseModel):
    video_id: str
    filename: str
    file_path: str
    file_size: int
    duration: float
    width: int
    height: int
    fps: float
    created_at: datetime
    format: str
    status: str = "pending"

class ProcessingResults(BaseModel):
    metadata: VideoMetadata
    scenes: List[Dict]
    objects: List[Dict]
    audio_segments: List[Dict]
    frame_embeddings: Optional[List[float]] = None 

@dataclass
class SceneSegment:
    """Represents a scene segment in a video"""
    start_frame: int
    end_frame: int
    confidence: float
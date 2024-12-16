from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

class VideoMetadata(BaseModel):
    duration: float
    fps: float
    total_frames: int
    resolution: tuple[int, int]

class SceneInfo(BaseModel):
    timestamp: float
    scenes: List[Dict[str, float]]  # label: confidence

class ObjectInfo(BaseModel):
    timestamp: float
    detections: List[Dict]  # objects with boxes and scores

class ProcessingResult(BaseModel):
    metadata: VideoMetadata
    scenes: List[SceneInfo]
    objects: List[ObjectInfo]
    processed_at: datetime = datetime.utcnow()
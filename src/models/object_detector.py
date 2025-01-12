from typing import Optional, List, Dict, Any
from pydantic import BaseModel

class DetectedObject(BaseModel):
    label: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    track_id: Optional[int] = None

class ObjectDetector(BaseModel):
    model_path: str
    confidence_threshold: float = 0.5
    device: Optional[str] = None
    enable_tracking: bool = False
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

@dataclass
class Scene:
    start_time: float
    end_time: float
    description: str

@dataclass
class ObjectDetection:
    label: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] normalized coordinates
    frame_idx: int

@dataclass
class ProcessingResult:
    video_id: str
    scenes: List[Scene]
    objects: List[ObjectDetection]
    transcript: Dict[float, str]  # timestamp -> text
    embeddings: Dict[str, Any]
    metadata: Dict[str, Any] 
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

@dataclass
class DetectedObject:
    """Represents a detected object in a frame"""
    label: str
    confidence: float
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized coordinates
    frame_idx: int
    track_id: Optional[int] = None

@dataclass
class Scene:
    """Represents a detected scene/segment in the video"""
    start_frame: int
    end_frame: int
    label: str
    confidence: float

@dataclass
class VideoMetadata:
    """Video metadata information"""
    duration: float
    fps: float
    frame_count: int
    resolution: Tuple[int, int]  # (width, height)

@dataclass
class ProcessingResult:
    """Combined results from video processing pipeline"""
    scenes: List[Scene]
    objects: List[DetectedObject]
    metadata: VideoMetadata
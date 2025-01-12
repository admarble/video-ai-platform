from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime

@dataclass
class ProcessingProgress:
    stage: str
    progress: float  # 0 to 100
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class ProcessingStage(Enum):
    INITIALIZATION = "initialization"
    FRAME_EXTRACTION = "frame_extraction"
    SCENE_ANALYSIS = "scene_analysis"
    OBJECT_DETECTION = "object_detection"
    AUDIO_PROCESSING = "audio_processing"
    EMBEDDING_GENERATION = "embedding_generation"
    RESULT_COMPILATION = "result_compilation"
    VALIDATION = "validation"
    COMPLETION = "completion"

class ValidationLevel(Enum):
    BASIC = "basic"
    THOROUGH = "thorough"
    STRICT = "strict" 
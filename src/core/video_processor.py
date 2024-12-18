from pathlib import Path
import logging
from typing import Dict, Optional, List, Any, Union
from dataclasses import dataclass
import json
import hashlib
from datetime import datetime
import os
import shutil
import numpy as np

from src.services.service_manager import ServiceManager, ModelConfig
from src.exceptions import VideoProcessingError, ServiceInitializationError
from src.core.config import get_config

@dataclass
class VideoMetadata:
    """Metadata for a video file"""
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

@dataclass
class ProcessingResults:
    """Results from video processing pipeline"""
    metadata: VideoMetadata
    scenes: List[Dict]
    objects: List[Dict]
    audio_segments: List[Dict]
    frame_embeddings: Optional[List[float]] = None
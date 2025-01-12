from typing import Dict, Any, Optional, List
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import psutil
import logging
import gc
import os

from .exceptions import ServiceCleanupError, VideoProcessingError
from src.core.config import ModelConfig
from src.services.ml.scene import SceneProcessor
from src.services.ml.objects import ObjectDetector
from src.models.video import SceneSegment

class ServiceManager:
    """Manages video processing services"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self._services: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        
    def initialize_services(self, service_names: List[str]) -> None:
        """Initialize requested services"""
        for name in service_names:
            if name == 'scene_processor':
                self._services[name] = SceneProcessor()
            elif name == 'object_detector':
                self._services[name] = ObjectDetector()
                
    def get_service(self, name: str) -> Any:
        """Get an initialized service"""
        if name not in self._services:
            raise KeyError(f"Service {name} not initialized")
        return self._services[name]
        
    def cleanup(self) -> None:
        """Cleanup resources"""
        cleanup_errors = []
        for name, service in self._services.items():
            try:
                if hasattr(service, 'cleanup'):
                    service.cleanup()
            except Exception as e:
                cleanup_errors.append(f"{name}: {str(e)}")
                
        self._services.clear()
        
        if cleanup_errors:
            raise ServiceCleanupError(
                f"Failed to cleanup services: {', '.join(cleanup_errors)}"
            )

def _extract_frames(video_path: str) -> tuple[np.ndarray, float]:
    """Extract frames from video file"""
    if not os.path.exists(video_path):
        raise VideoProcessingError(f"Video file not found: {video_path}")
        
    # Simplified implementation for testing
    frames = np.random.randint(0, 255, (30, 640, 480, 3), dtype=np.uint8)
    fps = 30.0
    return frames, fps
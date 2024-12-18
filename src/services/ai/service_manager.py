from pathlib import Path
import logging
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
import torch
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import time

from src.exceptions import ServiceInitializationError
from src.models.scene_processor import SceneProcessor
from src.models.object_detector import ObjectDetector
from src.models.audio_processor import AudioProcessor
from src.models.text_video_aligner import TextVideoAligner

@dataclass
class ModelConfig:
    """Configuration for ML models"""
    scene_model: str = "MCG-NJU/videomae-base-finetuned-kinetics"
    object_model: str = "facebook/detr-resnet-50"
    audio_model: str = "facebook/wav2vec2-base-960h"
    alignment_model: str = "openai/clip-vit-base-patch32"
    device: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 4

class ServiceManager:
    """Manages initialization and coordination of ML services"""
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        cache_dir: Optional[str] = None
    ):
        self.config = config or ModelConfig()
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".video_ai_cache"
        self.device = self._setup_device()
        self._services: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._initialization_status = {}
        
    def _setup_device(self) -> str:
        if self.config.device:
            return self.config.device
        return "cuda" if torch.cuda.is_available() else "cpu"
        
    def _initialize_model_cache(self) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Initialized model cache at {self.cache_dir}")
        except Exception as e:
            raise ServiceInitializationError(f"Failed to create cache directory: {str(e)}")
            
    def _initialize_service(self, service_name: str) -> None:
        try:
            if service_name == "scene_processor":
                service = SceneProcessor(
                    model_name=self.config.scene_model,
                    device=self.device
                )
            elif service_name == "object_detector":
                service = ObjectDetector(
                    model_name=self.config.object_model,
                    device=self.device,
                    batch_size=self.config.batch_size
                )
            elif service_name == "audio_processor":
                service = AudioProcessor(
                    model_name=self.config.audio_model,
                    device=self.device
                )
            elif service_name == "text_aligner":
                service = TextVideoAligner(
                    model_name=self.config.alignment_model,
                    device=self.device,
                    batch_size=self.config.batch_size
                )
            else:
                raise ValueError(f"Unknown service: {service_name}")
                
            with self._lock:
                self._services[service_name] = service
                self._initialization_status[service_name] = True
                
            logging.info(f"Successfully initialized {service_name}")
            
        except Exception as e:
            self._initialization_status[service_name] = False
            raise ServiceInitializationError(f"Failed to initialize {service_name}: {str(e)}") 
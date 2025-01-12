from contextlib import contextmanager
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
import gc
import torch
from .exceptions import ServiceCleanupError
from .resource_monitor import ResourceMonitor
import psutil
from datetime import datetime
import structlog
import time
from dataclasses import dataclass
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from src.core.logging import LoggingManager, LogLevel

@dataclass
class ModelVersion:
    version: str
    path: str
    created_at: datetime
    metadata: Dict[str, Any]

@dataclass
class SceneSegment:
    """Represents a scene segment in a video"""
    start_frame: int
    end_frame: int
    confidence: float

class VideoProcessingError(Exception):
    """Custom exception for video processing errors"""
    pass

@dataclass
class ModelConfig:
    """Configuration for ML models"""
    model_path: str = './models'
    batch_size: int = 32
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class VideoProcessingLogger:
    """Handles logging for video processing operations"""
    
    def __init__(self):
        self.logger = structlog.get_logger()
        
    def start_operation(self, operation: str, **kwargs):
        self.logger.info(f"Starting {operation}", **kwargs)
        
    def end_operation(self, operation: str, **kwargs):
        self.logger.info(f"Completed {operation}", **kwargs)
        
    def log_progress(self, operation: str, progress: float, **kwargs):
        self.logger.info(f"{operation} progress", progress=progress, **kwargs)
        
    def info(self, message: str, **kwargs):
        self.logger.info(message, **kwargs)

class ErrorHandler:
    """Handles errors during video processing"""
    
    def __init__(self, logger: VideoProcessingLogger):
        self.logger = logger
        
    def handle_error(
        self,
        error: Exception,
        operation: str,
        **context
    ) -> Dict[str, Any]:
        """Handle errors and determine if retry is needed"""
        self.logger.info(
            "Error occurred",
            operation=operation,
            error=str(error),
            **context
        )
        
        # Determine if error is retryable
        should_retry = isinstance(error, (RuntimeError, TimeoutError))
        
        return {
            'error': str(error),
            'should_retry': should_retry,
            'context': context
        }

class ServiceManager:
    """Manages video processing services"""
    
    def __init__(self, config: Optional[ModelConfig] = None, cache_dir: Optional[str] = None):
        self.config = config or ModelConfig()
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".video_ai_cache"
        self.device = self._setup_device()
        self._services: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._initialization_status = {}
        self._resource_monitor = ResourceMonitor()
        
        # Initialize logging manager
        self.logging_manager = LoggingManager(
            base_dir=Path(__file__).parent.parent.parent,
            config={
                "system": {
                    "level": "INFO",
                    "format": "%(asctime)s - %(name)s - [%(levelname)s] %(message)s"
                }
            }
        )
        self.logger = self.logging_manager.get_logger("system")
        self.error_handler = ErrorHandler(self.logger)
        self.model_cache = {}
        self.model_versions: Dict[str, ModelVersion] = {}
        
    @contextmanager
    def service_context(self, services: Optional[List[str]] = None):
        """Context manager for automatic service cleanup"""
        try:
            self.initialize_services(services)
            yield self
        finally:
            self.cleanup()
            
    def cleanup(self) -> None:
        """Enhanced cleanup with memory management"""
        cleanup_errors = []
        
        # Release model resources
        for service_name, service in self._services.items():
            if hasattr(service, 'cleanup'):
                try:
                    service.cleanup()
                except Exception as e:
                    cleanup_errors.append((service_name, str(e)))
                    self.logger.error(f"Failed to cleanup service '{service_name}': {str(e)}")
                    
        # Clear internal dictionaries
        self._services.clear()
        self._initialization_status.clear()
        
        # GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        # Force garbage collection
        gc.collect()
        
        # Monitor resource usage
        self._resource_monitor.log_usage()
        
        # If there were any errors during cleanup, raise a consolidated exception
        if cleanup_errors:
            error_msg = "\n".join(
                f"- {service_name}: {error}" 
                for service_name, error in cleanup_errors
            )
            raise ServiceCleanupError(f"Cleanup failed for some services:\n{error_msg}")

    def __enter__(self):
        """Enable context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure cleanup is called when using context manager"""
        self.cleanup()

    def calculate_batch_size(self, sample_input_size: int) -> int:
        """Dynamically calculate batch size based on available memory"""
        available_memory = psutil.virtual_memory().available
        gpu_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        
        # Calculate memory per sample (with safety factor)
        safety_factor = 0.7
        memory_per_sample = sample_input_size * 4  # Assuming float32
        
        # Use minimum of CPU and GPU memory if GPU is available
        usable_memory = min(available_memory, gpu_memory) if gpu_memory > 0 else available_memory
        max_batch_size = int((usable_memory * safety_factor) // memory_per_sample)
        
        return max(1, min(max_batch_size, 32))  # Cap between 1 and 32

    def process_batch(self, items: List[Any], processor_func) -> List[Any]:
        """Process items in optimally-sized batches with error handling"""
        batch_size = self.calculate_batch_size(len(items[0]) if items else 0)
        results = []
        total_items = len(items)
        
        try:
            self.logger.start_operation("batch_processing", total_items=total_items, batch_size=batch_size)
            
            for i in range(0, total_items, batch_size):
                batch = items[i:i + batch_size]
                try:
                    batch_results = processor_func(batch)
                    results.extend(batch_results)
                    
                    # Log progress
                    progress = min(100, (i + len(batch)) * 100 / total_items)
                    self.logger.log_progress("batch_processing", progress, current_batch=i//batch_size)
                    
                except Exception as e:
                    # Handle batch error
                    error_result = self.error_handler.handle_error(
                        e,
                        "batch_processing",
                        batch_index=i,
                        batch_size=batch_size
                    )
                    
                    if error_result.get('should_retry'):
                        # Retry with smaller batch size
                        batch_size = max(1, batch_size // 2)
                        i -= len(batch)  # Retry current batch
                        
            self.logger.end_operation("batch_processing")
            return results
            
        except Exception as e:
            self.error_handler.handle_error(e, "batch_processing_complete")
            raise

    def load_model(self, model_name: str, version: str, quantize: bool = False) -> Any:
        """Load model with caching and version management"""
        cache_key = f"{model_name}_{version}_{quantize}"
        
        try:
            # Check cache first
            if cache_key in self.model_cache:
                self.logger.info("Model loaded from cache", model=model_name, version=version)
                return self.model_cache[cache_key]
            
            # Get model version info
            model_version = self.model_versions.get(version)
            if not model_version:
                raise ValueError(f"Model version {version} not found")
            
            self.logger.start_operation("model_loading", model=model_name, version=version)
            
            # Load model
            model = torch.load(model_version.path)
            
            # Apply quantization if requested
            if quantize:
                model = self.quantize_model(model)
            
            # Cache the model
            self.model_cache[cache_key] = model
            
            self.logger.end_operation("model_loading")
            return model
            
        except Exception as e:
            self.error_handler.handle_error(e, "model_loading", model=model_name, version=version)
            raise

    def quantize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Quantize model to reduce memory footprint"""
        try:
            self.logger.start_operation("model_quantization")
            
            # Quantize model to int8
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
            
            self.logger.end_operation("model_quantization")
            return quantized_model
            
        except Exception as e:
            self.error_handler.handle_error(e, "model_quantization")
            raise

    def register_model_version(self, version: str, path: str, metadata: Dict[str, Any] = None):
        """Register a new model version"""
        self.model_versions[version] = ModelVersion(
            version=version,
            path=path,
            created_at=datetime.utcnow(),
            metadata=metadata or {}
        )

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
        
    def batch_process_parallel(
        self,
        items: List[Any],
        service_name: str,
        method_name: str,
        batch_size: int = 10
    ) -> List[Any]:
        """Process items in parallel batches"""
        service = self.get_service(service_name)
        
        results = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                future = executor.submit(
                    getattr(service, method_name),
                    batch
                )
                futures.append(future)
                
            results = [f.result() for f in futures]
            
        return results

class SceneProcessor:
    """Handles scene detection and segmentation"""
    
    def process_scenes(self, frames: np.ndarray) -> List[SceneSegment]:
        """Process video frames and detect scene changes"""
        # Simplified implementation for testing
        scenes = [
            SceneSegment(0, len(frames) // 2, 0.9),
            SceneSegment(len(frames) // 2, len(frames), 0.8)
        ]
        return scenes
        
    def cleanup(self):
        pass

class ObjectDetector:
    """Handles object detection in video frames"""
    
    def process_frames(
        self,
        frames: np.ndarray,
        enable_tracking: bool = False
    ) -> List[Dict[str, Any]]:
        """Detect objects in video frames"""
        # Simplified implementation for testing
        return [{'objects': []} for _ in range(len(frames))]
        
    def cleanup(self):
        pass

def _extract_frames(video_path: str) -> tuple[np.ndarray, float]:
    """Extract frames from video file"""
    if not os.path.exists(video_path):
        raise VideoProcessingError(f"Video file not found: {video_path}")
        
    # Simplified implementation for testing
    frames = np.random.randint(0, 255, (30, 640, 480, 3), dtype=np.uint8)
    fps = 30.0
    return frames, fps
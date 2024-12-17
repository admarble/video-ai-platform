from dataclasses import dataclass
from typing import Optional, List, Tuple
import asyncio
import logging
import torch
import numpy as np
import decord
import gc
import os

from src.core.model_loader import ModelManager
from src.services.ml import SceneAnalyzer, ObjectDetector
from src.models.domain.video import ProcessingResult

@dataclass
class ProcessingConfig:
    gpu_enabled: bool = True
    batch_size: int = 32
    max_memory_gb: float = 4.0
    num_workers: int = 4
    cache_dir: Optional[str] = None
    logging_level: str = "INFO"

class VideoProcessor:
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.logger = logging.getLogger(__name__)
        self.model_manager = ModelManager()
        self.scene_analyzer = SceneAnalyzer(self.model_manager)
        self.object_detector = ObjectDetector(self.model_manager)
        self._setup_logging()

    def __enter__(self):
        self._initialize_services()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_resources()

    def _initialize_services(self):
        """Initialize processing services and GPU if available"""
        if self.config.gpu_enabled and torch.cuda.is_available():
            self.logger.info("GPU acceleration enabled")
            self.device = torch.device("cuda")
        else:
            self.logger.info("Running on CPU")
            self.device = torch.device("cpu")

    def _cleanup_resources(self):
        """Clean up GPU memory and other resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.logger.info("Resources cleaned up")

    def _setup_logging(self):
        """Configure logging based on config"""
        logging.basicConfig(
            level=self.config.logging_level,
            format='%(asctime)s %(levelname)s [%(name)s] %(message)s'
        )

    async def process_video(self, video_path: str) -> ProcessingResult:
        """Process video through ML pipeline with improved memory management"""
        self.logger.info(f"Starting video processing: {video_path}")
        
        try:
            # Initialize video reader with GPU support if available
            ctx = decord.gpu(0) if (self.config.gpu_enabled and torch.cuda.is_available()) else decord.cpu(0)
            video_reader = decord.VideoReader(video_path, ctx=ctx)
            total_frames = len(video_reader)
            
            self.logger.info(f"Processing {total_frames} frames")
            
            # Process in optimized batches
            batch_results = []
            for batch_frames, frame_indices in self._get_frame_batches(video_reader, total_frames):
                self.logger.debug(f"Processing batch of {len(frame_indices)} frames")
                
                # Process batch
                scene_results = await self.scene_analyzer.analyze(batch_frames)
                object_results = await self._process_batch_with_memory_limit(
                    batch_frames,
                    max_memory_gb=self.config.max_memory_gb
                )
                
                batch_results.append((scene_results, object_results))
                
            result = self._combine_results(batch_results)
            self.logger.info("Video processing completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}", exc_info=True)
            raise

    def _get_frame_batches(self, video_reader: decord.VideoReader, total_frames: int) -> Tuple[np.ndarray, List[int]]:
        """Generator for memory-efficient frame batch extraction
        
        Args:
            video_reader: Decord video reader instance
            total_frames: Total number of frames to process
            
        Yields:
            Tuple containing:
                - np.ndarray: Batch of frames
                - List[int]: Frame indices for the batch
        """
        for batch_idx in range(0, total_frames, self.config.batch_size):
            frame_indices = list(range(
                batch_idx,
                min(batch_idx + self.config.batch_size, total_frames)
            ))
            batch_frames = video_reader.get_batch(frame_indices).asnumpy()
            yield batch_frames, frame_indices

    async def _process_batch_with_memory_limit(
        self,
        frames: np.ndarray,
        max_memory_gb: float = 4.0
    ) -> List[List[dict]]:
        """Process frames with memory limit"""
        # Calculate memory usage per frame (approximate)
        frame_memory = frames[0].nbytes * frames.shape[0]
        available_memory = max_memory_gb * 1024 * 1024 * 1024  # Convert to bytes
        
        if frame_memory > available_memory:
            # Adjust batch size to fit memory constraints
            ratio = available_memory / frame_memory
            adjusted_batch_size = max(1, int(self.config.batch_size * ratio))
            self.logger.warning(f"Adjusting batch size to {adjusted_batch_size} due to memory constraints")
            
            results = []
            for i in range(0, len(frames), adjusted_batch_size):
                batch = frames[i:i + adjusted_batch_size]
                result = self.object_detector.process_frames(
                    batch,
                    batch_size=adjusted_batch_size,
                    enable_tracking=True
                )
                results.extend(result)
            return results
        
        return self.object_detector.process_frames(
            frames,
            batch_size=self.config.batch_size,
            enable_tracking=True
        )

    def _combine_results(self, batch_results: List[Tuple[List, List]]) -> ProcessingResult:
        """Combine batch results into final output"""
        scenes = []
        objects = []
        
        for scene_batch, object_batch in batch_results:
            scenes.extend(scene_batch)
            objects.extend(object_batch)
        
        return ProcessingResult(
            scenes=scenes,
            objects=objects,
            metadata=self._extract_metadata()
        )

    def _extract_metadata(self) -> dict:
        """Extract processing metadata"""
        return {
            "gpu_enabled": self.config.gpu_enabled and torch.cuda.is_available(),
            "batch_size": self.config.batch_size,
            "device": str(self.device)
        }
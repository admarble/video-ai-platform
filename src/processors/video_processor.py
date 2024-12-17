import asyncio
import logging
import time
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Any

from ..models.processing import (
    ProcessingProgress,
    ProcessingStage,
    ValidationLevel
)
from ..models.results import ProcessingResult
from ..exceptions import VideoProcessingError
from ..utils.video import _extract_frames

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        self.progress_tracker: Dict[str, Dict[ProcessingStage, ProcessingProgress]] = {}
        self.validation_level = ValidationLevel.THOROUGH
        
    async def process_video(self, video_path: str) -> ProcessingResult:
        """
        Enhanced video processing with detailed progress tracking and validation.
        """
        video_id = str(Path(video_path).stem)
        self.progress_tracker[video_id] = {}
        
        try:
            # Initialize progress tracking
            await self._update_progress(video_id, ProcessingStage.INITIALIZATION, 0)
            
            # Validate input
            await self._validate_input(video_path)
            await self._update_progress(video_id, ProcessingStage.INITIALIZATION, 100)

            # Extract frames with progress tracking
            await self._update_progress(video_id, ProcessingStage.FRAME_EXTRACTION, 0)
            frames = await self._extract_frames_with_progress(video_path, video_id)
            await self._update_progress(video_id, ProcessingStage.FRAME_EXTRACTION, 100)

            # Create and track processing tasks
            processing_tasks = [
                self._create_tracked_task(
                    video_id,
                    ProcessingStage.SCENE_ANALYSIS,
                    self._process_scenes,
                    frames
                ),
                self._create_tracked_task(
                    video_id,
                    ProcessingStage.OBJECT_DETECTION,
                    self._process_objects,
                    frames
                ),
                self._create_tracked_task(
                    video_id,
                    ProcessingStage.AUDIO_PROCESSING,
                    self._process_audio,
                    video_path
                ),
                self._create_tracked_task(
                    video_id,
                    ProcessingStage.EMBEDDING_GENERATION,
                    self._generate_embeddings,
                    frames
                )
            ]

            results = await asyncio.gather(*processing_tasks)
            
            # Combine and validate results
            await self._update_progress(video_id, ProcessingStage.RESULT_COMPILATION, 0)
            combined_results = await self._combine_and_validate_results(video_id, video_path, results)
            await self._update_progress(video_id, ProcessingStage.COMPLETION, 100)

            return combined_results

        except Exception as e:
            current_stage = next(
                (stage for stage, progress in self.progress_tracker[video_id].items() 
                 if progress.status == "in_progress"),
                ProcessingStage.INITIALIZATION
            )
            await self._update_progress(
                video_id,
                current_stage,
                0,
                error=str(e)
            )
            raise 

    async def _extract_frames_with_progress(self, video_path: str, video_id: str) -> List[np.ndarray]:
        """Extract frames with progress tracking"""
        try:
            # Extract frames using decord with adaptive sampling
            frames, fps = _extract_frames(
                video_path,
                sampling_rate=None,  # Use adaptive sampling
                target_memory_gb=4.0  # Adjust based on your system's capabilities
            )
            
            # Update progress to 100% since decord extracts frames in batch
            await self._update_progress(
                video_id,
                ProcessingStage.FRAME_EXTRACTION,
                100,
                details={
                    "frames_processed": len(frames),
                    "fps": fps
                }
            )
            
            return frames
            
        except Exception as e:
            await self._update_progress(
                video_id,
                ProcessingStage.FRAME_EXTRACTION,
                0,
                error=str(e)
            )
            raise VideoProcessingError(f"Frame extraction failed: {str(e)}")
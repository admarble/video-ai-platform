import asyncio
import logging
import time
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from ..models.processing import (
    ProcessingProgress,
    ProcessingStage,
    ValidationLevel
)
from ..models.results import ProcessingResult
from ..exceptions import VideoProcessingError
from ..utils.video import _extract_frames
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification
import torch
from dataclasses import dataclass
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

@dataclass
class SceneAnalysisResult:
    """Container for scene analysis results"""
    scene_labels: List[str]
    confidence_scores: List[float]
    temporal_segments: List[Tuple[int, int]]
    scene_embeddings: np.ndarray

class SceneAnalysisError(Exception):
    """Custom exception for scene analysis errors"""
    pass

def _process_scenes(
    frames: np.ndarray,
    batch_size: int = 8,
    threshold: float = 0.5,
    temporal_window: int = 16
) -> SceneAnalysisResult:
    """
    Analyzes video scenes using VideoMAE model.
    
    Args:
        frames (np.ndarray): Array of video frames (num_frames, height, width, channels)
        batch_size (int): Number of frame sequences to process at once
        threshold (float): Confidence threshold for scene detection
        temporal_window (int): Number of frames to analyze together
    
    Returns:
        SceneAnalysisResult: Contains scene labels, confidence scores, and temporal segments
    
    Raises:
        SceneAnalysisError: If scene analysis fails
    """
    try:
        # Initialize VideoMAE model and feature extractor
        feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-kinetics")
        model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-kinetics")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        num_frames = frames.shape[0]
        all_scene_labels = []
        all_confidence_scores = []
        all_temporal_segments = []
        all_embeddings = []
        
        # Process video in temporal windows
        for start_idx in range(0, num_frames - temporal_window + 1, temporal_window // 2):
            end_idx = min(start_idx + temporal_window, num_frames)
            frame_sequence = frames[start_idx:end_idx]
            
            # Prepare input for VideoMAE
            inputs = feature_extractor(
                frame_sequence,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                embeddings = outputs.hidden_states[-1][:, 0, :]  # Use CLS token embedding
                
                # Get predicted labels and scores
                probs = torch.nn.functional.softmax(logits, dim=-1)
                top_probs, top_labels = torch.max(probs, dim=-1)
                
                # Convert to numpy and get label names
                scene_probs = top_probs.cpu().numpy()
                scene_indices = top_labels.cpu().numpy()
                scene_names = [model.config.id2label[idx] for idx in scene_indices]
                
                # Store results
                all_scene_labels.extend(scene_names)
                all_confidence_scores.extend(scene_probs.tolist())
                all_temporal_segments.append((start_idx, end_idx))
                all_embeddings.append(embeddings.cpu().numpy())
        
        # Combine embeddings
        scene_embeddings = np.concatenate(all_embeddings, axis=0)
        
        # Filter results by confidence threshold
        filtered_results = [
            (label, score, segment)
            for label, score, segment in zip(
                all_scene_labels, 
                all_confidence_scores, 
                all_temporal_segments
            )
            if score >= threshold
        ]
        
        # Unzip filtered results
        labels, scores, segments = zip(*filtered_results) if filtered_results else ([], [], [])
        
        logging.info(
            f"Scene analysis complete: detected {len(labels)} scenes "
            f"above threshold {threshold}"
        )
        
        return SceneAnalysisResult(
            scene_labels=list(labels),
            confidence_scores=list(scores),
            temporal_segments=list(segments),
            scene_embeddings=scene_embeddings
        )
        
    except Exception as e:
        raise SceneAnalysisError(f"Failed to analyze scenes: {str(e)}")

@dataclass
class SceneSegment:
    """Represents a detected scene segment"""
    start_frame: int
    end_frame: int
    scene_type: str
    confidence: float
    keyframe_index: int

class SceneProcessor:
    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base-finetuned-kinetics",
        threshold: float = 0.5,
        min_segment_frames: int = 30
    ):
        try:
            self.feature_extractor = VideoMAEFeatureExtractor.from_pretrained(model_name)
            self.model = VideoMAEForVideoClassification.from_pretrained(model_name)
            self.threshold = threshold
            self.min_segment_frames = min_segment_frames
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        except Exception as e:
            raise SceneAnalysisError(f"Failed to initialize SceneProcessor: {str(e)}")

    def _compute_frame_differences(
        self,
        frames: np.ndarray,
        method: str = "histogram"
    ) -> np.ndarray:
        """
        Compute frame-to-frame differences for temporal segmentation.
        
        Args:
            frames: numpy array of video frames (N, H, W, C)
            method: difference method ('histogram' or 'pixel')
            
        Returns:
            Array of frame difference scores
        """
        differences = []
        
        for i in range(1, len(frames)):
            if method == "histogram":
                # Compute color histogram difference
                prev_hist = np.histogram(frames[i-1], bins=64, range=(0,255))[0]
                curr_hist = np.histogram(frames[i], bins=64, range=(0,255))[0]
                diff = np.sum(np.abs(prev_hist - curr_hist))
            else:
                # Compute pixel-wise difference
                diff = np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))
            
            differences.append(diff)
            
        return np.array(differences)
    
    def _detect_scene_boundaries(
        self,
        frame_differences: np.ndarray,
        prominence: float = 0.5
    ) -> List[int]:
        """
        Detect scene boundaries using peak detection on frame differences.
        
        Args:
            frame_differences: Array of frame difference scores
            prominence: Required prominence of peaks
            
        Returns:
            List of frame indices where scene changes occur
        """
        # Normalize differences
        normalized_diffs = frame_differences / np.max(frame_differences)
        
        # Find peaks in frame differences
        peaks, _ = find_peaks(normalized_diffs, prominence=prominence)
        
        # Filter peaks by minimum segment length
        filtered_peaks = []
        last_peak = -self.min_segment_frames
        
        for peak in peaks:
            if peak - last_peak >= self.min_segment_frames:
                filtered_peaks.append(peak)
                last_peak = peak
                
        return filtered_peaks
    
    def _classify_segment(
        self,
        frames: np.ndarray,
        start_idx: int,
        end_idx: int
    ) -> Tuple[str, float, int]:
        """
        Classify a video segment using VideoMAE.
        
        Args:
            frames: Full array of video frames
            start_idx: Start frame index of segment
            end_idx: End frame index of segment
            
        Returns:
            Tuple of (scene_type, confidence, keyframe_index)
        """
        # Select frames from segment
        segment_frames = frames[start_idx:end_idx]
        
        # Sample frames if segment is too long
        if len(segment_frames) > 16:  # VideoMAE typically uses 16 frames
            indices = np.linspace(0, len(segment_frames)-1, 16, dtype=int)
            segment_frames = segment_frames[indices]
        
        # Prepare frames for model
        inputs = self.feature_extractor(
            list(segment_frames),
            return_tensors="pt"
        ).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits[0], dim=0)
            
        # Get predicted class and confidence
        confidence, pred_idx = torch.max(probs, dim=0)
        scene_type = self.model.config.id2label[pred_idx.item()]
        
        # Find keyframe (middle frame of segment)
        keyframe_idx = start_idx + (end_idx - start_idx) // 2
        
        return scene_type, confidence.item(), keyframe_idx
    
    def process_scenes(
        self,
        frames: np.ndarray
    ) -> List[SceneSegment]:
        try:
            logging.info(f"Processing {len(frames)} frames for scene analysis")
            
            # Detect scene boundaries
            frame_differences = self._compute_frame_differences(frames)
            scene_boundaries = self._detect_scene_boundaries(frame_differences)
            
            # Add start and end frames
            all_boundaries = [0] + scene_boundaries + [len(frames)]
            
            # Process each segment
            scenes = []
            for i in range(len(all_boundaries) - 1):
                start_idx = all_boundaries[i]
                end_idx = all_boundaries[i + 1]
                
                # Skip if segment is too short
                if end_idx - start_idx < self.min_segment_frames:
                    continue
                
                # Classify segment
                scene_type, confidence, keyframe_idx = self._classify_segment(
                    frames, start_idx, end_idx
                )
                
                # Create scene segment
                scene = SceneSegment(
                    start_frame=start_idx,
                    end_frame=end_idx,
                    scene_type=scene_type,
                    confidence=confidence,
                    keyframe_index=keyframe_idx
                )
                
                scenes.append(scene)
                
            logging.info(f"Detected {len(scenes)} scenes in video")
            return scenes
        except Exception as e:
            raise SceneAnalysisError(f"Scene processing failed: {str(e)}")

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

    async def _process_scenes(self, frames: np.ndarray) -> SceneAnalysisResult:
        """Process scenes using the SceneProcessor"""
        try:
            scene_processor = SceneProcessor()
            scenes = scene_processor.process_scenes(frames)
            
            # Convert SceneSegment results to SceneAnalysisResult format
            return SceneAnalysisResult(
                scene_labels=[scene.scene_type for scene in scenes],
                confidence_scores=[scene.confidence for scene in scenes],
                temporal_segments=[(scene.start_frame, scene.end_frame) for scene in scenes],
                scene_embeddings=np.array([])  # TODO: Add embeddings if needed
            )
        except Exception as e:
            raise SceneAnalysisError(f"Scene analysis failed: {str(e)}")
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
import logging
from dataclasses import dataclass
from ..exceptions import ProcessingError

@dataclass
class SceneAnalysisResult:
    """Container for scene analysis results"""
    scene_labels: List[str]
    confidence_scores: List[float]
    temporal_segments: List[Tuple[int, int]]
    scene_embeddings: np.ndarray

class SceneProcessor:
    """Processor class for video scene analysis using VideoMAE"""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the scene processor.
        
        Args:
            device (str, optional): Device to run model on ('cuda' or 'cpu'). 
                                  If None, automatically selects available device.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize VideoMAE model and feature extractor"""
        try:
            self.feature_extractor = VideoMAEFeatureExtractor.from_pretrained(
                "MCG-NJU/videomae-base-kinetics"
            )
            self.model = VideoMAEForVideoClassification.from_pretrained(
                "MCG-NJU/videomae-base-kinetics"
            ).to(self.device)
            self.model.eval()
            
        except Exception as e:
            raise ProcessingError(f"Failed to initialize VideoMAE model: {str(e)}")
    
    def process_scenes(
        self,
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
            ProcessingError: If scene analysis fails
        """
        try:
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
                inputs = self.feature_extractor(
                    frame_sequence,
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get model predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    embeddings = outputs.hidden_states[-1][:, 0, :]  # Use CLS token embedding
                    
                    # Get predicted labels and scores
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    top_probs, top_labels = torch.max(probs, dim=-1)
                    
                    # Convert to numpy and get label names
                    scene_probs = top_probs.cpu().numpy()
                    scene_indices = top_labels.cpu().numpy()
                    scene_names = [
                        self.model.config.id2label[idx] 
                        for idx in scene_indices
                    ]
                    
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
            raise ProcessingError(f"Failed to analyze scenes: {str(e)}")

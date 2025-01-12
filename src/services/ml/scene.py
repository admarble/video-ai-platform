from typing import List
import numpy as np
from src.models.video import SceneSegment

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
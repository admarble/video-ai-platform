from transformers import CLIPProcessor, CLIPModel
from src.models.domain.query import QueryResult
import torch
from typing import List
import numpy as np

class VideoQueryAssistant:
    def __init__(self, model_manager):
        self.clip = model_manager.get_model("clip")
        self.processor = model_manager.get_processor("clip")

    async def process_query(self, query: str, video_frames: List[np.ndarray]) -> QueryResult:
        """Process user query using CLIP for video-text alignment"""
        # Encode text query
        text_inputs = self.processor(
            text=[query],
            return_tensors="pt",
            padding=True
        )
        
        # Encode video frames
        frame_inputs = self.processor(
            images=video_frames,
            return_tensors="pt",
            padding=True
        )
        
        # Get similarity scores
        with torch.inference_mode():
            text_features = self.clip.get_text_features(**text_inputs)
            image_features = self.clip.get_image_features(**frame_inputs)
            
            similarity = torch.nn.functional.cosine_similarity(
                text_features[:, None],
                image_features[None, :],
                dim=-1
            )
        
        # Find most relevant frames
        relevant_indices = torch.topk(similarity, k=5).indices[0]
        
        return QueryResult(
            relevant_frames=relevant_indices.tolist(),
            similarity_scores=similarity[0, relevant_indices].tolist()
        )
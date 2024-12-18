from typing import List, Dict, Any
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor

from .base_model import BaseModel

class ObjectDetector(BaseModel):
    """Detects objects in video frames using DETR"""
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        super().__init__(model_name, device)
        self.batch_size = batch_size
        self._load_model()
        
    def _load_model(self) -> None:
        self.model = DetrForObjectDetection.from_pretrained(
            self.model_name
        ).to(self.device)
        self.processor = DetrImageProcessor.from_pretrained(
            self.model_name
        )
        
    def detect_objects(
        self,
        frames: torch.Tensor,
        confidence_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in video frames
        
        Args:
            frames: Tensor of shape (B, C, H, W)
            confidence_threshold: Minimum confidence score
            
        Returns:
            List of dictionaries containing detections
        """
        inputs = self.processor(
            images=frames,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Post-process outputs
        results = []
        for frame_idx in range(len(frames)):
            scores = outputs.logits[frame_idx].softmax(-1)
            boxes = outputs.pred_boxes[frame_idx]
            
            # Filter by confidence
            mask = scores.max(-1).values > confidence_threshold
            
            results.append({
                'boxes': boxes[mask].cpu().numpy(),
                'scores': scores[mask].cpu().numpy(),
                'labels': scores[mask].argmax(-1).cpu().numpy()
            })
            
        return results 
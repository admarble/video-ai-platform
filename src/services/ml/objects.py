import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging

@dataclass
class DetectedObject:
    """Represents a detected object in a frame"""
    label: str
    confidence: float
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized coordinates
    frame_idx: int
    track_id: Optional[int] = None

class ObjectDetector:
    def __init__(
        self,
        model_manager,
        model_name: str = "facebook/detr-resnet-50",
        confidence_threshold: float = 0.7,
        device: Optional[str] = None
    ):
        """Initialize DETR object detector."""
        self.model_manager = model_manager
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and processor through model manager
        self.processor = self.model_manager.load_processor(
            model_name, 
            DetrImageProcessor
        )
        self.model = self.model_manager.load_model(
            model_name, 
            DetrForObjectDetection
        )
        self.model.to(self.device)
        
        logging.info(f"Initialized DETR object detector on {self.device}")

    @torch.inference_mode()
    async def detect(self, frames) -> List[Dict]:
        """Detect objects in video frames"""
        inputs = self.processor(frames, return_tensors="pt")
        outputs = await self.model(**inputs)
        
        # Process detections
        return self._process_detections(outputs)

    def _process_detections(self, outputs) -> List[Dict]:
        """Process model outputs into structured format"""
        results = []
        for frame_output in outputs:
            boxes = frame_output.pred_boxes.tolist()
            scores = frame_output.scores.tolist()
            labels = frame_output.pred_labels.tolist()
            
            frame_detections = [
                {
                    "label": self.model.config.id2label[label],
                    "score": score,
                    "box": box
                }
                for label, score, box in zip(labels, scores, boxes)
                if score > 0.5  # Confidence threshold
            ]
            results.append({"detections": frame_detections})
        
        return results
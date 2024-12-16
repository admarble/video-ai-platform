import torch
from typing import List, Dict

class ObjectDetector:
    def __init__(self, model_manager):
        self.model = model_manager.get_model("object")
        self.processor = model_manager.get_processor("object")

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
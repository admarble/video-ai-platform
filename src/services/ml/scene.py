import torch
import numpy as np
from typing import List, Dict

class SceneAnalyzer:
    def __init__(self, model_manager):
        self.model = model_manager.get_model("scene")
        self.processor = model_manager.get_processor("scene")

    @torch.inference_mode()
    async def analyze(self, frames: np.ndarray) -> List[Dict]:
        """Analyze video frames for scene classification"""
        inputs = self.processor(frames, return_tensors="pt")
        outputs = await self.model(**inputs)
        
        # Process model outputs
        predictions = outputs.logits.softmax(dim=-1)
        return self._format_predictions(predictions)

    def _format_predictions(self, predictions: torch.Tensor) -> List[Dict]:
        """Format model predictions into structured output"""
        results = []
        for pred in predictions:
            top_scores, top_indices = torch.topk(pred, k=3)
            results.append({
                "scenes": [
                    {"label": self.model.config.id2label[idx.item()],
                     "score": score.item()}
                    for idx, score in zip(top_indices, top_scores)
                ]
            })
        return results
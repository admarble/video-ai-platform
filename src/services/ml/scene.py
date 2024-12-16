from transformers import VideoMAEForVideoClassification
import torch

class SceneAnalyzer:
    def __init__(self, model_manager):
        self.model = model_manager.models["scene"]
        self.processor = model_manager.processors["scene"]

    @torch.inference_mode()  # Optimization for inference
    async def analyze(self, frames):
        inputs = self.processor(frames, return_tensors="pt")
        outputs = await self.model(**inputs)
        return self._process_outputs(outputs) 
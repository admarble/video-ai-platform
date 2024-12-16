from transformers import AutoModel, AutoProcessor
from typing import Dict, Any
import torch

class ModelManager:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.processors: Dict[str, Any] = {}
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all HF models with optimization"""
        self.models["scene"] = AutoModel.from_pretrained(
            "MCG-NJU/videomae-base",
            torch_dtype=torch.float16  # Use half precision
        )
        self.models["object"] = AutoModel.from_pretrained(
            "facebook/detr-resnet-50",
            device_map="auto"  # Automatic device optimization
        ) 
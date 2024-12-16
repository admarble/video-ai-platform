from transformers import AutoModel, AutoProcessor
from typing import Dict, Any
import torch
from src.core.config import settings

class ModelManager:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.processors: Dict[str, Any] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all HF models with optimization"""
        # Scene Analysis Model
        self.models["scene"] = AutoModel.from_pretrained(
            settings.SCENE_MODEL,
            torch_dtype=torch.float16
        ).to(self.device)
        
        # Object Detection Model
        self.models["object"] = AutoModel.from_pretrained(
            settings.OBJECT_MODEL,
            device_map="auto"
        )
        
        # CLIP Model for text-video alignment
        self.models["clip"] = AutoModel.from_pretrained(
            settings.CLIP_MODEL,
            torch_dtype=torch.float16
        ).to(self.device)

        self._initialize_processors()
    
    def _initialize_processors(self):
        """Initialize model processors"""
        self.processors["scene"] = AutoProcessor.from_pretrained(settings.SCENE_MODEL)
        self.processors["object"] = AutoProcessor.from_pretrained(settings.OBJECT_MODEL)
        self.processors["clip"] = AutoProcessor.from_pretrained(settings.CLIP_MODEL)
    
    def get_model(self, model_type: str):
        """Get model by type"""
        return self.models.get(model_type)
    
    def get_processor(self, model_type: str):
        """Get processor by type"""
        return self.processors.get(model_type)
from transformers import AutoModel, AutoProcessor, AutoFeatureExtractor
from typing import Dict, Any
import torch
from src.core.config import settings

class ModelManager:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.processors: Dict[str, Any] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() and settings.USE_CUDA else "cpu")
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all models with optimization settings"""
        # Scene Analysis Model
        self.models["scene"] = AutoModel.from_pretrained(
            settings.SCENE_MODEL,
            torch_dtype=torch.float16 if settings.FP16 else torch.float32,
            device_map="auto"
        )
        
        # Object Detection Model
        self.models["object"] = AutoModel.from_pretrained(
            settings.OBJECT_MODEL,
            device_map="auto"
        )
        
        # Audio Processing Model
        self.models["audio"] = AutoModel.from_pretrained(
            settings.AUDIO_MODEL,
            device_map="auto"
        )
        
        # CLIP Model
        self.models["clip"] = AutoModel.from_pretrained(
            settings.CLIP_MODEL,
            device_map="auto"
        )
        
        # Initialize processors
        self._initialize_processors()
    
    def _initialize_processors(self):
        """Initialize model processors and feature extractors"""
        self.processors["scene"] = AutoProcessor.from_pretrained(settings.SCENE_MODEL)
        self.processors["object"] = AutoFeatureExtractor.from_pretrained(settings.OBJECT_MODEL)
        self.processors["audio"] = AutoProcessor.from_pretrained(settings.AUDIO_MODEL)
        self.processors["clip"] = AutoProcessor.from_pretrained(settings.CLIP_MODEL)
    
    def get_model(self, name: str) -> Any:
        """Get a model by name"""
        return self.models[name]
    
    def get_processor(self, name: str) -> Any:
        """Get a processor by name"""
        return self.processors[name]
    
    def optimize_models(self):
        """Apply optimization techniques to all models"""
        if settings.FP16:
            for model in self.models.values():
                model.half()
        
        for model in self.models.values():
            model.eval()  # Set to evaluation mode
            model.to(self.device)
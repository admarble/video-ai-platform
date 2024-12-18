from typing import Any, Optional
import torch
from transformers import AutoModel

class BaseModel:
    """Base class for all ML models"""
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
    def _load_model(self) -> None:
        """Load the model - to be implemented by child classes"""
        raise NotImplementedError
        
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to appropriate device"""
        return tensor.to(self.device)
        
    def cleanup(self) -> None:
        """Cleanup resources"""
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache() 
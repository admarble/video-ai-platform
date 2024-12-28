from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import torch
import numpy as np
from transformers import TimesformerModel, TimesformerConfig
import logging

@dataclass
class TimesformerSettings:
    num_frames: int = 8
    image_size: int = 224
    patch_size: int = 16
    hidden_size: int = 768
    num_attention_heads: int = 12
    batch_size: int = 4

class TimesformerProcessor:
    def __init__(self, settings: TimesformerSettings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = TimesformerModel.from_pretrained(
                "facebook/timesformer-base-finetuned-k400",
                num_frames=settings.num_frames,
                image_size=settings.image_size
            ).to(self.device)
            self.model.eval()
        except Exception as e:
            self.logger.error(f"Failed to initialize TimeSformer model: {str(e)}")
            raise

    def _create_temporal_windows(
        self,
        frames: np.ndarray,
        window_size: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Create overlapping temporal windows from input frames."""
        if window_size is None:
            window_size = self.settings.num_frames
            
        windows = []
        stride = window_size // 2  # 50% overlap
        
        for i in range(0, len(frames) - window_size + 1, stride):
            window = frames[i:i + window_size]
            # Convert to torch tensor and normalize
            window = torch.from_numpy(window).float()
            window = window.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
            window = window / 255.0  # Normalize to [0, 1]
            windows.append(window)
            
        return windows

    def process_frames(
        self,
        frames: np.ndarray,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process video frames through TimeSformer model."""
        if batch_size is None:
            batch_size = self.settings.batch_size
            
        try:
            # Create temporal windows
            windows = self._create_temporal_windows(frames)
            
            results = []
            for i in range(0, len(windows), batch_size):
                batch = windows[i:i + batch_size]
                batch = torch.stack(batch).to(self.device)
                
                with torch.no_grad():
                    features = self.model(
                        pixel_values=batch,
                        return_dict=True
                    )
                    
                results.extend(self.extract_temporal_features(features))
                
            return self._aggregate_results(results)
            
        except Exception as e:
            self.logger.error(f"Error processing frames: {str(e)}")
            return self.handle_processing_error(e, {'frames': frames})

    def extract_temporal_features(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        """Extract relevant features from model output."""
        # Extract frame-level features
        frame_features = features['last_hidden_state']
        
        # Extract temporal attention maps
        temporal_attention = features['attentions']
        
        # Process spatial-temporal patterns
        patterns = self._process_attention_patterns(temporal_attention)
        
        return {
            'frame_features': frame_features.cpu().numpy(),
            'temporal_patterns': patterns,
            'attention_maps': temporal_attention[-1].cpu().numpy()  # Use last layer
        }

    def _process_attention_patterns(
        self,
        attention_maps: torch.Tensor
    ) -> np.ndarray:
        """Process attention patterns to extract temporal relationships."""
        # Take mean across heads and layers
        mean_attention = torch.mean(attention_maps, dim=(0, 1))
        
        # Extract temporal relationships
        temporal_patterns = mean_attention[:self.settings.num_frames,
                                        :self.settings.num_frames]
        
        return temporal_patterns.cpu().numpy()

    def _aggregate_results(
        self,
        results: List[Dict[str, np.ndarray]]
    ) -> Dict[str, Any]:
        """Aggregate results from multiple windows."""
        if not results:
            return {}
            
        aggregated = {
            'frame_features': np.concatenate([r['frame_features'] for r in results]),
            'temporal_patterns': np.mean([r['temporal_patterns'] for r in results], axis=0),
            'attention_maps': np.concatenate([r['attention_maps'] for r in results])
        }
        
        return aggregated

    def handle_processing_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle processing errors and return partial results if possible."""
        self.logger.error(f"Processing error: {str(error)}")
        
        # Try to free up GPU memory
        if isinstance(error, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            
            # Try processing with smaller batch size
            if context.get('frames') is not None:
                try:
                    return self.process_frames(
                        context['frames'],
                        batch_size=self.settings.batch_size // 2
                    )
                except Exception as e:
                    self.logger.error(f"Recovery attempt failed: {str(e)}")
        
        return {} 
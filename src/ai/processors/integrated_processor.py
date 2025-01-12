from typing import Dict, List, Optional, Any, Tuple
import torch
import numpy as np
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from .timesformer_processor import TimesformerProcessor, TimesformerSettings
from .yolo_processor import YOLOProcessor, YOLOSettings
from .whisper_processor import WhisperProcessor, WhisperSettings

@dataclass
class IntegratedSettings:
    timesformer: TimesformerSettings = TimesformerSettings()
    yolo: YOLOSettings = YOLOSettings()
    whisper: WhisperSettings = WhisperSettings()
    max_workers: int = 3
    device: Optional[str] = None

class IntegratedProcessor:
    def __init__(self, settings: IntegratedSettings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.device = settings.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Initialize individual processors
            self.timesformer = TimesformerProcessor(settings.timesformer, device=self.device)
            self.yolo = YOLOProcessor(settings.yolo, device=self.device)
            self.whisper = WhisperProcessor(settings.whisper, device=self.device)
            
            # Initialize thread pool for parallel processing
            self.executor = ThreadPoolExecutor(max_workers=settings.max_workers)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize integrated processor: {str(e)}")
            raise

    def process_video(
        self,
        frames: np.ndarray,
        audio_path: str,
        fps: float
    ) -> Dict[str, Any]:
        """Process video frames and audio in parallel."""
        try:
            # Create futures for parallel processing
            futures = {
                'temporal': self.executor.submit(
                    self.timesformer.process_frames, frames
                ),
                'objects': self.executor.submit(
                    self.yolo.detect_objects, frames
                ),
                'audio': self.executor.submit(
                    self.whisper.process_audio, audio_path
                )
            }
            
            # Collect results
            results = {}
            for name, future in futures.items():
                try:
                    results[name] = future.result()
                except Exception as e:
                    self.logger.error(f"Error in {name} processing: {str(e)}")
                    results[name] = {}
            
            # Process spatial relationships if object detection succeeded
            if results.get('objects'):
                results['spatial'] = self.yolo.analyze_spatial_relationships(
                    results['objects']
                )
            
            # Align audio timestamps with frames if audio processing succeeded
            if results.get('audio'):
                results['aligned_audio'] = self.whisper.align_timestamps(
                    results['audio'],
                    fps
                )
            
            # Combine all results
            return self._combine_results(results)
            
        except Exception as e:
            self.logger.error(f"Error in integrated processing: {str(e)}")
            return self.handle_processing_error(e, {
                'frames': frames,
                'audio_path': audio_path
            })

    def _combine_results(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine results from all processors into a unified format."""
        combined = {
            'temporal_analysis': results.get('temporal', {}),
            'object_detection': results.get('objects', []),
            'spatial_relationships': results.get('spatial', {}),
            'audio_transcription': results.get('audio', {}),
            'frame_alignments': results.get('aligned_audio', {})
        }
        
        # Add metadata
        combined['metadata'] = {
            'success_status': {
                'temporal': bool(results.get('temporal')),
                'objects': bool(results.get('objects')),
                'spatial': bool(results.get('spatial')),
                'audio': bool(results.get('audio')),
                'alignments': bool(results.get('aligned_audio'))
            }
        }
        
        return combined

    def handle_processing_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle processing errors and return partial results if possible."""
        self.logger.error(f"Processing error: {str(e)}")
        
        if isinstance(error, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            
            # Try processing with reduced settings
            try:
                # Reduce batch sizes
                self.settings.timesformer.batch_size //= 2
                self.settings.yolo.batch_size //= 2
                self.settings.whisper.batch_size //= 2
                
                # Retry processing
                return self.process_video(
                    context['frames'],
                    context['audio_path']
                )
            except Exception as e:
                self.logger.error(f"Recovery attempt failed: {str(e)}")
        
        return {
            'temporal_analysis': {},
            'object_detection': [],
            'spatial_relationships': {},
            'audio_transcription': {},
            'frame_alignments': {},
            'metadata': {
                'error': str(error),
                'success_status': {
                    'temporal': False,
                    'objects': False,
                    'spatial': False,
                    'audio': False,
                    'alignments': False
                }
            }
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True) 
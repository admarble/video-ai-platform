from typing import Dict, List, Optional, Any, Tuple
import torch
import numpy as np
import whisper
from whisper_timestamped import transcribe
import logging
from dataclasses import dataclass

@dataclass
class WhisperSettings:
    model_size: str = "base"  # tiny, base, small, medium, large
    language: Optional[str] = None
    batch_size: int = 16
    compute_type: str = "float16"  # float32, float16, int8

class WhisperProcessor:
    def __init__(
        self,
        settings: WhisperSettings,
        device: Optional[str] = None
    ):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            self.model = whisper.load_model(
                settings.model_size,
                device=self.device,
                compute_type=settings.compute_type
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Whisper model: {str(e)}")
            raise

    def process_audio(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Transcribe audio with word-level timestamps."""
        try:
            # Use provided language or detect it
            lang = language or self.settings.language
            
            # Transcribe with timestamps
            result = transcribe(
                audio_path,
                self.model,
                language=lang,
                batch_size=self.settings.batch_size,
                compute_type=self.settings.compute_type
            )
            
            # Extract word-level alignments
            words = self._extract_word_alignments(result)
            
            return {
                'segments': result['segments'],
                'words': words,
                'language': result['language']
            }
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {str(e)}")
            return self.handle_processing_error(e, {'audio_path': audio_path})

    def _extract_word_alignments(
        self,
        result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract word-level alignments from transcription result."""
        words = []
        
        for segment in result['segments']:
            for word in segment['words']:
                words.append({
                    'text': word['text'],
                    'start': word['start'],
                    'end': word['end'],
                    'confidence': word['confidence'],
                    'speaker': segment.get('speaker'),
                    'segment_id': segment.get('id')
                })
                
        return words

    def align_timestamps(
        self,
        transcription: Dict[str, Any],
        fps: float
    ) -> Dict[str, Any]:
        """Align word timestamps with video frames."""
        try:
            # Convert time to frame indices
            frame_aligned_words = []
            for word in transcription['words']:
                aligned_word = word.copy()
                aligned_word.update({
                    'start_frame': int(word['start'] * fps),
                    'end_frame': int(word['end'] * fps)
                })
                frame_aligned_words.append(aligned_word)
            
            # Create frame-to-word mapping
            frame_words = self._create_frame_word_mapping(frame_aligned_words)
            
            return {
                'aligned_words': frame_aligned_words,
                'frame_word_map': frame_words
            }
            
        except Exception as e:
            self.logger.error(f"Error aligning timestamps: {str(e)}")
            return {}

    def _create_frame_word_mapping(
        self,
        aligned_words: List[Dict[str, Any]]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Create mapping from frame indices to words."""
        frame_words = {}
        
        for word in aligned_words:
            for frame in range(word['start_frame'], word['end_frame'] + 1):
                if frame not in frame_words:
                    frame_words[frame] = []
                frame_words[frame].append(word)
                
        return frame_words

    def handle_processing_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle processing errors and return partial results if possible."""
        self.logger.error(f"Processing error: {str(error)}")
        
        if isinstance(error, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            
            # Try processing with smaller batch size
            if context.get('audio_path'):
                try:
                    # Reduce batch size and switch to float32
                    self.settings.batch_size //= 2
                    self.settings.compute_type = "float32"
                    return self.process_audio(context['audio_path'])
                except Exception as e:
                    self.logger.error(f"Recovery attempt failed: {str(e)}")
        
        return {} 
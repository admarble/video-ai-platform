from dataclasses import dataclass
from typing import Optional

@dataclass
class AudioSegment:
    """Represents a processed audio segment with transcription"""
    start_time: float
    end_time: float
    text: str
    confidence: float

class AudioProcessor:
    """Basic audio processor for testing"""
    def __init__(self, model_name: str = "test-model", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or "cpu"
    
    def process_audio(self, audio_path: str) -> list[AudioSegment]:
        """Dummy implementation for testing"""
        return [
            AudioSegment(
                start_time=0.0,
                end_time=1.0,
                text="Test transcription",
                confidence=0.9
            )
        ] 
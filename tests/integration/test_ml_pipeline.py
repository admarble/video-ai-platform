import numpy as np
import pytest
from src.processors.video_processor import _process_scenes, SceneAnalysisError
from pathlib import Path
from src.services.video.processor import AudioProcessor, AudioProcessingError

def test_process_scenes():
    # Create dummy video frames
    frames = np.random.rand(32, 224, 224, 3).astype(np.float32)
    
    # Process scenes
    result = _process_scenes(
        frames=frames,
        batch_size=8,
        threshold=0.5,
        temporal_window=16
    )
    
    # Verify result structure
    assert hasattr(result, 'scene_labels')
    assert hasattr(result, 'confidence_scores')
    assert hasattr(result, 'temporal_segments')
    assert hasattr(result, 'scene_embeddings')
    
    # Verify data types and shapes
    assert isinstance(result.scene_labels, list)
    assert isinstance(result.confidence_scores, list)
    assert isinstance(result.temporal_segments, list)
    assert isinstance(result.scene_embeddings, np.ndarray)
    
    # Verify confidence scores are within valid range
    assert all(0 <= score <= 1 for score in result.confidence_scores)

def test_process_scenes_invalid_input():
    # Test with invalid input
    with pytest.raises(SceneAnalysisError):
        _process_scenes(np.zeros((1, 1, 1)))  # Invalid frame dimensions

def test_audio_processor_initialization():
    processor = AudioProcessor()
    assert processor.device in ['cuda', 'cpu']
    assert processor.sample_rate == 16000

@pytest.mark.integration
def test_audio_processing():
    processor = AudioProcessor()
    
    # Assuming you have a test video file
    video_path = Path("tests/fixtures/test_video.mp4")
    
    if not video_path.exists():
        pytest.skip("Test video file not found")
    
    segments = processor.process_audio(str(video_path))
    
    assert len(segments) > 0
    for segment in segments:
        assert segment.start_time >= 0
        assert segment.end_time > segment.start_time
        assert isinstance(segment.text, str)
        assert 0 <= segment.confidence <= 1

def test_invalid_video_path():
    processor = AudioProcessor()
    with pytest.raises(AudioProcessingError):
        processor.process_audio("nonexistent_video.mp4")
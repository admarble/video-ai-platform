import pytest
from pathlib import Path
import tempfile
import numpy as np

from src.services.ai.service_manager import ServiceManager
from src.models.audio_processor import AudioProcessor
from src.core.config import ModelConfig

@pytest.fixture
def service_manager():
    config = ModelConfig(gpu_enabled=False)
    manager = ServiceManager(config)
    manager.initialize_services(['scene_processor', 'object_detector'])
    yield manager
    manager.cleanup()

@pytest.fixture
def test_data():
    # Create test video frames
    frames = np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8)
    
    # Create temporary test file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(b'dummy video content')
        video_path = Path(f.name)
    
    yield {
        'frames': frames,
        'video_path': video_path
    }
    
    # Cleanup
    if video_path.exists():
        video_path.unlink()

def test_scene_processing(service_manager, test_data):
    """Test scene processing pipeline"""
    scenes = service_manager.get_service('scene_processor').process_scenes(test_data['frames'])
    assert len(scenes) > 0
    assert all(s.confidence > 0 for s in scenes)

def test_object_detection(service_manager, test_data):
    """Test object detection pipeline"""
    detections = service_manager.get_service('object_detector').process_frames(
        test_data['frames'],
        enable_tracking=True
    )
    assert len(detections) == len(test_data['frames'])
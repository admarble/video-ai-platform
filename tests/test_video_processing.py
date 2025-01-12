import pytest
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Generator

from src.services.ai.service_manager import (
    ServiceManager,
    _extract_frames
)
from src.services.ai.exceptions import VideoProcessingError
from src.models.video import SceneSegment

class VideoTestHelper:
    """Helper class for video processing tests"""
    
    @staticmethod
    def create_test_video(
        duration: int = 5,
        fps: int = 30,
        resolution: tuple = (640, 480)
    ) -> Path:
        """Create a test video file"""
        temp_dir = tempfile.mkdtemp()
        video_path = Path(temp_dir) / "test_video.mp4"
        
        # Create frames
        frames = np.random.randint(0, 255, (*resolution, 3), dtype=np.uint8)
        
        # Save dummy video file for testing
        with open(video_path, 'wb') as f:
            f.write(b'dummy video content')
            
        return video_path

@pytest.fixture
def video_processor():
    """Fixture for video processor with mocked models"""
    with patch('torch.cuda.is_available', return_value=False):
        processor = ServiceManager()
        processor.initialize_services(['scene_processor', 'object_detector'])
        yield processor
        processor.cleanup()

@pytest.fixture
def test_video() -> Generator[Path, None, None]:
    """Fixture for temporary test video"""
    video_path = VideoTestHelper.create_test_video()
    yield video_path
    # Cleanup
    if video_path.exists():
        video_path.unlink()
        video_path.parent.rmdir()

class TestPerformance:
    """Performance testing suite"""
    
    def test_frame_extraction_performance(self, test_video, benchmark):
        """Benchmark frame extraction performance"""
        def extract_frames():
            return _extract_frames(str(test_video))
            
        result = benchmark(extract_frames)
        assert result.stats.mean < 1.0  # Should complete within 1 second
        
    def test_memory_usage(self, video_processor, test_video):
        """Test memory usage during processing"""
        import psutil
        initial_memory = psutil.Process().memory_info().rss
        
        # Process video
        frames, _ = _extract_frames(str(test_video))
        video_processor.get_service('scene_processor').process_scenes(frames)
        
        final_memory = psutil.Process().memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        assert memory_increase < 1000  # Less than 1GB increase

class TestIntegration:
    """Integration testing suite"""
    
    def test_full_pipeline(self, video_processor, test_video):
        """Test complete video processing pipeline"""
        # Extract frames
        frames, fps = _extract_frames(str(test_video))
        
        # Process scenes
        scenes = video_processor.get_service('scene_processor').process_scenes(frames)
        
        # Detect objects
        detections = video_processor.get_service('object_detector').process_frames(
            frames,
            enable_tracking=True
        )
        
        # Verify results
        assert len(scenes) > 0
        assert all(isinstance(s, SceneSegment) for s in scenes)
        assert len(detections) == len(frames)
        
    def test_concurrent_processing(self, video_processor):
        """Test concurrent video processing"""
        videos = [VideoTestHelper.create_test_video() for _ in range(3)]
        
        start_time = time.time()
        
        # Process videos concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(self._process_single_video, video_processor, video)
                for video in videos
            ]
            
            results = [future.result() for future in futures]
            
        duration = time.time() - start_time
        assert duration < 10  # Should complete within 10 seconds
        
        # Cleanup
        for video in videos:
            if video.exists():
                video.unlink()
                video.parent.rmdir()
        
    def _process_single_video(self, processor, video_path):
        """Helper method for processing a single video"""
        frames, _ = _extract_frames(str(video_path))
        scenes = processor.get_service('scene_processor').process_scenes(frames)
        return scenes

def test_error_handling(video_processor):
    """Test error handling and recovery"""
    with pytest.raises(VideoProcessingError):
        _extract_frames("nonexistent_video.mp4")

@pytest.mark.benchmark
def test_batch_processing_performance(benchmark):
    """Benchmark batch processing performance"""
    frames = np.random.randint(0, 255, (100, 224, 224, 3), dtype=np.uint8)
    
    def process_batch():
        manager = ServiceManager()
        manager.initialize_services(['object_detector'])
        manager.batch_process_parallel(
            [frames[i:i+10] for i in range(0, len(frames), 10)],
            'object_detector',
            'process_frames',
            batch_size=10
        )
        manager.cleanup()
        
    benchmark(process_batch) 
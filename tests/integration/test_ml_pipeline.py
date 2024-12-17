import pytest
from src.services.ml import SceneAnalyzer, ObjectDetector
from src.services.video.processor import VideoProcessor

@pytest.mark.asyncio
async def test_parallel_processing():
    video_processor = VideoProcessor()
    result = await video_processor.process_video("test_video.mp4")
    
    assert "scenes" in result
    assert "objects" in result
    assert len(result["scenes"]) > 0 
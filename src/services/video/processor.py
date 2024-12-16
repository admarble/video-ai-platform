from src.core.model_loader import ModelManager
from src.services.ml import SceneAnalyzer, ObjectDetector
import asyncio

class VideoProcessor:
    def __init__(self):
        self.model_manager = ModelManager()
        self.scene_analyzer = SceneAnalyzer(self.model_manager)
        self.object_detector = ObjectDetector(self.model_manager)

    async def process_video(self, video_path: str):
        """Parallel video processing pipeline"""
        frames = self._extract_frames(video_path)
        
        # Parallel processing
        results = await asyncio.gather(
            self.scene_analyzer.analyze(frames),
            self.object_detector.detect(frames),
            self._process_audio(video_path)
        )
        
        return self._combine_results(results) 
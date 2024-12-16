from src.core.model_loader import ModelManager
from src.services.ml import SceneAnalyzer, ObjectDetector
from src.models.domain.video import ProcessingResult
import asyncio
import decord
import torch

class VideoProcessor:
    def __init__(self):
        self.model_manager = ModelManager()
        self.scene_analyzer = SceneAnalyzer(self.model_manager)
        self.object_detector = ObjectDetector(self.model_manager)

    async def process_video(self, video_path: str) -> ProcessingResult:
        """Process video through ML pipeline"""
        # Load video efficiently using decord
        video_reader = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total_frames = len(video_reader)
        
        # Process in batches
        batch_results = []
        for batch_idx in range(0, total_frames, self.batch_size):
            batch_frames = video_reader.get_batch(
                list(range(batch_idx, min(batch_idx + self.batch_size, total_frames)))
            ).asnumpy()
            
            # Parallel processing of batch
            batch_result = await asyncio.gather(
                self.scene_analyzer.analyze(batch_frames),
                self.object_detector.detect(batch_frames)
            )
            batch_results.append(batch_result)
        
        return self._combine_results(batch_results)

    def _combine_results(self, batch_results) -> ProcessingResult:
        """Combine batch results into final output"""
        scenes = []
        objects = []
        
        for batch in batch_results:
            scenes.extend(batch[0])
            objects.extend(batch[1])
        
        return ProcessingResult(
            scenes=scenes,
            objects=objects,
            metadata=self._extract_metadata()
        )
from transformers import CLIPProcessor, CLIPModel
from src.models.domain.query import QueryResult

class VideoQueryAssistant:
    def __init__(self, model_manager):
        self.clip = model_manager.models["clip"]
        self.processor = model_manager.processors["clip"]

    async def process_query(self, query: str, video_context: dict) -> QueryResult:
        """Process user query using CLIP for video-text alignment"""
        relevant_frames = await self._find_relevant_frames(query, video_context)
        response = await self._generate_response(query, relevant_frames)
        return QueryResult(response=response, clips=relevant_frames) 
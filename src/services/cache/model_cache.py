from functools import lru_cache
import torch

class ModelCache:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @lru_cache(maxsize=128)
    async def get_scene_embeddings(self, video_id: str):
        """Cache scene embeddings for faster retrieval"""
        return await self._compute_embeddings(video_id)

    def optimize_models(self):
        """Apply optimization techniques"""
        for model in self.models.values():
            model.half()  # Use FP16
            model.to(self.device) 
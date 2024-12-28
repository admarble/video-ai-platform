from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import timedelta

class TimeSegment(BaseModel):
    start_time: float
    end_time: float
    duration: float

class SceneContext(BaseModel):
    scene_type: str
    participants: List[str]
    action_description: str
    dialogue_summary: str
    emotions: List[str]
    location: Optional[str]

class VideoScene(BaseModel):
    segment: TimeSegment
    context: SceneContext
    transcript: str
    embedding_id: str
    previous_scene_id: Optional[str]
    next_scene_id: Optional[str]

class SearchQuery(BaseModel):
    query_text: str
    scene_type: Optional[str]
    temporal_context: bool = False
    max_segments: int = 3

class SearchResult(BaseModel):
    scenes: List[VideoScene]
    relevance_scores: Dict[str, float]
    context_summary: str

class SceneFeatures(BaseModel):
    motion_intensity: float
    visual_complexity: float
    dominant_colors: List[str]
    camera_movement: str

class RAGConfig(BaseModel):
    vector_store: Dict[str, Any]
    llm: Dict[str, Any]
    clip_generator: Dict[str, Any]

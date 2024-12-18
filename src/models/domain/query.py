from typing import List
from dataclasses import dataclass

@dataclass
class QueryResult:
    """Results from a video-text query"""
    relevant_frames: List[int]
    similarity_scores: List[float] 
from typing import Dict, Any, Optional, List
import numpy as np
from pathlib import Path

class VideoProcessorIntegration:
    def __init__(
        self,
        video_processor: 'VideoProcessor',
        rag_service: 'RAGService',
        cache_manager: 'CacheManager'
    ):
        self.video_processor = video_processor
        self.rag_service = rag_service
        self.cache_manager = cache_manager
        
    async def process_video_with_rag(
        self,
        video_path: str,
        extract_scenes: bool = True
    ) -> Dict[str, Any]:
        """Process video and prepare for RAG indexing"""
        
        # Check cache first
        cache_key = f"rag_processed_{video_path}"
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result
            
        # Process video using existing pipeline
        frames, fps = await self.video_processor.extract_frames(video_path)
        scenes = await self.video_processor.detect_scenes(frames)
        objects = await self.video_processor.detect_objects(frames)
        audio_segments = await self.video_processor.process_audio(video_path)
        
        # Prepare scene data for RAG
        scene_data = []
        for scene in scenes:
            # Get objects in scene timeframe
            scene_objects = self._get_objects_in_timeframe(
                objects, scene.start_frame, scene.end_frame
            )
            
            # Get audio/transcript for scene
            scene_audio = self._get_audio_in_timeframe(
                audio_segments, scene.start_time, scene.end_time
            )
            
            # Create scene context
            scene_data.append({
                'scene_id': scene.id,
                'timeframe': {
                    'start': scene.start_time,
                    'end': scene.end_time
                },
                'objects': scene_objects,
                'transcript': scene_audio.transcript,
                'context': scene.context
            })
        
        # Index scenes in RAG system
        await self.rag_service.index_scenes(scene_data)
        
        # Cache results
        await self.cache_manager.set(cache_key, scene_data)
        
        return scene_data

    def _get_objects_in_timeframe(
        self,
        objects: List[Dict[str, Any]],
        start_frame: int,
        end_frame: int
    ) -> List[Dict[str, Any]]:
        """Extract objects detected within a specific timeframe"""
        return [
            obj for obj in objects
            if start_frame <= obj['frame'] <= end_frame
        ]

    def _get_audio_in_timeframe(
        self,
        audio_segments: List[Dict[str, Any]],
        start_time: float,
        end_time: float
    ) -> Dict[str, Any]:
        """Extract audio segments within a specific timeframe"""
        relevant_segments = [
            segment for segment in audio_segments
            if (segment['start_time'] <= end_time and
                segment['end_time'] >= start_time)
        ]
        return {
            'transcript': ' '.join(seg['transcript'] for seg in relevant_segments),
            'segments': relevant_segments
        }

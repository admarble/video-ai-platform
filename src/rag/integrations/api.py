from typing import Dict, Any, Optional, List
from fastapi import HTTPException, BackgroundTasks
import asyncio

class VideoSearchAPI:
    def __init__(
        self,
        rag_service: 'RAGService',
        clip_generator: 'ClipGenerator',
        cache_manager: 'CacheManager'
    ):
        self.rag_service = rag_service
        self.clip_generator = clip_generator
        self.cache_manager = cache_manager
        
    async def search_video(
        self,
        query: str,
        video_id: str,
        temporal_context: bool = False,
        background_tasks: Optional[BackgroundTasks] = None
    ) -> Dict[str, Any]:
        """Search video content and return relevant clips"""
        try:
            # Check cache for query results
            cache_key = f"search_{video_id}_{query}"
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                return cached_result
            
            # Process search query
            search_results = await self.rag_service.search(
                query=query,
                video_id=video_id,
                include_context=temporal_context
            )
            
            if not search_results.scenes:
                return {
                    'status': 'no_results',
                    'message': 'No matching scenes found'
                }
            
            # Generate video clips
            clips = await self.clip_generator.create_clips(
                scenes=search_results.scenes,
                video_id=video_id
            )
            
            # Prepare response
            response = {
                'status': 'success',
                'clips': clips,
                'context': search_results.context_summary,
                'metadata': {
                    'query': query,
                    'video_id': video_id,
                    'temporal_context': temporal_context,
                    'scene_count': len(search_results.scenes)
                }
            }
            
            # Cache results
            if background_tasks:
                background_tasks.add_task(
                    self.cache_manager.set,
                    cache_key,
                    response
                )
            
            return response
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing video search: {str(e)}"
            )
            
    async def generate_clip(
        self,
        scene_id: str,
        video_id: str,
        start_time: float,
        end_time: float
    ) -> Dict[str, Any]:
        """Generate a video clip for a specific scene"""
        try:
            clip_data = await self.clip_generator.generate_clip(
                video_id=video_id,
                start_time=start_time,
                end_time=end_time,
                scene_id=scene_id
            )
            
            return {
                'status': 'success',
                'clip_url': clip_data['url'],
                'duration': clip_data['duration'],
                'format': clip_data['format']
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating clip: {str(e)}"
            )
            
    async def get_scene_context(
        self,
        scene_id: str,
        video_id: str
    ) -> Dict[str, Any]:
        """Get contextual information for a specific scene"""
        try:
            context = await self.rag_service.get_scene_context(
                scene_id=scene_id,
                video_id=video_id
            )
            
            return {
                'status': 'success',
                'context': context
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error retrieving scene context: {str(e)}"
            )

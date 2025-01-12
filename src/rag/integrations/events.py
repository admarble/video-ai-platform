from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime

class RAGEventHandler:
    def __init__(
        self,
        rag_service: 'RAGService',
        cache_manager: 'CacheManager'
    ):
        self.rag_service = rag_service
        self.cache_manager = cache_manager
        self._processing_queue = asyncio.Queue()
        self._is_processing = False
        
    async def start(self):
        """Start the event processing loop"""
        self._is_processing = True
        asyncio.create_task(self._process_events())
        
    async def stop(self):
        """Stop the event processing loop"""
        self._is_processing = False
        
    async def handle_video_processed(self, event: Dict[str, Any]):
        """Handle video processing completed event"""
        await self._processing_queue.put({
            'type': 'video_processed',
            'data': event,
            'timestamp': datetime.utcnow()
        })
        
    async def handle_scene_updated(self, event: Dict[str, Any]):
        """Handle scene update event"""
        await self._processing_queue.put({
            'type': 'scene_updated',
            'data': event,
            'timestamp': datetime.utcnow()
        })
        
    async def handle_index_updated(self, event: Dict[str, Any]):
        """Handle index update event"""
        await self._processing_queue.put({
            'type': 'index_updated',
            'data': event,
            'timestamp': datetime.utcnow()
        })
        
    async def _process_events(self):
        """Process events from the queue"""
        while self._is_processing:
            try:
                event = await self._processing_queue.get()
                await self._handle_event(event)
                self._processing_queue.task_done()
            except Exception as e:
                # Log error but continue processing
                print(f"Error processing event: {str(e)}")
                
    async def _handle_event(self, event: Dict[str, Any]):
        """Handle different event types"""
        handlers = {
            'video_processed': self._handle_video_processed_event,
            'scene_updated': self._handle_scene_updated_event,
            'index_updated': self._handle_index_updated_event
        }
        
        handler = handlers.get(event['type'])
        if handler:
            await handler(event['data'])
            
    async def _handle_video_processed_event(self, data: Dict[str, Any]):
        """Handle video processed event"""
        video_id = data['video_id']
        scene_data = data['scene_data']
        
        # Index new scenes
        await self.rag_service.index_scenes(scene_data)
        
        # Invalidate related caches
        await self._invalidate_caches(video_id)
        
    async def _handle_scene_updated_event(self, data: Dict[str, Any]):
        """Handle scene update event"""
        scene_id = data['scene_id']
        updates = data['updates']
        
        # Update scene in vector store
        await self.rag_service.update_scene(scene_id, updates)
        
        # Invalidate related caches
        if 'video_id' in data:
            await self._invalidate_caches(data['video_id'])
            
    async def _handle_index_updated_event(self, data: Dict[str, Any]):
        """Handle index update event"""
        index_id = data['index_id']
        
        # Refresh index metadata
        await self.rag_service.refresh_index_metadata(index_id)
        
        # Invalidate related caches
        if 'video_ids' in data:
            for video_id in data['video_ids']:
                await self._invalidate_caches(video_id)
                
    async def _invalidate_caches(self, video_id: str):
        """Invalidate caches related to a video"""
        # Get all cache keys for the video
        cache_keys = await self.cache_manager.get_keys(f"*{video_id}*")
        
        # Delete all related caches
        for key in cache_keys:
            await self.cache_manager.delete(key)

from typing import Dict, Any, Optional
import numpy as np

class SceneProcessorIntegration:
    def __init__(
        self,
        scene_processor: 'SceneProcessor',
        rag_service: 'RAGService'
    ):
        self.scene_processor = scene_processor
        self.rag_service = rag_service
        
    async def process_scene_for_rag(
        self,
        scene: 'Scene',
        frames: np.ndarray
    ) -> Dict[str, Any]:
        """Process scene for RAG indexing"""
        
        # Get scene classification
        scene_type = await self.scene_processor.classify_scene(frames)
        
        # Generate scene embedding
        scene_embedding = await self.rag_service.generate_scene_embedding(
            frames, scene_type
        )
        
        # Extract temporal context
        prev_context = await self.scene_processor.get_previous_context(scene.id)
        next_context = await self.scene_processor.get_next_context(scene.id)
        
        # Create scene metadata
        scene_metadata = {
            'scene_type': scene_type,
            'embedding': scene_embedding,
            'context': await self.scene_processor.extract_context(frames),
            'temporal_context': {
                'previous': prev_context,
                'next': next_context
            }
        }
        
        # Add scene-specific features
        scene_metadata.update(await self._extract_scene_features(scene, frames))
        
        return scene_metadata
        
    async def _extract_scene_features(
        self,
        scene: 'Scene',
        frames: np.ndarray
    ) -> Dict[str, Any]:
        """Extract additional scene-specific features"""
        features = {}
        
        # Extract motion features
        motion_score = await self.scene_processor.analyze_motion(frames)
        features['motion_intensity'] = motion_score
        
        # Extract visual complexity
        complexity = await self.scene_processor.analyze_complexity(frames)
        features['visual_complexity'] = complexity
        
        # Extract dominant colors
        colors = await self.scene_processor.extract_dominant_colors(frames)
        features['dominant_colors'] = colors
        
        # Extract camera movement
        camera_movement = await self.scene_processor.detect_camera_movement(frames)
        features['camera_movement'] = camera_movement
        
        return features

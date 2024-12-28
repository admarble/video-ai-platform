from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import ollama
from .models import SearchQuery, SearchResult, VideoScene, SceneContext, TimeSegment

class VideoRAGSystem:
    def __init__(self, vector_store_path: str):
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=vector_store_path
        ))
        self.collection = self.client.get_or_create_collection(
            name="video_scenes",
            metadata={"hnsw:space": "cosine"}
        )
        self.llm = ollama.Client()
        
    async def process_query(self, query: SearchQuery) -> SearchResult:
        """Process a search query and return relevant video scenes"""
        # Generate query embedding using Mistral
        query_embedding = await self._generate_query_embedding(query.query_text)
        
        # Search vector store
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=query.max_segments * 2  # Get extra results for filtering
        )
        
        # Filter and rank results
        relevant_scenes = await self._filter_results(results, query)
        
        # Get temporal context if requested
        if query.temporal_context:
            scenes = await self._add_temporal_context(relevant_scenes)
        else:
            scenes = relevant_scenes
            
        # Generate context summary
        summary = await self._generate_summary(scenes, query)
        
        return SearchResult(
            scenes=scenes,
            relevance_scores=self._calculate_relevance(scenes),
            context_summary=summary
        )
    
    async def index_scenes(self, scenes: List[VideoScene]):
        """Index video scenes in the vector store"""
        embeddings = []
        metadatas = []
        ids = []
        
        for scene in scenes:
            # Generate scene embedding
            scene_embedding = await self._generate_scene_embedding(scene)
            embeddings.append(scene_embedding)
            
            # Prepare metadata
            metadatas.append({
                "scene_type": scene.context.scene_type,
                "participants": ",".join(scene.context.participants),
                "action_description": scene.context.action_description,
                "dialogue_summary": scene.context.dialogue_summary,
                "emotions": ",".join(scene.context.emotions),
                "location": scene.context.location,
                "start_time": str(scene.segment.start_time),
                "end_time": str(scene.segment.end_time)
            })
            
            ids.append(scene.embedding_id)
        
        # Add to vector store
        self.collection.add(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    async def _generate_query_embedding(self, query_text: str) -> List[float]:
        """Generate embedding for search query using Mistral"""
        response = await self.llm.embeddings(model="mistral", prompt=query_text)
        return response['embedding']
    
    async def _generate_scene_embedding(self, scene: VideoScene) -> List[float]:
        """Generate embedding for a video scene"""
        # Combine scene information for embedding
        scene_text = f"""
        Type: {scene.context.scene_type}
        Action: {scene.context.action_description}
        Dialogue: {scene.context.dialogue_summary}
        Participants: {', '.join(scene.context.participants)}
        Emotions: {', '.join(scene.context.emotions)}
        Location: {scene.context.location or 'unknown'}
        Transcript: {scene.transcript}
        """
        
        response = await self.llm.embeddings(model="mistral", prompt=scene_text)
        return response['embedding']
    
    async def _filter_results(
        self, 
        results: Dict[str, Any], 
        query: SearchQuery
    ) -> List[VideoScene]:
        """Filter and rank search results"""
        filtered_scenes = []
        
        # Use Mistral to evaluate relevance of each scene
        for idx, metadata in enumerate(results['metadatas']):
            scene = self._metadata_to_scene(metadata, results['ids'][idx])
            
            # Check scene type if specified
            if query.scene_type and scene.context.scene_type != query.scene_type:
                continue
            
            # Evaluate semantic relevance
            relevance = await self._evaluate_scene_relevance(scene, query)
            if relevance > 0.7:  # Relevance threshold
                filtered_scenes.append(scene)
            
            if len(filtered_scenes) >= query.max_segments:
                break
                
        return filtered_scenes
    
    async def _evaluate_scene_relevance(
        self,
        scene: VideoScene,
        query: SearchQuery
    ) -> float:
        """Evaluate scene relevance using Mistral"""
        prompt = f"""
        Query: {query.query_text}
        Scene description: {scene.context.action_description}
        Dialogue: {scene.transcript}
        
        Evaluate the relevance of this scene to the query on a scale of 0 to 1.
        Consider:
        1. Action match
        2. Dialogue relevance
        3. Participant involvement
        4. Emotional context
        
        Return only the numeric score.
        """
        
        response = await self.llm.generate(
            model="mistral",
            prompt=prompt,
            stream=False
        )
        
        try:
            return float(response['response'].strip())
        except ValueError:
            return 0.0
    
    async def _add_temporal_context(self, scenes: List[VideoScene]) -> List[VideoScene]:
        """Add temporal context to search results"""
        all_scenes = []
        for scene in scenes:
            if scene.previous_scene_id:
                prev_scene = await self._get_scene_by_id(scene.previous_scene_id)
                if prev_scene:
                    all_scenes.append(prev_scene)
            
            all_scenes.append(scene)
            
            if scene.next_scene_id:
                next_scene = await self._get_scene_by_id(scene.next_scene_id)
                if next_scene:
                    all_scenes.append(next_scene)
                
        return all_scenes
    
    async def _get_scene_by_id(self, scene_id: str) -> Optional[VideoScene]:
        """Retrieve a scene by its ID"""
        result = self.collection.get(ids=[scene_id])
        if result['ids']:
            return self._metadata_to_scene(result['metadatas'][0], scene_id)
        return None
    
    def _metadata_to_scene(self, metadata: Dict[str, Any], scene_id: str) -> VideoScene:
        """Convert vector store metadata to VideoScene object"""
        return VideoScene(
            embedding_id=scene_id,
            segment=TimeSegment(
                start_time=float(metadata['start_time']),
                end_time=float(metadata['end_time']),
                duration=float(metadata['end_time']) - float(metadata['start_time'])
            ),
            context=SceneContext(
                scene_type=metadata['scene_type'],
                participants=metadata['participants'].split(','),
                action_description=metadata['action_description'],
                dialogue_summary=metadata['dialogue_summary'],
                emotions=metadata['emotions'].split(','),
                location=metadata.get('location')
            ),
            transcript="",  # Transcript would be loaded separately when needed
            previous_scene_id=None,  # Would be populated when temporal context is needed
            next_scene_id=None
        )
    
    async def _generate_summary(
        self,
        scenes: List[VideoScene],
        query: SearchQuery
    ) -> str:
        """Generate a summary of the search results using Mistral"""
        scenes_text = "\n".join([
            f"Scene {i+1}:\n{scene.context.action_description}\n{scene.context.dialogue_summary}"
            for i, scene in enumerate(scenes)
        ])
        
        prompt = f"""
        Query: {query.query_text}
        
        Scenes:
        {scenes_text}
        
        Generate a brief summary explaining how these scenes answer the query.
        Focus on the cause-and-effect relationships if temporal context is requested.
        """
        
        response = await self.llm.generate(
            model="mistral",
            prompt=prompt,
            stream=False
        )
        
        return response['response'].strip()
    
    def _calculate_relevance(self, scenes: List[VideoScene]) -> Dict[str, float]:
        """Calculate relevance scores for scenes"""
        scores = {}
        for scene in scenes:
            # Calculate score based on position in results
            # First results are most relevant
            scores[scene.embedding_id] = 1.0 - (len(scores) * 0.1)
        return scores

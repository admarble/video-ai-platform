# Video RAG System API Reference

## Core API

### VideoRAGSystem

The main class for video search and retrieval.

#### Constructor

```python
VideoRAGSystem(vector_store_path: str)
```

**Parameters:**
- `vector_store_path`: Path to store vector embeddings

#### Methods

##### process_query
```python
async def process_query(query: SearchQuery) -> SearchResult
```
Process a search query and return relevant video scenes.

**Parameters:**
- `query`: SearchQuery object containing search parameters

**Returns:**
- `SearchResult`: Object containing matched scenes and relevance scores

##### index_scenes
```python
async def index_scenes(scenes: List[VideoScene])
```
Index video scenes in the vector store.

**Parameters:**
- `scenes`: List of VideoScene objects to index

## Models

### SearchQuery
```python
class SearchQuery(BaseModel):
    query_text: str
    scene_type: Optional[str]
    temporal_context: bool = False
    max_segments: int = 3
```

### VideoScene
```python
class VideoScene(BaseModel):
    segment: TimeSegment
    context: SceneContext
    transcript: str
    embedding_id: str
    previous_scene_id: Optional[str]
    next_scene_id: Optional[str]
```

### SearchResult
```python
class SearchResult(BaseModel):
    scenes: List[VideoScene]
    relevance_scores: Dict[str, float]
    context_summary: str
```

## Integration APIs

### VideoProcessorIntegration

#### Constructor
```python
VideoProcessorIntegration(
    video_processor: VideoProcessor,
    rag_service: RAGService,
    cache_manager: CacheManager
)
```

#### Methods

##### process_video_with_rag
```python
async def process_video_with_rag(
    video_path: str,
    extract_scenes: bool = True
) -> Dict[str, Any]
```

**Parameters:**
- `video_path`: Path to video file
- `extract_scenes`: Whether to extract scenes

**Returns:**
- Dictionary containing processed scene data

### SceneProcessorIntegration

#### Constructor
```python
SceneProcessorIntegration(
    scene_processor: SceneProcessor,
    rag_service: RAGService
)
```

#### Methods

##### process_scene_for_rag
```python
async def process_scene_for_rag(
    scene: Scene,
    frames: np.ndarray
) -> Dict[str, Any]
```

**Parameters:**
- `scene`: Scene object
- `frames`: Video frames as numpy array

**Returns:**
- Dictionary containing scene metadata

### ClipGenerator

#### Constructor
```python
ClipGenerator(
    temp_dir: str,
    max_concurrent: int = 5
)
```

#### Methods

##### create_clips
```python
async def create_clips(
    scenes: List[VideoScene],
    video_id: str
) -> List[Dict[str, Any]]
```

**Parameters:**
- `scenes`: List of scenes to generate clips from
- `video_id`: Video identifier

**Returns:**
- List of clip metadata dictionaries

##### generate_clip
```python
async def generate_clip(
    video_id: str,
    start_time: float,
    end_time: float,
    scene_id: str,
    format: str = 'mp4'
) -> Dict[str, Any]
```

**Parameters:**
- `video_id`: Video identifier
- `start_time`: Clip start time in seconds
- `end_time`: Clip end time in seconds
- `scene_id`: Scene identifier
- `format`: Output video format

**Returns:**
- Dictionary containing clip metadata

## REST API Endpoints

### Search

#### POST /api/v1/search
Search for video scenes.

**Request:**
```json
{
    "query": "Find the team discussion",
    "video_id": "video123",
    "temporal_context": true,
    "max_segments": 3
}
```

**Response:**
```json
{
    "status": "success",
    "clips": [
        {
            "url": "/clips/video123_scene456.mp4",
            "duration": 15.5,
            "timeframe": {
                "start": 120.0,
                "end": 135.5
            }
        }
    ],
    "context": "Team discussion about project timeline",
    "metadata": {
        "scene_count": 1
    }
}
```

### Clip Generation

#### POST /api/v1/clips
Generate a video clip.

**Request:**
```json
{
    "scene_id": "scene123",
    "video_id": "video456",
    "start_time": 25.0,
    "end_time": 35.0
}
```

**Response:**
```json
{
    "status": "success",
    "clip_url": "/clips/video456_scene123.mp4",
    "duration": 10.0,
    "format": "mp4"
}
```

## Event System

### RAGEventHandler

#### Constructor
```python
RAGEventHandler(
    rag_service: RAGService,
    cache_manager: CacheManager
)
```

#### Methods

##### handle_video_processed
```python
async def handle_video_processed(event: Dict[str, Any])
```

**Parameters:**
- `event`: Event data containing video processing results

##### handle_scene_updated
```python
async def handle_scene_updated(event: Dict[str, Any])
```

**Parameters:**
- `event`: Event data containing scene updates

## Error Handling

### Common Exceptions

```python
class VideoNotFoundError(Exception):
    pass

class SceneProcessingError(Exception):
    pass

class ClipGenerationError(Exception):
    pass

class RAGSearchError(Exception):
    pass
```

### Error Responses

```json
{
    "status": "error",
    "detail": "Error message",
    "error_code": "VIDEO_NOT_FOUND"
}
```

## Configuration

### Vector Store Configuration
```yaml
vector_store:
  type: "chromadb"
  path: "./data/vectors"
  settings:
    chroma_db_impl: "duckdb+parquet"
```

### LLM Configuration
```yaml
llm:
  type: "mistral"
  model: "mistral-7b"
  endpoint: "http://localhost:11434"
```

### Clip Generator Configuration
```yaml
clip_generator:
  temp_dir: "./temp/clips"
  max_concurrent: 5
  settings:
    format: "mp4"
    video_codec: "h264"
``` 
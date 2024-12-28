# Video RAG System Usage Guide

## Overview

The Video RAG (Retrieval-Augmented Generation) system enables semantic search and clip generation from video content. It combines scene detection, natural language understanding, and temporal context to provide relevant video segments based on user queries.

## Quick Start

```python
from rag.core import VideoRAGSystem
from rag.models import SearchQuery

# Initialize the system
rag_system = VideoRAGSystem(vector_store_path="./data/vectors")

# Create a search query
query = SearchQuery(
    query_text="Show me the fight scene between John and Mike",
    temporal_context=True,
    max_segments=3
)

# Process the query
results = await rag_system.process_query(query)

# Access the results
for scene in results.scenes:
    print(f"Scene: {scene.context.action_description}")
    print(f"Time: {scene.segment.start_time} - {scene.segment.end_time}")
```

## Core Components

### 1. Video Processing

Process videos and extract scenes:

```python
from rag.integrations.video_processor import VideoProcessorIntegration

# Initialize components
video_processor = VideoProcessorIntegration(
    video_processor=video_processor,
    rag_service=rag_service,
    cache_manager=cache_manager
)

# Process a video
scene_data = await video_processor.process_video_with_rag(
    video_path="path/to/video.mp4",
    extract_scenes=True
)
```

### 2. Scene Processing

Process individual scenes with detailed analysis:

```python
from rag.integrations.scene_processor import SceneProcessorIntegration

# Initialize scene processor
scene_processor = SceneProcessorIntegration(
    scene_processor=scene_processor,
    rag_service=rag_service
)

# Process a scene
scene_metadata = await scene_processor.process_scene_for_rag(
    scene=scene,
    frames=video_frames
)
```

### 3. Clip Generation

Generate video clips from scenes:

```python
from rag.processors.clip_generator import ClipGenerator

# Initialize clip generator
clip_generator = ClipGenerator(
    temp_dir="./temp/clips",
    max_concurrent=5
)

# Generate clips for scenes
clips = await clip_generator.create_clips(
    scenes=results.scenes,
    video_id="video123"
)

# Generate single clip
clip = await clip_generator.generate_clip(
    video_id="video123",
    start_time=10.5,
    end_time=15.2,
    scene_id="scene456"
)
```

## API Integration

### 1. Search Endpoint

```python
from rag.integrations.api import VideoSearchAPI

# Initialize API
api = VideoSearchAPI(
    rag_service=rag_service,
    clip_generator=clip_generator,
    cache_manager=cache_manager
)

# Search video content
results = await api.search_video(
    query="Find the conversation about project deadlines",
    video_id="video123",
    temporal_context=True
)
```

### 2. Clip Generation Endpoint

```python
# Generate clip for specific scene
clip = await api.generate_clip(
    scene_id="scene123",
    video_id="video456",
    start_time=25.0,
    end_time=35.0
)
```

## Example Queries

The system supports various types of natural language queries:

1. Action-based queries:
```python
query = SearchQuery(
    query_text="Show me the scene where John enters the office",
    scene_type="action"
)
```

2. Dialogue-based queries:
```python
query = SearchQuery(
    query_text="Find the discussion about the new project timeline",
    scene_type="conversation"
)
```

3. Temporal queries:
```python
query = SearchQuery(
    query_text="What happened after the team meeting?",
    temporal_context=True
)
```

4. Emotional context queries:
```python
query = SearchQuery(
    query_text="Show me the tense argument between team members",
    scene_type="conflict"
)
```

## Configuration

The system is configured through `rag_config.yaml`:

```yaml
vector_store:
  type: "chromadb"
  path: "./data/vectors"

llm:
  type: "mistral"
  model: "mistral-7b"
  endpoint: "http://localhost:11434"

clip_generator:
  temp_dir: "./temp/clips"
  max_concurrent: 5
```

## Best Practices

1. **Video Processing**:
   - Pre-process videos to extract scenes and generate embeddings
   - Use caching for frequently accessed scenes
   - Clean up temporary clips regularly

2. **Query Optimization**:
   - Be specific in queries for better results
   - Use temporal context when sequence matters
   - Specify scene type for focused searches

3. **Resource Management**:
   - Monitor clip storage usage
   - Set appropriate concurrent processing limits
   - Implement cleanup strategies for temporary files

4. **Error Handling**:
   - Handle missing videos gracefully
   - Implement retry logic for LLM calls
   - Validate time ranges for clip generation

## Common Use Cases

1. **Meeting Summarization**:
```python
query = SearchQuery(
    query_text="Find all discussions about budget planning",
    max_segments=5
)
```

2. **Event Detection**:
```python
query = SearchQuery(
    query_text="Show me when the team celebrated project completion",
    scene_type="celebration"
)
```

3. **Sequential Analysis**:
```python
query = SearchQuery(
    query_text="Show the sequence of events leading to the client presentation",
    temporal_context=True,
    max_segments=4
)
```

## Monitoring and Maintenance

1. **Health Checks**:
```python
# Check system health
await service_manager.initialize_rag_services()
```

2. **Cache Management**:
```python
# Clear cache for video
await cache_manager.delete(f"video_{video_id}_*")
```

3. **Cleanup**:
```python
# Clean up temporary clips
await clip_generator.cleanup_all()
```

## Troubleshooting

Common issues and solutions:

1. **Slow Search Results**:
   - Check vector store indexing
   - Optimize cache settings
   - Adjust concurrent processing limits

2. **Poor Search Relevance**:
   - Refine query phrasing
   - Adjust relevance thresholds
   - Check scene classification accuracy

3. **Clip Generation Issues**:
   - Verify FFmpeg installation
   - Check storage permissions
   - Monitor temporary directory space

## Advanced Features

1. **Custom Scene Processing**:
```python
# Add custom scene features
features = await scene_processor._extract_scene_features(
    scene=scene,
    frames=frames
)
```

2. **Event Handling**:
```python
# Handle video updates
await event_handler.handle_video_processed({
    'video_id': 'video123',
    'scene_data': scene_data
})
```

3. **Custom Embeddings**:
```python
# Generate custom embeddings
embedding = await rag_system._generate_scene_embedding(scene)
``` 
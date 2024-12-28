# Video RAG System Quick Start Guide

## Prerequisites

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Install Ollama and download Mistral model:
```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull Mistral model
ollama pull mistral
```

3. Install FFmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt-get install ffmpeg
```

## Basic Setup

1. Create necessary directories:
```bash
mkdir -p data/vectors temp/clips logs
```

2. Copy configuration template:
```bash
cp config/rag_config.yaml.template config/rag_config.yaml
```

3. Update configuration:
```yaml
vector_store:
  path: "./data/vectors"

llm:
  endpoint: "http://localhost:11434"

clip_generator:
  temp_dir: "./temp/clips"
```

## Simple Usage Example

```python
import asyncio
from rag.core import VideoRAGSystem
from rag.models import SearchQuery
from rag.processors.clip_generator import ClipGenerator

async def main():
    # Initialize RAG system
    rag = VideoRAGSystem(vector_store_path="./data/vectors")
    
    # Initialize clip generator
    clip_gen = ClipGenerator(temp_dir="./temp/clips")
    
    # Create a search query
    query = SearchQuery(
        query_text="Show me the team discussion about the new feature",
        temporal_context=True
    )
    
    # Search for relevant scenes
    results = await rag.process_query(query)
    
    # Generate clips
    clips = await clip_gen.create_clips(
        scenes=results.scenes,
        video_id="video123"
    )
    
    # Print results
    print("\nSearch Results:")
    print(f"Found {len(results.scenes)} relevant scenes")
    print(f"\nContext Summary:")
    print(results.context_summary)
    
    print("\nGenerated Clips:")
    for clip in clips:
        print(f"- {clip['url']} ({clip['duration']}s)")

if __name__ == "__main__":
    asyncio.run(main())
```

## Processing a Video

```python
from rag.integrations.video_processor import VideoProcessorIntegration

async def process_video():
    # Initialize components
    processor = VideoProcessorIntegration(
        video_processor=video_processor,
        rag_service=rag_service,
        cache_manager=cache_manager
    )
    
    # Process video
    scene_data = await processor.process_video_with_rag(
        video_path="path/to/video.mp4"
    )
    
    print(f"Processed {len(scene_data)} scenes")

asyncio.run(process_video())
```

## API Usage

```python
from fastapi import FastAPI
from rag.integrations.api import VideoSearchAPI

app = FastAPI()
api = VideoSearchAPI(
    rag_service=rag_service,
    clip_generator=clip_generator,
    cache_manager=cache_manager
)

@app.post("/search")
async def search_video(
    query: str,
    video_id: str,
    temporal_context: bool = False
):
    return await api.search_video(
        query=query,
        video_id=video_id,
        temporal_context=temporal_context
    )

@app.post("/generate-clip")
async def generate_clip(
    scene_id: str,
    video_id: str,
    start_time: float,
    end_time: float
):
    return await api.generate_clip(
        scene_id=scene_id,
        video_id=video_id,
        start_time=start_time,
        end_time=end_time
    )
```

## Common Operations

### 1. Search with Temporal Context

```python
# Find what led to an event
query = SearchQuery(
    query_text="What led to the team disagreement?",
    temporal_context=True,
    max_segments=5
)
```

### 2. Generate Highlight Clips

```python
# Generate clips of key moments
clips = await clip_generator.create_clips(
    scenes=results.scenes,
    video_id="meeting_123"
)
```

### 3. Cache Management

```python
# Clear search cache
await cache_manager.delete("search_*")

# Clear video processing cache
await cache_manager.delete(f"video_{video_id}_*")
```

## Next Steps

1. Read the full [Usage Guide](USAGE.md) for detailed features
2. Check [Configuration Guide](CONFIGURATION.md) for advanced settings
3. See [API Reference](API.md) for complete API documentation

## Troubleshooting

1. If search is slow:
```python
# Optimize vector store
await rag_system.collection.optimize()
```

2. If clips aren't generating:
```python
# Check FFmpeg installation
import ffmpeg
ffmpeg.probe("test.mp4")
```

3. If LLM isn't responding:
```bash
# Check Ollama service
curl http://localhost:11434/api/tags
``` 
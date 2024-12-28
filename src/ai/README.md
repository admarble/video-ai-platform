# AI Video Processing Components

This module provides a comprehensive video processing pipeline that integrates three powerful AI models:
- TimeSformer for temporal analysis
- YOLOv8 for object detection and tracking
- Whisper for audio transcription

## Architecture

The implementation consists of three main processors and an integrated processor that combines them:

```
src/ai/
├── processors/
│   ├── timesformer_processor.py  # Temporal analysis
│   ├── yolo_processor.py         # Object detection
│   ├── whisper_processor.py      # Audio transcription
│   └── integrated_processor.py   # Combined processing
```

### Components

1. **TimeSformer Processor**
   - Frame-level temporal analysis
   - Feature extraction
   - Attention pattern analysis

2. **YOLOv8 Processor**
   - Object detection and tracking
   - Spatial relationship analysis
   - Multi-object tracking

3. **Whisper Processor**
   - Audio transcription
   - Word-level timestamp alignment
   - Frame-to-word mapping

4. **Integrated Processor**
   - Parallel processing coordination
   - Result aggregation
   - Error handling and recovery

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have ffmpeg installed for audio processing:
```bash
# On macOS
brew install ffmpeg

# On Ubuntu
sudo apt-get install ffmpeg
```

## Usage

### Basic Usage

```python
from src.ai.processors.integrated_processor import (
    IntegratedProcessor,
    IntegratedSettings,
    TimesformerSettings,
    YOLOSettings,
    WhisperSettings
)

# Initialize settings
settings = IntegratedSettings(
    timesformer=TimesformerSettings(
        num_frames=8,
        image_size=224
    ),
    yolo=YOLOSettings(
        model_size='x',
        conf_threshold=0.25
    ),
    whisper=WhisperSettings(
        model_size='base',
        compute_type='float16'
    )
)

# Create processor
with IntegratedProcessor(settings) as processor:
    # Process video
    results = processor.process_video(frames, audio_path, fps)
```

### Command Line Usage

Use the example script to process a video file:

```bash
python examples/integrated_example.py video.mp4 --output-dir output --device cuda
```

## Output Format

The processor returns a dictionary with the following structure:

```python
{
    'temporal_analysis': {
        'frame_features': [...],      # Frame-level features
        'temporal_patterns': [...],   # Temporal attention patterns
        'attention_maps': [...]       # Attention visualization data
    },
    'object_detection': [
        {
            'track_id': int,          # Unique object ID
            'class_id': int,          # Object class ID
            'label': str,             # Object class label
            'confidence': float,       # Detection confidence
            'bbox': [x1, y1, x2, y2], # Bounding box coordinates
            'frame_idx': int          # Frame index
        }
    ],
    'spatial_relationships': {
        frame_idx: [
            {
                'object1': {...},      # First object info
                'object2': {...},      # Second object info
                'relationship': str,   # Relationship type
                'distance': float,     # Distance between objects
                'relative_position': {
                    'dx': float,      # X-axis relative position
                    'dy': float       # Y-axis relative position
                }
            }
        ]
    },
    'audio_transcription': {
        'segments': [...],            # Transcription segments
        'words': [...],               # Word-level transcription
        'language': str              # Detected language
    },
    'frame_alignments': {
        'aligned_words': [...],       # Words with frame indices
        'frame_word_map': {...}       # Frame to word mapping
    },
    'metadata': {
        'success_status': {          # Processing status for each component
            'temporal': bool,
            'objects': bool,
            'spatial': bool,
            'audio': bool,
            'alignments': bool
        }
    }
}
```

## Error Handling

The implementation includes comprehensive error handling:

1. **Memory Management**
   - Automatic GPU memory cleanup
   - Batch size reduction on OOM errors
   - Graceful degradation

2. **Parallel Processing**
   - Independent component processing
   - Partial result handling
   - Thread pool management

3. **Recovery Mechanisms**
   - Automatic retry with reduced settings
   - Partial result preservation
   - Detailed error reporting

## Performance Optimization

The implementation includes several optimizations:

1. **Batch Processing**
   - Configurable batch sizes
   - Memory-aware processing
   - Parallel execution

2. **Resource Management**
   - GPU memory monitoring
   - Thread pool execution
   - Automatic cleanup

3. **Efficiency Features**
   - Overlapping temporal windows
   - Cached model loading
   - Result aggregation

## Contributing

When contributing to this project:

1. Follow the existing code style
2. Add tests for new functionality
3. Update documentation
4. Handle errors gracefully
5. Optimize for both CPU and GPU execution 
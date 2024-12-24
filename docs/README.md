# Cuthrough - Adaptive Video Compression Tool

Cuthrough is an advanced video compression tool that uses adaptive optimization to achieve the best balance between quality, speed, and file size reduction. It features intelligent profile selection, real-time optimization, and comprehensive metrics tracking.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cuthrough.git
cd cuthrough
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make the CLI tool executable:
```bash
chmod +x src/cuthrough.py
# Optional: Create a symlink for system-wide access
ln -s $(pwd)/src/cuthrough.py /usr/local/bin/cuthrough
```

## Command Line Interface

Cuthrough provides a powerful CLI with three main commands: `compress`, `analyze`, and `profile`.

### Compression

The `compress` command is used to compress video files with various optimization options.

```bash
# Basic compression with default settings
cuthrough compress input.mp4 output.mp4

# Compression with specific quality targets
cuthrough compress input.mp4 output.mp4 \
    --quality 0.9 \
    --speed 0.5 \
    --size-reduction 0.6

# Using a predefined profile
cuthrough compress input.mp4 output.mp4 \
    --profile high_quality.json

# Save compression metrics
cuthrough compress input.mp4 output.mp4 \
    --save-metrics metrics.json
```

#### Compression Options

- `--quality FLOAT`: Target quality (0-1), default: 0.8
- `--speed FLOAT`: Target speed (0-1), default: 0.6
- `--size-reduction FLOAT`: Target size reduction (0-1), default: 0.7
- `--profile PATH`: Path to compression profile JSON
- `--optimization-strategy {SIMPLE,GRADIENT,ADAPTIVE,WEIGHTED}`: Optimization strategy
- `--learning-rate FLOAT`: Learning rate for optimization
- `--history-weight FLOAT`: Weight decay for historical performance
- `--save-metrics PATH`: Save compression metrics to JSON file

### Video Analysis

The `analyze` command provides detailed information about video files.

```bash
# Basic analysis
cuthrough analyze input.mp4

# JSON output
cuthrough analyze input.mp4 --json
```

Analysis output includes:
- Resolution
- Duration
- Frame rate
- Bitrate
- File size
- Container format
- Audio presence

### Profile Management

The `profile` command helps manage compression profiles.

```bash
# Create a new profile
cuthrough profile create high_quality.json \
    --name "High Quality" \
    --video-codec libx264 \
    --preset veryslow \
    --quality-value 18

# View profile settings
cuthrough profile show high_quality.json
cuthrough profile show high_quality.json --json
```

#### Profile Options

- `--name NAME`: Profile name
- `--video-codec CODEC`: Video codec (e.g., libx264, libx265, libvpx-vp9)
- `--audio-codec CODEC`: Audio codec (e.g., aac, opus)
- `--container FORMAT`: Container format (e.g., mp4, mkv, webm)
- `--preset PRESET`: Encoder preset (e.g., ultrafast, medium, veryslow)
- `--video-bitrate RATE`: Video bitrate (e.g., 2M, 5M)
- `--audio-bitrate RATE`: Audio bitrate (e.g., 128k, 192k)
- `--quality-value INT`: Quality value (e.g., CRF value)
- `--multipass`: Enable multipass encoding

### Common Options

These options are available for all commands:

- `--debug`: Enable debug logging
- `--log-file PATH`: Log to file
- `--quiet`: Suppress progress output

## Compression Features

### Adaptive Optimization

Cuthrough uses an adaptive optimization system that automatically tunes compression parameters based on:

1. **Video Characteristics**:
   - Resolution
   - Frame rate
   - Motion content
   - Duration

2. **Target Metrics**:
   - Quality (measured using VMAF/SSIM)
   - Speed (compression time relative to video duration)
   - Size reduction (output size relative to input)

3. **Optimization Strategies**:
   - SIMPLE: Basic parameter adjustment
   - GRADIENT: Gradient-based optimization
   - ADAPTIVE: Dynamic parameter tuning
   - WEIGHTED: Weighted metric optimization

### Compression Profiles

Profiles define compression settings and can be:

1. **Predefined**:
   - High Quality (optimized for quality)
   - Balanced (good balance of quality and size)
   - Fast (optimized for speed)

2. **Custom**:
   - User-defined settings
   - JSON format
   - Portable between sessions

Example profile JSON:
```json
{
  "name": "high_quality",
  "video_codec": "libx264",
  "audio_codec": "aac",
  "container_format": "mp4",
  "video_bitrate": "5M",
  "audio_bitrate": "192k",
  "preset": "veryslow",
  "quality_value": 18,
  "multipass": true
}
```

### Performance Metrics

Cuthrough tracks various metrics during compression:

1. **Quality Metrics**:
   - VMAF (Video Multimethod Assessment Fusion)
   - SSIM (Structural Similarity Index)
   - Custom quality scores

2. **Performance Metrics**:
   - Compression speed
   - Size reduction ratio
   - Memory usage

3. **Output Metrics**:
   - Resolution
   - Bitrate
   - File size
   - Duration

Metrics can be saved to JSON for analysis:
```json
{
  "input": "input.mp4",
  "output": "output.mp4",
  "profile": {
    "name": "high_quality",
    "video_codec": "libx264",
    ...
  },
  "metrics": {
    "quality": 0.95,
    "speed": 0.75,
    "size_reduction": 0.65
  },
  "video_info": {
    "input": {
      "width": 1920,
      "height": 1080,
      "duration": 300.0,
      "size": 1073741824
    },
    "output": {
      "width": 1920,
      "height": 1080,
      "duration": 300.0,
      "size": 375809638
    }
  }
}
```

## Examples

### Basic Compression

```bash
# Quick compression with default settings
cuthrough compress input.mp4 output.mp4

# High-quality compression
cuthrough compress input.mp4 output.mp4 \
    --quality 0.95 \
    --optimization-strategy ADAPTIVE
```

### Advanced Usage

```bash
# Create a custom profile
cuthrough profile create custom.json \
    --name "Custom 4K" \
    --video-codec libx265 \
    --preset slow \
    --video-bitrate 8M \
    --multipass

# Analyze input video
cuthrough analyze input.mp4 --json > analysis.json

# Compress with custom profile and save metrics
cuthrough compress input.mp4 output.mp4 \
    --profile custom.json \
    --save-metrics metrics.json \
    --debug \
    --log-file compression.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
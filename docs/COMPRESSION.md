# Compression Features and Optimization

## Adaptive Optimization System

Cuthrough uses an advanced adaptive optimization system that automatically tunes compression parameters to achieve the best balance between quality, speed, and file size.

### Optimization Strategies

1. **SIMPLE**
   - Basic parameter adjustment based on target metrics
   - Good for quick compressions
   - Minimal computational overhead
   - Example:
     ```bash
     cuthrough compress input.mp4 output.mp4 \
         --optimization-strategy SIMPLE \
         --quality 0.8
     ```

2. **GRADIENT**
   - Gradient-based parameter optimization
   - Better for finding optimal settings
   - Uses historical performance data
   - Example:
     ```bash
     cuthrough compress input.mp4 output.mp4 \
         --optimization-strategy GRADIENT \
         --learning-rate 0.1 \
         --history-weight 0.9
     ```

3. **ADAPTIVE**
   - Dynamic parameter tuning based on video content
   - Best for varying content
   - Adapts to changes in video characteristics
   - Example:
     ```bash
     cuthrough compress input.mp4 output.mp4 \
         --optimization-strategy ADAPTIVE \
         --quality 0.9 \
         --speed 0.7
     ```

4. **WEIGHTED**
   - Weighted optimization of multiple metrics
   - Good for specific requirements
   - Balances conflicting goals
   - Example:
     ```bash
     cuthrough compress input.mp4 output.mp4 \
         --optimization-strategy WEIGHTED \
         --quality 0.9 \
         --speed 0.5 \
         --size-reduction 0.7
     ```

### Target Metrics

1. **Quality (0-1)**
   - Measured using VMAF or SSIM
   - Higher values prioritize visual quality
   - Affects encoder settings and bitrate
   - Example: `--quality 0.9`

2. **Speed (0-1)**
   - Compression time relative to video duration
   - Higher values use faster presets
   - Trades quality for speed
   - Example: `--speed 0.8`

3. **Size Reduction (0-1)**
   - Output size relative to input
   - Higher values target smaller files
   - Affects bitrate and compression level
   - Example: `--size-reduction 0.7`

## Compression Profiles

### Predefined Profiles

1. **High Quality**
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

2. **Balanced**
```json
{
  "name": "balanced",
  "video_codec": "libx264",
  "audio_codec": "aac",
  "container_format": "mp4",
  "video_bitrate": "2M",
  "audio_bitrate": "128k",
  "preset": "medium",
  "quality_value": 23,
  "multipass": false
}
```

3. **Fast**
```json
{
  "name": "fast",
  "video_codec": "libx264",
  "audio_codec": "aac",
  "container_format": "mp4",
  "video_bitrate": "1M",
  "audio_bitrate": "128k",
  "preset": "ultrafast",
  "quality_value": 28,
  "multipass": false
}
```

### Custom Profiles

Create custom profiles for specific needs:

```bash
# 4K High Quality Profile
cuthrough profile create 4k_quality.json \
    --name "4K Quality" \
    --video-codec libx265 \
    --preset slow \
    --video-bitrate 8M \
    --quality-value 20 \
    --multipass

# Web Optimized Profile
cuthrough profile create web.json \
    --name "Web Optimized" \
    --video-codec libvpx-vp9 \
    --container webm \
    --video-bitrate 1M \
    --quality-value 31

# Quick Preview Profile
cuthrough profile create preview.json \
    --name "Preview" \
    --video-codec libx264 \
    --preset ultrafast \
    --video-bitrate 500k \
    --quality-value 35
```

## Quality Assessment

### VMAF (Video Multimethod Assessment Fusion)
- Perceptual video quality assessment
- Machine learning-based
- Netflix's open-source quality metric
- More accurate for modern content

### SSIM (Structural Similarity Index)
- Traditional quality metric
- Measures structural similarity
- Fallback when VMAF is unavailable
- Fast computation

### Custom Quality Scores
- Normalized between 0 and 1
- Combines multiple metrics
- Weighted based on content type
- Adapts to user preferences

## Performance Monitoring

### Real-time Metrics
```bash
cuthrough compress input.mp4 output.mp4 \
    --save-metrics metrics.json
```

Example metrics output:
```json
{
  "metrics": {
    "quality": 0.95,
    "speed": 0.75,
    "size_reduction": 0.65
  },
  "performance": {
    "compression_time": 120.5,
    "cpu_usage": 85.2,
    "memory_usage": 512.0
  }
}
```

### Progress Tracking
- Real-time progress display
- ETA calculation
- Frame processing rate
- Size estimation

### Logging
```bash
cuthrough compress input.mp4 output.mp4 \
    --debug \
    --log-file compression.log
```

Log output includes:
- Parameter adjustments
- Quality measurements
- Performance metrics
- Error handling

## Best Practices

1. **Quality-Focused Compression**
```bash
cuthrough compress input.mp4 output.mp4 \
    --quality 0.95 \
    --optimization-strategy ADAPTIVE \
    --multipass
```

2. **Fast Compression**
```bash
cuthrough compress input.mp4 output.mp4 \
    --speed 0.9 \
    --optimization-strategy SIMPLE
```

3. **Size-Focused Compression**
```bash
cuthrough compress input.mp4 output.mp4 \
    --size-reduction 0.8 \
    --optimization-strategy WEIGHTED
```

4. **Balanced Compression**
```bash
cuthrough compress input.mp4 output.mp4 \
    --quality 0.8 \
    --speed 0.7 \
    --size-reduction 0.6 \
    --optimization-strategy ADAPTIVE
``` 
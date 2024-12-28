# Quick Start Guide

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/cuthrough.git
cd cuthrough

# Install dependencies
pip install -r requirements.txt

# Make executable
chmod +x src/cuthrough.py
```

## Basic Usage

1. **Analyze a Video**
```bash
./src/cuthrough.py analyze input.mp4
```

2. **Quick Compression**
```bash
./src/cuthrough.py compress input.mp4 output.mp4
```

3. **High-Quality Compression**
```bash
./src/cuthrough.py compress input.mp4 output.mp4 \
    --quality 0.9 \
    --optimization-strategy ADAPTIVE
```

## Common Profiles

1. **High Quality**
```bash
./src/cuthrough.py profile create high_quality.json \
    --name "High Quality" \
    --video-codec libx264 \
    --preset veryslow \
    --quality-value 18
```

2. **Fast Compression**
```bash
./src/cuthrough.py profile create fast.json \
    --name "Fast" \
    --video-codec libx264 \
    --preset ultrafast \
    --quality-value 28
```

3. **Balanced**
```bash
./src/cuthrough.py profile create balanced.json \
    --name "Balanced" \
    --video-codec libx264 \
    --preset medium \
    --quality-value 23
```

## Using Profiles

```bash
# Compress using a profile
./src/cuthrough.py compress input.mp4 output.mp4 \
    --profile high_quality.json

# Save compression metrics
./src/cuthrough.py compress input.mp4 output.mp4 \
    --profile balanced.json \
    --save-metrics metrics.json
```

## Getting Help

```bash
# General help
./src/cuthrough.py --help

# Command-specific help
./src/cuthrough.py compress --help
./src/cuthrough.py analyze --help
./src/cuthrough.py profile --help
```

For more detailed information, see the full [documentation](README.md). 
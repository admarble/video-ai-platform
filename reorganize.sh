#!/bin/bash

# Create necessary directories if they don't exist
mkdir -p src/core/circuit_breaker
mkdir -p src/core/cache
mkdir -p examples
mkdir -p docs/api
mkdir -p config

# Move circuit breaker related files
mv circuit_breaker.py src/core/circuit_breaker/
mv distributed_circuit_breaker.py src/core/circuit_breaker/

# Move cache related files
mv cache_manager.py src/core/cache/

# Move example files
mv example.py examples/
mv example_distributed.py examples/
mv example_advanced.py examples/

# Move documentation
mv INSTALL.md docs/
mv CHECKLIST.md docs/

# Move configuration files
mv .env.example config/

# Clean up
rm -rf video-ai-platform
rm -rf enhanced_config_manager.egg-info
rm -rf video_ai_platform.egg-info
rm -rf ~/

# Remove build artifacts if they exist
rm -rf build/*
rm -rf *.egg-info

echo "Project restructuring completed!" 
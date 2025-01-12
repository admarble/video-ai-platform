#!/bin/bash

# Remove redundant directories
rm -rf cache/
rm -rf source/
rm -rf build/
rm -rf .pytest_cache/
rm -rf .benchmarks/
rm -rf enhanced_config_manager.egg-info/
rm -rf video_ai_platform.egg-info/

# Move .env to config directory
mv .env config/

# Move documentation-related files
mv mkdocs.yml docs/

# Update .gitignore to include new paths
echo "# Build" >> .gitignore
echo "build/" >> .gitignore
echo "*.egg-info/" >> .gitignore
echo "# Environment" >> .gitignore
echo "config/.env" >> .gitignore

echo "Cleanup completed!" 
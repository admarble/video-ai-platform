#!/bin/bash

# Update imports in example files
find examples -type f -name "*.py" -exec sed -i '' \
    -e 's/from circuit_breaker/from src.core.circuit_breaker/g' \
    -e 's/import circuit_breaker/import src.core.circuit_breaker/g' \
    -e 's/from cache_manager/from src.core.cache/g' \
    -e 's/import cache_manager/import src.core.cache/g' {} +

# Update imports in src files
find src -type f -name "*.py" -exec sed -i '' \
    -e 's/from circuit_breaker/from src.core.circuit_breaker/g' \
    -e 's/import circuit_breaker/import src.core.circuit_breaker/g' \
    -e 's/from cache_manager/from src.core.cache/g' \
    -e 's/import cache_manager/import src.core.cache/g' {} +

# Update imports in tests
find tests -type f -name "*.py" -exec sed -i '' \
    -e 's/from circuit_breaker/from src.core.circuit_breaker/g' \
    -e 's/import circuit_breaker/import src.core.circuit_breaker/g' \
    -e 's/from cache_manager/from src.core.cache/g' \
    -e 's/import cache_manager/import src.core.cache/g' {} +

echo "Import statements updated!" 
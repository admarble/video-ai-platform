import pytest
import time
from unittest.mock import Mock, patch
import redis
from typing import Generator, Any

from rate_limiter.storage.memory_storage import MemoryStorage
from rate_limiter.storage.redis_storage import RedisStorage
from rate_limiter import RateLimiter

@pytest.fixture
def mock_redis() -> Generator[Mock, Any, None]:
    """Mock Redis client for testing"""
    with patch('redis.Redis') as mock:
        yield mock

@pytest.fixture
def mock_time() -> Generator[Mock, Any, None]:
    """Mock time.time() for predictable testing"""
    with patch('time.time') as mock:
        mock.return_value = 1000.0
        yield mock

@pytest.fixture
def memory_storage() -> MemoryStorage:
    """Create memory storage instance"""
    return MemoryStorage()

@pytest.fixture
def redis_storage(mock_redis) -> RedisStorage:
    """Create Redis storage instance with mock client"""
    return RedisStorage(mock_redis)

@pytest.fixture
def rate_limiter(memory_storage) -> RateLimiter:
    """Create rate limiter with memory storage"""
    return RateLimiter(storage=memory_storage) 
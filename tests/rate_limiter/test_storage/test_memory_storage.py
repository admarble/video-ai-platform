import pytest
import time
import threading
import queue
from rate_limiter.storage.memory_storage import MemoryStorage
from rate_limiter.types import RateLimit, RateLimitType

class TestMemoryStorage:
    def test_check_limit_within_limit(self, memory_storage, mock_time):
        """Test check_limit when within limits"""
        rate_limit = RateLimit(limit=5, window=60, type=RateLimitType.REQUESTS)
        
        # Make multiple requests
        for _ in range(3):
            assert memory_storage.check_limit("test_key", rate_limit)
            
        # Verify count
        info = memory_storage.get_limit_info("test_key", rate_limit)
        assert info["remaining"] == 2
    
    def test_check_limit_exceeded(self, memory_storage, mock_time):
        """Test check_limit when limit is exceeded"""
        rate_limit = RateLimit(limit=2, window=60, type=RateLimitType.REQUESTS)
        
        # Make requests up to limit
        assert memory_storage.check_limit("test_key", rate_limit)
        assert memory_storage.check_limit("test_key", rate_limit)
        
        # This should exceed the limit
        assert not memory_storage.check_limit("test_key", rate_limit)
    
    def test_window_expiration(self, memory_storage, mock_time):
        """Test that old requests are expired from the window"""
        rate_limit = RateLimit(limit=2, window=60, type=RateLimitType.REQUESTS)
        
        # Make initial requests
        assert memory_storage.check_limit("test_key", rate_limit)
        assert memory_storage.check_limit("test_key", rate_limit)
        assert not memory_storage.check_limit("test_key", rate_limit)
        
        # Move time forward past window
        mock_time.return_value = 1100.0  # Original time + 100s
        
        # Should be allowed again
        assert memory_storage.check_limit("test_key", rate_limit)

    def test_concurrent_access(self, memory_storage):
        """Test thread safety of memory storage"""
        rate_limit = RateLimit(limit=100, window=60, type=RateLimitType.REQUESTS)
        results = queue.Queue()
        
        def make_requests():
            successes = 0
            for _ in range(50):
                if memory_storage.check_limit("test_key", rate_limit):
                    successes += 1
            results.put(successes)
        
        # Create multiple threads
        threads = [
            threading.Thread(target=make_requests)
            for _ in range(3)
        ]
        
        # Run threads
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            
        # Collect results
        total_successes = sum(results.get() for _ in range(3))
        assert total_successes <= rate_limit.limit

    def test_different_keys(self, memory_storage):
        """Test that different keys are tracked separately"""
        rate_limit = RateLimit(limit=2, window=60, type=RateLimitType.REQUESTS)
        
        # Both keys should work independently
        assert memory_storage.check_limit("key1", rate_limit)
        assert memory_storage.check_limit("key1", rate_limit)
        assert not memory_storage.check_limit("key1", rate_limit)
        
        assert memory_storage.check_limit("key2", rate_limit)
        assert memory_storage.check_limit("key2", rate_limit)
        assert not memory_storage.check_limit("key2", rate_limit)
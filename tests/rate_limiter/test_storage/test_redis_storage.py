import pytest
import redis
from datetime import datetime
from rate_limiter.storage.redis_storage import RedisStorage
from rate_limiter.types import RateLimit, RateLimitType

class TestRedisStorage:
    def test_check_limit_within_limit(self, redis_storage, mock_redis, mock_time):
        """Test check_limit with Redis when within limits"""
        rate_limit = RateLimit(limit=5, window=60, type=RateLimitType.REQUESTS)
        
        # Setup Redis mock
        mock_redis.pipeline.return_value.execute.return_value = [
            True,  # zremrangebyscore
            1,    # zadd
            3,    # zcount
            True  # expire
        ]
        
        assert redis_storage.check_limit("test_key", rate_limit)
        
        # Verify Redis calls
        mock_redis.pipeline.assert_called_once()
    
    def test_check_limit_exceeded(self, redis_storage, mock_redis, mock_time):
        """Test check_limit with Redis when limit is exceeded"""
        rate_limit = RateLimit(limit=2, window=60, type=RateLimitType.REQUESTS)
        
        # Setup Redis mock to indicate limit exceeded
        mock_redis.pipeline.return_value.execute.return_value = [
            True,  # zremrangebyscore
            1,    # zadd
            3,    # zcount (> limit)
            True  # expire
        ]
        
        assert not redis_storage.check_limit("test_key", rate_limit)
    
    def test_redis_error_fallback(self, redis_storage, mock_redis):
        """Test fallback behavior when Redis fails"""
        rate_limit = RateLimit(limit=5, window=60, type=RateLimitType.REQUESTS)
        
        # Make Redis fail
        mock_redis.pipeline.side_effect = redis.RedisError("Connection failed")
        
        # Should still work using in-memory fallback
        assert redis_storage.check_limit("test_key", rate_limit)

    def test_get_limit_info(self, redis_storage, mock_redis, mock_time):
        """Test getting rate limit information from Redis"""
        rate_limit = RateLimit(limit=5, window=60, type=RateLimitType.REQUESTS)
        
        # Setup Redis mock
        mock_redis.zcount.return_value = 3
        
        info = redis_storage.get_limit_info("test_key", rate_limit)
        
        assert info["limit"] == 5
        assert info["remaining"] == 2
        assert isinstance(info["reset"], str)
        assert info["window"] == 60
        
        # Verify reset is valid ISO format
        datetime.fromisoformat(info["reset"])

    def test_cleanup_expired(self, redis_storage, mock_redis, mock_time):
        """Test cleanup of expired entries"""
        rate_limit = RateLimit(limit=5, window=60, type=RateLimitType.REQUESTS)
        
        # Setup Redis mock
        mock_redis.pipeline.return_value.execute.return_value = [
            True,  # zremrangebyscore
            1,    # zadd
            1,    # zcount
            True  # expire
        ]
        
        redis_storage.check_limit("test_key", rate_limit)
        
        # Verify cleanup call
        pipeline = mock_redis.pipeline.return_value
        pipeline.zremrangebyscore.assert_called_once_with(
            "test_key",
            0,
            mock_time.return_value - rate_limit.window
        )
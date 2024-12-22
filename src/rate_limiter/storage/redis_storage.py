import redis
from typing import Dict, Any
from datetime import datetime
import time
from .base import BaseStorage
from ..types import RateLimit

class RedisStorage(BaseStorage):
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    def check_limit(self, key: str, rate_limit: RateLimit) -> bool:
        """Check if request is within rate limit using Redis"""
        current_time = time.time()
        
        pipeline = self.redis.pipeline()
        
        # Clean up old counts
        pipeline.zremrangebyscore(key, 0, current_time - rate_limit.window)
        
        # Add current request
        pipeline.zadd(key, {str(current_time): current_time})
        
        # Get count in window
        pipeline.zcount(key, current_time - rate_limit.window, current_time)
        
        # Set expiry on key
        pipeline.expire(key, rate_limit.window)
        
        # Execute pipeline
        _, _, count, _ = pipeline.execute()
        
        return count <= rate_limit.limit

    def get_limit_info(self, key: str, rate_limit: RateLimit) -> Dict[str, Any]:
        """Get information about current rate limit status"""
        current_time = time.time()
        
        count = self.redis.zcount(
            key,
            current_time - rate_limit.window,
            current_time
        )
        
        remaining = max(0, rate_limit.limit - count)
        reset_time = current_time + rate_limit.window
        
        return {
            "limit": rate_limit.limit,
            "remaining": remaining,
            "reset": datetime.fromtimestamp(reset_time).isoformat(),
            "window": rate_limit.window
        } 
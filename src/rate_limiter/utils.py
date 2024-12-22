from typing import Optional, Dict, Any
import redis
import logging
from .core import RateLimiter
from .storage.redis_storage import RedisStorage
from .storage.memory_storage import MemoryStorage

def setup_rate_limiter(
    redis_url: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> RateLimiter:
    """Create and configure rate limiter instance"""
    storage = None
    
    if redis_url:
        try:
            redis_client = redis.from_url(redis_url)
            storage = RedisStorage(redis_client)
            logging.info("Using Redis storage for rate limiting")
        except Exception as e:
            logging.warning(f"Failed to connect to Redis: {str(e)}")
            storage = MemoryStorage()
            logging.info("Falling back to in-memory storage for rate limiting")
    else:
        storage = MemoryStorage()
        logging.info("Using in-memory storage for rate limiting")
        
    return RateLimiter(storage=storage, config=config) 
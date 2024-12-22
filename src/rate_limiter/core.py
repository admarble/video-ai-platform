from typing import Dict, Optional, Any
import logging
from .types import RateLimitType, RateLimit
from .storage.base import BaseStorage
from .storage.memory_storage import MemoryStorage

class RateLimiter:
    """Rate limiter with support for different limit types and scopes"""
    
    def __init__(
        self,
        storage: Optional[BaseStorage] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.storage = storage or MemoryStorage()
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Setup default rate limits
        self._setup_default_limits()
        
    def _setup_default_limits(self) -> None:
        """Setup default rate limits for different types"""
        self.default_limits = {
            RateLimitType.REQUESTS: RateLimit(
                limit=self.config.get('default_requests_per_minute', 60),
                window=60,
                type=RateLimitType.REQUESTS
            ),
            RateLimitType.PROCESSING: RateLimit(
                limit=self.config.get('default_processes_per_hour', 10),
                window=3600,
                type=RateLimitType.PROCESSING
            ),
            RateLimitType.UPLOADS: RateLimit(
                limit=self.config.get('default_uploads_per_hour', 20),
                window=3600,
                type=RateLimitType.UPLOADS
            ),
            RateLimitType.API: RateLimit(
                limit=self.config.get('default_api_requests_per_minute', 30),
                window=60,
                type=RateLimitType.API
            )
        }
    
    def _get_storage_key(self, rate_limit: RateLimit, scope: str) -> str:
        """Generate storage key for rate limit tracking"""
        return f"rate_limit:{rate_limit.type.value}:{scope}"
        
    def check_rate_limit(
        self,
        limit_type: RateLimitType,
        scope: str,
        custom_limit: Optional[RateLimit] = None
    ) -> bool:
        """Check if request is within rate limits"""
        rate_limit = custom_limit or self.default_limits[limit_type]
        key = self._get_storage_key(rate_limit, scope)
        return self.storage.check_limit(key, rate_limit)
            
    def get_limit_info(
        self,
        limit_type: RateLimitType,
        scope: str,
        custom_limit: Optional[RateLimit] = None
    ) -> Dict[str, Any]:
        """Get information about current rate limit status"""
        rate_limit = custom_limit or self.default_limits[limit_type]
        key = self._get_storage_key(rate_limit, scope)
        return self.storage.get_limit_info(key, rate_limit) 
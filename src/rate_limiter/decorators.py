from functools import wraps
from typing import Optional, Callable, Any
from .types import RateLimitType, RateLimit
from .core import RateLimiter

class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""
    pass

class RateLimitDecorator:
    """Decorator for rate limiting function calls"""
    
    def __init__(
        self,
        rate_limiter: RateLimiter,
        limit_type: RateLimitType,
        scope_func: Optional[Callable] = None,
        custom_limit: Optional[RateLimit] = None
    ):
        self.rate_limiter = rate_limiter
        self.limit_type = limit_type
        self.scope_func = scope_func
        self.custom_limit = custom_limit
        
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get scope
            scope = "global"
            if self.scope_func:
                scope = self.scope_func(*args, **kwargs)
                
            # Check rate limit
            if not self.rate_limiter.check_rate_limit(
                self.limit_type,
                scope,
                self.custom_limit
            ):
                limit_info = self.rate_limiter.get_limit_info(
                    self.limit_type,
                    scope,
                    self.custom_limit
                )
                raise RateLimitExceeded(
                    f"Rate limit exceeded. Try again after {limit_info['reset']}"
                )
                
            return func(*args, **kwargs)
            
        return wrapper 
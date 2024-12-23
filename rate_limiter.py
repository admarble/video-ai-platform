from typing import Dict, Optional, Tuple
import time
from collections import defaultdict
import logging
from dataclasses import dataclass
from datetime import datetime

@dataclass
class RateLimit:
    """Rate limit configuration"""
    requests: int
    window: int  # in seconds
    
@dataclass
class RateTracker:
    """Tracks request counts and timestamps"""
    count: int = 0
    window_start: float = 0.0

class RateLimiter:
    """Implements rate limiting functionality"""
    
    def __init__(self):
        self.limits: Dict[str, Dict[str, RateTracker]] = defaultdict(lambda: defaultdict(RateTracker))
        self.logger = logging.getLogger(__name__)

    def check_rate_limit(
        self,
        key: str,
        rule_name: str,
        limit: RateLimit
    ) -> Tuple[bool, Optional[int]]:
        """
        Check if request is within rate limits
        Returns (allowed, retry_after)
        """
        try:
            tracker = self.limits[rule_name][key]
            current_time = time.time()
            
            # Reset tracker if window has expired
            if current_time - tracker.window_start >= limit.window:
                tracker.count = 0
                tracker.window_start = current_time
                
            # Check if limit is exceeded
            if tracker.count >= limit.requests:
                retry_after = int(limit.window - (current_time - tracker.window_start))
                return False, retry_after
                
            # Increment counter
            tracker.count += 1
            if tracker.count == 1:
                tracker.window_start = current_time
                
            return True, None
            
        except Exception as e:
            self.logger.error(f"Rate limit check error: {str(e)}")
            return True, None  # Fail open on errors

    def parse_rate_limit(self, config: Dict[str, int]) -> Optional[RateLimit]:
        """Parse rate limit configuration"""
        try:
            if 'requests_per_minute' in config:
                return RateLimit(
                    requests=config['requests_per_minute'],
                    window=60
                )
            elif 'requests_per_hour' in config:
                return RateLimit(
                    requests=config['requests_per_hour'],
                    window=3600
                )
            elif 'requests_per_day' in config:
                return RateLimit(
                    requests=config['requests_per_day'],
                    window=86400
                )
            return None
        except Exception as e:
            self.logger.error(f"Rate limit parse error: {str(e)}")
            return None

    def get_rate_limit_headers(
        self,
        key: str,
        rule_name: str,
        limit: RateLimit
    ) -> Dict[str, str]:
        """Get rate limit headers for response"""
        tracker = self.limits[rule_name][key]
        remaining = max(0, limit.requests - tracker.count)
        reset_time = int(tracker.window_start + limit.window)
        
        return {
            'X-RateLimit-Limit': str(limit.requests),
            'X-RateLimit-Remaining': str(remaining),
            'X-RateLimit-Reset': str(reset_time)
        }

    def clear_expired(self) -> None:
        """Clear expired rate limit trackers"""
        current_time = time.time()
        for rule_name in list(self.limits.keys()):
            for key in list(self.limits[rule_name].keys()):
                tracker = self.limits[rule_name][key]
                if current_time - tracker.window_start >= 86400:  # Clear after 24 hours
                    del self.limits[rule_name][key] 
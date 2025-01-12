from typing import Dict, Any, DefaultDict
from collections import defaultdict
import threading
import time
from datetime import datetime
from .base import BaseStorage
from ..types import RateLimit

class MemoryStorage(BaseStorage):
    def __init__(self):
        self.counters: DefaultDict[str, DefaultDict[float, int]] = defaultdict(lambda: defaultdict(int))
        self.lock = threading.Lock()

    def check_limit(self, key: str, rate_limit: RateLimit) -> bool:
        """Check if request is within rate limit using local storage"""
        current_time = time.time()
        
        with self.lock:
            # Clean up old counts
            self.counters[key] = {
                ts: count for ts, count in self.counters[key].items()
                if ts > current_time - rate_limit.window
            }
            
            # Add current request
            self.counters[key][current_time] = 1
            
            # Count requests in window
            count = sum(self.counters[key].values())
            
            return count <= rate_limit.limit

    def get_limit_info(self, key: str, rate_limit: RateLimit) -> Dict[str, Any]:
        """Get information about current rate limit status"""
        current_time = time.time()
        
        with self.lock:
            count = sum(1 for ts in self.counters[key]
                       if ts > current_time - rate_limit.window)
            
            remaining = max(0, rate_limit.limit - count)
            reset_time = current_time + rate_limit.window
            
            return {
                "limit": rate_limit.limit,
                "remaining": remaining,
                "reset": datetime.fromtimestamp(reset_time).isoformat(),
                "window": rate_limit.window
            } 
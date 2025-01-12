from abc import ABC, abstractmethod
from typing import Dict, Any
from ..types import RateLimit

class BaseStorage(ABC):
    """Base class for rate limit storage implementations"""
    
    @abstractmethod
    def check_limit(self, key: str, rate_limit: RateLimit) -> bool:
        """Check if request is within rate limit"""
        pass
    
    @abstractmethod
    def get_limit_info(self, key: str, rate_limit: RateLimit) -> Dict[str, Any]:
        """Get current rate limit status"""
        pass 
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class RateLimitType(Enum):
    """Types of rate limits"""
    REQUESTS = "requests"  # General request rate limiting
    PROCESSING = "processing"  # Video processing rate limiting
    UPLOADS = "uploads"  # File upload rate limiting
    API = "api"  # API endpoint rate limiting

@dataclass
class RateLimit:
    """Defines a rate limit configuration"""
    limit: int  # Maximum number of requests
    window: int  # Time window in seconds
    type: RateLimitType
    scope: str = "global"  # Identifier for the scope (e.g., user_id, ip_address) 
from typing import TypeVar, Callable, Optional, Any
from dataclasses import dataclass
import time
import logging
import functools
from enum import Enum
import random

T = TypeVar('T')

class RetryStrategy(Enum):
    """Different retry strategies"""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    RANDOM = "random"

@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter: bool = True
    exceptions: tuple = (Exception,)

class RetryError(Exception):
    """Raised when all retry attempts fail"""
    def __init__(self, last_error: Exception, attempts: int):
        self.last_error = last_error
        self.attempts = attempts
        super().__init__(f"Failed after {attempts} attempts. Last error: {str(last_error)}")

class RetryHandler:
    """Handles retry logic with various strategies"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay based on retry strategy"""
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.initial_delay
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.initial_delay * (2 ** (attempt - 1))
        else:  # RANDOM
            delay = random.uniform(self.config.initial_delay, self.config.max_delay)
            
        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)
        
        # Add jitter if enabled
        if self.config.jitter:
            delay *= random.uniform(0.8, 1.2)
            
        return delay
        
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for adding retry behavior to functions"""
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_error = None
            
            for attempt in range(1, self.config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                    
                except self.config.exceptions as e:
                    last_error = e
                    
                    if attempt == self.config.max_attempts:
                        raise RetryError(last_error, attempt)
                        
                    delay = self.calculate_delay(attempt)
                    
                    self.logger.warning(
                        f"Attempt {attempt} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    time.sleep(delay)
                    
            raise RetryError(last_error, self.config.max_attempts)
            
        return wrapper 
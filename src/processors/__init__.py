"""Video processing module with circuit breaker pattern."""

from .video_exceptions import (
    VideoProcessingError,
    VideoCorruptedError,
    ResourceExhaustedError,
    ModelError,
    ProcessingTimeoutError,
    CircuitOpenError
)

from .video_circuit_config import (
    CircuitConfig,
    VideoCircuitConfig
)

from .video_circuit_breaker import (
    CircuitState,
    CircuitBreaker,
    VideoCircuitBreaker
)

from .video_circuit_decorator import (
    CircuitBreakerRegistry,
    video_circuit_breaker
)

__all__ = [
    # Exceptions
    'VideoProcessingError',
    'VideoCorruptedError',
    'ResourceExhaustedError',
    'ModelError',
    'ProcessingTimeoutError',
    'CircuitOpenError',
    
    # Configurations
    'CircuitConfig',
    'VideoCircuitConfig',
    
    # Core classes
    'CircuitState',
    'CircuitBreaker',
    'VideoCircuitBreaker',
    
    # Decorator and registry
    'CircuitBreakerRegistry',
    'video_circuit_breaker'
] 
"""Configuration classes for video circuit breaker."""

from dataclasses import dataclass
from typing import Tuple, Type
from .video_exceptions import VideoProcessingError

@dataclass
class CircuitConfig:
    """Base circuit breaker configuration."""
    failure_threshold: int = 5        # Number of failures before opening
    reset_timeout: int = 60           # Seconds before attempting reset
    half_open_limit: int = 3          # Number of requests to test in half-open state
    window_size: int = 60             # Time window for failure counting (seconds)
    success_threshold: int = 2        # Successes needed to close circuit
    exceptions: Tuple[Type[Exception], ...] = (Exception,)  # Exception types to count as failures

@dataclass
class VideoCircuitConfig(CircuitConfig):
    """Circuit breaker configuration specific to video processing."""
    max_processing_time: int = 300    # Maximum processing time in seconds
    min_free_memory: int = 1024      # Minimum free memory in MB
    corruption_threshold: int = 3     # Max corrupted files before opening
    model_error_threshold: int = 5    # Max model errors before opening
    resource_check_interval: int = 30  # Resource check interval in seconds
    exceptions: Tuple[Type[Exception], ...] = (VideoProcessingError,)  # Override to catch video-specific errors 
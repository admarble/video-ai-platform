"""Video processing specific exceptions."""

class VideoProcessingError(Exception):
    """Base exception for video processing errors."""
    pass

class VideoCorruptedError(VideoProcessingError):
    """Raised when video file is corrupted."""
    pass

class ResourceExhaustedError(VideoProcessingError):
    """Raised when system resources are exhausted."""
    pass

class ModelError(VideoProcessingError):
    """Raised when ML model fails."""
    pass

class ProcessingTimeoutError(VideoProcessingError):
    """Raised when processing exceeds time limit."""
    pass

class CircuitOpenError(VideoProcessingError):
    """Raised when circuit breaker is open."""
    pass 
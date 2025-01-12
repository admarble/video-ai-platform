class VideoProcessingError(Exception):
    """Raised when video processing operations fail"""
    pass

class AudioProcessingError(Exception):
    """Raised when audio processing operations fail"""
    pass

class ServiceInitializationError(Exception):
    """Raised when services fail to initialize"""
    pass
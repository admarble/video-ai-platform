class ServiceCleanupError(Exception):
    """Raised when service cleanup fails"""
    pass

class ServiceInitializationError(Exception):
    """Raised when service initialization fails"""
    pass

class VideoProcessingError(Exception):
    """Raised when video processing fails"""
    pass 
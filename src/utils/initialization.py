from pathlib import Path
from src.core.config import settings
from src.processors.video_processor import create_video_processor
from src.core.model_loader import init_model_config

def initialize_processing_environment():
    """Initialize processing environment and return processor"""
    # Create required directories
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.PROCESSING_DIR).mkdir(parents=True, exist_ok=True)
    
    # Initialize model configuration
    model_config = init_model_config()
    
    # Create and return video processor
    return create_video_processor(
        upload_dir=settings.UPLOAD_DIR,
        processing_dir=settings.PROCESSING_DIR,
        model_config=model_config
    ) 
import logging
import logging.config
import os
from pathlib import Path
from typing import Optional

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> None:
    """
    Configure logging with console and optional file output
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        log_format: Format string for log messages
    """
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": log_format,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console"],
                "level": log_level,
                "propagate": True,
            },
            "src.processors": {  # Logger for processors module
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
            "ffmpeg": {  # Logger for FFmpeg-specific messages
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
        },
    }
    
    # Add file handler if log file is specified
    if log_file:
        log_dir = log_file.parent
        if not log_dir.exists():
            log_dir.mkdir(parents=True)
            
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "detailed",
            "filename": str(log_file),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8",
        }
        
        # Add file handler to all loggers
        for logger in config["loggers"].values():
            logger["handlers"].append("file")
    
    logging.config.dictConfig(config)
    
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)

# Example usage in other modules:
# from src.utils.logging_config import get_logger
# logger = get_logger(__name__) 
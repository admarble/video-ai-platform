"""
Example demonstrating how to use the Cuthrough logging system.
"""

from pathlib import Path
from src.core.logging import LoggingManager, LogLevel

def main():
    # Initialize logging manager with default configuration
    base_dir = Path(__file__).parent.parent
    logging_manager = LoggingManager(base_dir)

    # Get different loggers
    processing_logger = logging_manager.get_logger("video_processing")
    error_logger = logging_manager.get_logger("error")
    system_logger = logging_manager.get_logger("system")
    access_logger = logging_manager.get_logger("access")

    # Example: Video processing logs
    processing_logger.info("Starting video processing for file: example.mp4")
    processing_logger.debug("Frame rate: 30fps, Resolution: 1920x1080")
    processing_logger.warning("Low disk space warning: 15% remaining")

    # Example: Error logs with exception
    try:
        raise ValueError("Invalid video format")
    except Exception as e:
        error_logger.error("Failed to process video", exc_info=True)

    # Example: System logs
    system_logger.info("System startup complete")
    system_logger.info("CPU Usage: 45%, Memory: 2.5GB")
    system_logger.warning("High memory usage detected")

    # Example: Access logs
    access_logger.info("GET /api/v1/videos - 200 OK - User: john_doe")
    access_logger.info("POST /api/v1/process - 201 Created - User: admin")
    access_logger.warning("Failed login attempt - IP: 192.168.1.100")

    # Example: Using custom configuration from file
    config_path = base_dir / "src" / "core" / "logging" / "config" / "default_logging.yaml"
    custom_logging_manager = LoggingManager.from_config_file(base_dir, config_path)
    
    # Get logger from custom configuration
    custom_logger = custom_logging_manager.get_logger("video_processing")
    custom_logger.info("Using custom configured logger")

    # Example: Cleanup and rotation
    logging_manager.cleanup_old_logs(days=30)
    logging_manager.rotate_all_logs()

if __name__ == "__main__":
    main() 
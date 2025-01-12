"""Tests for the logging system."""

import pytest
import os
import json
import yaml
from pathlib import Path
import tempfile
import shutil
import logging
from datetime import datetime, timedelta

from src.core.logging import LoggingManager, LogLevel, LoggerConfig
from src.models.video import VideoMetadata, ProcessingResults

class TestLoggingManager:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test logs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
        
    @pytest.fixture
    def logging_manager(self, temp_dir):
        """Create a logging manager instance for testing."""
        config = {
            "test_logger": {
                "level": "INFO",
                "max_bytes": 1024,  # Small size for testing rotation
                "backup_count": 3,
                "format": "%(asctime)s - %(levelname)s - %(message)s",
                "rotation_when": "S",  # Rotate every second for testing
                "rotation_interval": 1,
                "use_console": False
            }
        }
        return LoggingManager(base_dir=temp_dir, config=config)
        
    def test_logger_creation(self, logging_manager):
        """Test basic logger creation and configuration."""
        logger = logging_manager.get_logger("test_logger")
        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.INFO
        
    def test_log_rotation_by_size(self, logging_manager, temp_dir):
        """Test that logs are rotated when they exceed max_bytes."""
        logger = logging_manager.get_logger("test_logger")
        log_path = temp_dir / "logs" / "test_logger.log"
        
        # Write enough data to trigger rotation
        large_message = "x" * 512  # Half of max_bytes
        for _ in range(4):  # Should create 2 backup files
            logger.info(large_message)
            
        # Check that backup files were created
        assert log_path.exists()
        assert (log_path.parent / "test_logger.log.1").exists()
        assert (log_path.parent / "test_logger.log.2").exists()
        
    def test_log_rotation_by_time(self, logging_manager, temp_dir):
        """Test that logs are rotated based on time."""
        logger = logging_manager.get_logger("test_logger")
        log_path = temp_dir / "logs" / "test_logger.timed.log"
        
        # Write logs and wait for rotation
        logger.info("First message")
        import time
        time.sleep(1.1)  # Wait for rotation interval
        logger.info("Second message")
        
        # Check that timed backup file was created
        backup_files = list(log_path.parent.glob("test_logger.timed.log.*"))
        assert len(backup_files) > 0
        
    def test_log_cleanup(self, logging_manager, temp_dir):
        """Test cleanup of old log files."""
        logger = logging_manager.get_logger("test_logger")
        log_dir = temp_dir / "logs"
        
        # Create some old log files
        old_file = log_dir / "old.log"
        old_file.write_text("old log")
        os.utime(old_file, (datetime.now().timestamp() - 31*24*60*60,) * 2)
        
        # Create a recent log file
        recent_file = log_dir / "recent.log"
        recent_file.write_text("recent log")
        
        # Run cleanup
        logging_manager.cleanup_old_logs(days=30)
        
        # Check that old file was removed but recent file remains
        assert not old_file.exists()
        assert recent_file.exists()
        
    def test_custom_format(self, logging_manager, temp_dir):
        """Test custom log format configuration."""
        config = {
            "custom_format": {
                "level": "INFO",
                "format": "CUSTOM - %(levelname)s - %(message)s"
            }
        }
        manager = LoggingManager(base_dir=temp_dir, config=config)
        logger = manager.get_logger("custom_format")
        log_path = temp_dir / "logs" / "custom_format.log"
        
        logger.info("Test message")
        
        log_content = log_path.read_text()
        assert "CUSTOM - INFO - Test message" in log_content
        
    def test_multiple_loggers(self, temp_dir):
        """Test multiple loggers with different configurations."""
        config = {
            "error_logger": {
                "level": "ERROR",
                "format": "ERROR: %(message)s"
            },
            "debug_logger": {
                "level": "DEBUG",
                "format": "DEBUG: %(message)s"
            }
        }
        manager = LoggingManager(base_dir=temp_dir, config=config)
        
        error_logger = manager.get_logger("error_logger")
        debug_logger = manager.get_logger("debug_logger")
        
        error_logger.error("Error message")
        debug_logger.debug("Debug message")
        
        error_log = (temp_dir / "logs" / "error_logger.log").read_text()
        debug_log = (temp_dir / "logs" / "debug_logger.log").read_text()
        
        assert "ERROR: Error message" in error_log
        assert "DEBUG: Debug message" in debug_log
        
    def test_yaml_config_loading(self, temp_dir):
        """Test loading configuration from YAML file."""
        config_path = temp_dir / "logging_config.yaml"
        config = {
            "yaml_logger": {
                "level": "INFO",
                "format": "YAML: %(message)s"
            }
        }
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            
        manager = LoggingManager.from_config_file(temp_dir, config_path)
        logger = manager.get_logger("yaml_logger")
        
        logger.info("Test message")
        
        log_content = (temp_dir / "logs" / "yaml_logger.log").read_text()
        assert "YAML: Test message" in log_content
        
    def test_error_logging_with_exception(self, logging_manager, temp_dir):
        """Test logging exceptions with traceback."""
        logger = logging_manager.get_logger("test_logger")
        log_path = temp_dir / "logs" / "test_logger.log"
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            logger.error("Error occurred", exc_info=True)
            
        log_content = log_path.read_text()
        assert "ValueError: Test error" in log_content
        assert "Traceback" in log_content
        
    def test_video_processing_logging(self, temp_dir):
        """Test logging in video processing context."""
        config = {
            "video_processing": {
                "level": "INFO",
                "format": "%(asctime)s - [%(levelname)s] %(message)s - %(video_id)s"
            }
        }
        manager = LoggingManager(base_dir=temp_dir, config=config)
        logger = manager.get_logger("video_processing")
        
        # Simulate video processing logs
        video_id = "test_video_123"
        logger.info(
            "Starting video processing",
            extra={"video_id": video_id}
        )
        logger.info(
            "Extracted 100 frames",
            extra={"video_id": video_id}
        )
        
        log_content = (temp_dir / "logs" / "video_processing.log").read_text()
        assert video_id in log_content
        assert "Starting video processing" in log_content
        assert "Extracted 100 frames" in log_content
        
    def test_access_logging(self, temp_dir):
        """Test API access logging format."""
        config = {
            "access": {
                "level": "INFO",
                "format": "%(asctime)s - %(method)s %(path)s - %(status)s - %(message)s"
            }
        }
        manager = LoggingManager(base_dir=temp_dir, config=config)
        logger = manager.get_logger("access")
        
        # Simulate API access logs
        logger.info(
            "Video upload request",
            extra={
                "method": "POST",
                "path": "/upload",
                "status": "SUCCESS"
            }
        )
        
        log_content = (temp_dir / "logs" / "access.log").read_text()
        assert "POST /upload - SUCCESS" in log_content
        
    def test_log_rotation_all(self, logging_manager, temp_dir):
        """Test manual rotation of all logs."""
        logger = logging_manager.get_logger("test_logger")
        logger.info("Message before rotation")
        
        logging_manager.rotate_all_logs()
        
        logger.info("Message after rotation")
        
        # Check that rotation created backup files
        backup_files = list((temp_dir / "logs").glob("test_logger.log.*"))
        assert len(backup_files) > 0 
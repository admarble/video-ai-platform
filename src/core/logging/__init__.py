"""
Core logging module providing a centralized logging system with rotation and configuration capabilities.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
import yaml
from enum import Enum

class LogLevel(str, Enum):
    """Log levels with string values for configuration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class LoggerConfig:
    """Configuration for individual logger"""
    name: str
    level: LogLevel
    file_path: Path
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    rotation_when: str = "midnight"
    rotation_interval: int = 1
    use_console: bool = True

class LoggingManager:
    """Manages logging configuration and rotation"""
    
    def __init__(
        self,
        base_dir: Path,
        config: Optional[Dict[str, Any]] = None,
        default_level: LogLevel = LogLevel.INFO
    ):
        self.base_dir = base_dir
        self.config = config or {}
        self.default_level = default_level
        self.loggers: Dict[str, logging.Logger] = {}
        
        # Create logs directory
        self.logs_dir = base_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize default loggers
        self._setup_default_loggers()
        
    def _setup_default_loggers(self):
        """Setup default loggers for different components"""
        default_configs = {
            "video_processing": LoggerConfig(
                name="video_processing",
                level=LogLevel.INFO,
                file_path=self.logs_dir / "processing.log",
                max_bytes=20 * 1024 * 1024,  # 20MB
                backup_count=10,
                rotation_when="D",  # Daily rotation
                rotation_interval=1
            ),
            "error": LoggerConfig(
                name="error",
                level=LogLevel.ERROR,
                file_path=self.logs_dir / "error.log",
                max_bytes=50 * 1024 * 1024,  # 50MB
                backup_count=20,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(exc_info)s"
            ),
            "system": LoggerConfig(
                name="system",
                level=LogLevel.INFO,
                file_path=self.logs_dir / "system.log",
                max_bytes=10 * 1024 * 1024,  # 10MB
                backup_count=5,
                rotation_when="H",  # Hourly rotation
                rotation_interval=12
            ),
            "access": LoggerConfig(
                name="access",
                level=LogLevel.INFO,
                file_path=self.logs_dir / "access.log",
                max_bytes=15 * 1024 * 1024,  # 15MB
                backup_count=7,
                format="%(asctime)s - %(message)s"
            )
        }
        
        # Override with custom configs if provided
        for name, default_config in default_configs.items():
            custom_config = self.config.get(name, {})
            config = self._merge_configs(default_config, custom_config)
            self.setup_logger(config)
            
    def _merge_configs(
        self,
        default_config: LoggerConfig,
        custom_config: Dict[str, Any]
    ) -> LoggerConfig:
        """Merge custom config with default config"""
        if not custom_config:
            return default_config
            
        merged_config = LoggerConfig(
            name=custom_config.get('name', default_config.name),
            level=LogLevel(custom_config.get('level', default_config.level)),
            file_path=Path(custom_config.get('file_path', default_config.file_path)),
            max_bytes=custom_config.get('max_bytes', default_config.max_bytes),
            backup_count=custom_config.get('backup_count', default_config.backup_count),
            format=custom_config.get('format', default_config.format),
            rotation_when=custom_config.get('rotation_when', default_config.rotation_when),
            rotation_interval=custom_config.get('rotation_interval', default_config.rotation_interval),
            use_console=custom_config.get('use_console', default_config.use_console)
        )
        
        return merged_config
        
    def setup_logger(self, config: LoggerConfig) -> logging.Logger:
        """Setup individual logger with rotation"""
        logger = logging.getLogger(config.name)
        logger.setLevel(config.level.value)
        
        # Remove existing handlers
        logger.handlers = []
        
        # Create formatter
        formatter = logging.Formatter(config.format)
        
        # Add rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            config.file_path,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Add timed rotating handler
        timed_handler = logging.handlers.TimedRotatingFileHandler(
            config.file_path.with_suffix('.timed.log'),
            when=config.rotation_when,
            interval=config.rotation_interval,
            backupCount=config.backup_count
        )
        timed_handler.setFormatter(formatter)
        logger.addHandler(timed_handler)
        
        # Add console handler if enabled
        if config.use_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
        self.loggers[config.name] = logger
        return logger
        
    def get_logger(self, name: str) -> logging.Logger:
        """Get logger by name"""
        return self.loggers.get(name) or logging.getLogger(name)
        
    @classmethod
    def from_config_file(
        cls,
        base_dir: Path,
        config_path: Path
    ) -> 'LoggingManager':
        """Create LoggingManager from config file"""
        if not config_path.exists():
            return cls(base_dir)
            
        # Load config file
        with open(config_path) as f:
            if config_path.suffix == '.yaml':
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
                
        return cls(base_dir, config)
        
    def cleanup_old_logs(self, days: int = 30):
        """Clean up log files older than specified days"""
        import time
        current_time = time.time()
        max_age = days * 24 * 60 * 60
        
        for log_file in self.logs_dir.glob("*.log*"):
            try:
                if current_time - log_file.stat().st_mtime > max_age:
                    log_file.unlink()
            except Exception as e:
                print(f"Error cleaning up log file {log_file}: {str(e)}")
                
    def rotate_all_logs(self):
        """Force rotation of all log files"""
        for logger in self.loggers.values():
            for handler in logger.handlers:
                if isinstance(handler, (
                    logging.handlers.RotatingFileHandler,
                    logging.handlers.TimedRotatingFileHandler
                )):
                    handler.doRollover() 
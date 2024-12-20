"""
Configuration management system for the Video AI Platform.
Handles loading, validation, and environment-specific configurations.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
import os
from dataclasses import dataclass, asdict
import logging
from enum import Enum
import secrets
import copy
import json
import time
import hashlib
from cryptography.fernet import Fernet
from .models import Config, ConfigVersion, Environment
from .secrets import SecretsManager
from .watcher import ConfigObserver

class ConfigurationError(Exception):
    """Custom exception for configuration errors"""
    pass

class ConfigManager:
    """Enhanced configuration manager with dynamic updates"""
    
    def __init__(self):
        self._config: Optional[Config] = None
        self._config_path: Optional[Path] = None
        self.logger = logging.getLogger(__name__)
        self._secrets_manager: Optional[SecretsManager] = None
        self._version_history: List[ConfigVersion] = []
        self._observers: List[callable] = []
        self._file_observer: Optional[ConfigObserver] = None
        
    def load_config(self, config_path: str, environment: Environment) -> Config:
        """Load configuration with secrets management and version tracking"""
        self._config_path = Path(config_path)
        
        if not self._config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
            
        try:
            # Load and merge configurations
            config_data = self._load_and_merge_configs(environment)
            
            # Initialize secrets manager
            self._init_secrets_manager(config_data['secrets'])
            
            # Load secrets into configuration
            config_data = self._inject_secrets(config_data)
            
            # Create config instance with version tracking
            self._config = Config.from_dict(config_data)
            
            # Track version
            self._version_history.append(self._config.version_info)
            
            # Start file watcher
            self._start_file_watcher()
            
            return self._config
            
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {str(e)}")
    
    def _load_and_merge_configs(self, environment: Environment) -> dict:
        """Load and merge base and environment-specific configs"""
        with open(self._config_path) as f:
            config_data = yaml.safe_load(f)
            
        env_config_path = self._config_path.parent / f"config.{environment.value}.yaml"
        if env_config_path.exists():
            with open(env_config_path) as f:
                env_config = yaml.safe_load(f)
                config_data = self._deep_merge(config_data, env_config)
                
        return config_data
    
    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge two dictionaries"""
        merged = base.copy()
        
        for key, value in override.items():
            if isinstance(value, dict) and key in merged:
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
                
        return merged
    
    def _init_secrets_manager(self, secrets_config: dict) -> None:
        """Initialize secrets manager"""
        self._secrets_manager = SecretsManager(
            secrets_config['vault_path'],
            secrets_config['encryption_key']
        )
        
        # Check if key rotation is needed
        if time.time() - secrets_config['last_rotation'] > secrets_config['key_rotation_days'] * 86400:
            new_key = Fernet.generate_key().decode()
            self._secrets_manager.rotate_encryption_key(new_key)
            secrets_config.update({
                'encryption_key': new_key,
                'last_rotation': time.time()
            })
    
    def _inject_secrets(self, config_data: dict) -> dict:
        """Inject secrets into configuration"""
        def inject_secret(obj: Any) -> Any:
            if isinstance(obj, str) and obj.startswith("secret://"):
                secret_key = obj[9:]
                secret_value = self._secrets_manager.get_secret(secret_key)
                if secret_value is None:
                    raise ConfigurationError(f"Secret not found: {secret_key}")
                return secret_value
            elif isinstance(obj, dict):
                return {k: inject_secret(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [inject_secret(v) for v in obj]
            return obj
            
        return inject_secret(config_data)
    
    def _start_file_watcher(self) -> None:
        """Start watching configuration files for changes"""
        if self._file_observer is None:
            self._file_observer = ConfigObserver(
                str(self._config_path.parent),
                self
            )
            self._file_observer.start()
    
    def add_observer(self, callback: callable) -> None:
        """Add observer for configuration changes"""
        self._observers.append(callback)
    
    def reload_config(self) -> None:
        """Reload configuration and notify observers"""
        if self._config and self._config_path:
            new_config = self.load_config(
                str(self._config_path),
                self._config.environment
            )
            
            # Notify observers of change
            for observer in self._observers:
                observer(new_config)
    
    def store_secret(self, key: str, value: str) -> None:
        """Store a new secret"""
        if self._secrets_manager is None:
            raise ConfigurationError("Secrets manager not initialized")
        self._secrets_manager.store_secret(key, value)
    
    def get_version_history(self) -> List[ConfigVersion]:
        """Get configuration version history"""
        return self._version_history
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        if self._file_observer:
            self._file_observer.stop()

class Environment(Enum):
    """Environment types for configuration"""
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"

@dataclass
class ModelSettings:
    """ML model configuration settings"""
    scene_model: str
    object_model: str
    audio_model: str
    alignment_model: str
    batch_size: int
    device: Optional[str] = None

@dataclass
class ProcessingSettings:
    """Video processing configuration"""
    max_video_size: int
    sampling_rate: int
    min_segment_frames: int
    confidence_threshold: float

@dataclass
class SecuritySettings:
    """Security configuration"""
    api_key: str
    secret_key: str
    allowed_origins: list[str]
    max_requests_per_minute: int

@dataclass
class Config:
    """Main configuration class"""
    environment: Environment
    version: str
    models: ModelSettings
    processing: ProcessingSettings
    security: SecuritySettings
    cache_dir: Path
    log_level: str
    enable_gpu: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create Config instance from dictionary"""
        models = ModelSettings(**data.pop('models'))
        processing = ProcessingSettings(**data.pop('processing'))
        security = SecuritySettings(**data.pop('security'))
        data['environment'] = Environment(data['environment'])
        data['cache_dir'] = Path(data['cache_dir'])
        
        return cls(
            models=models,
            processing=processing,
            security=security,
            **data
        )

# ... rest of the implementation as provided ... 
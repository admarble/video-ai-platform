from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json
import os
from dataclasses import dataclass, asdict
import logging
from enum import Enum
import secrets
import copy
from typing import TypeVar, Type

class Environment(Enum):
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"

@dataclass
class ModelSettings:
    """Settings for ML models"""
    scene_model: str
    object_model: str
    audio_model: str
    alignment_model: str
    batch_size: int
    device: Optional[str] = None

@dataclass
class ProcessingSettings:
    """Video processing settings"""
    max_video_size: int
    sampling_rate: int
    min_segment_frames: int
    confidence_threshold: float

@dataclass
class SecuritySettings:
    """Security-related settings"""
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

class ConfigurationError(Exception):
    """Custom exception for configuration errors"""
    pass

class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self):
        self._config: Optional[Config] = None
        self._config_path: Optional[Path] = None
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config_path: str, environment: Environment) -> Config:
        """Load configuration for specified environment"""
        self._config_path = Path(config_path)
        base_config_path = self._config_path.parent / "config.base.yaml"
        
        if not base_config_path.exists():
            raise ConfigurationError(f"Base configuration file not found: {base_config_path}")
            
        try:
            # Load base configuration
            with open(base_config_path) as f:
                config_data = yaml.safe_load(f)
                
            # Load environment-specific configuration
            env_config_path = self._config_path.parent / f"config.{environment.value}.yaml"
            if env_config_path.exists():
                with open(env_config_path) as f:
                    env_config = yaml.safe_load(f)
                    config_data = self._deep_merge(config_data, env_config)
                    
            # Load secrets from environment variables
            config_data = self._load_secrets(config_data)
            
            # Create config instance
            self._config = Config.from_dict(config_data)
            
            # Validate configuration
            self._validate_config(self._config)
            
            return self._config
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing configuration: {str(e)}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {str(e)}")
            
    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Recursively merge two dictionaries"""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
                
        return result
        
    def _load_secrets(self, config: dict) -> dict:
        """Load secrets from environment variables"""
        result = copy.deepcopy(config)
        
        def replace_env_vars(obj: Any) -> Any:
            if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
                env_var = obj[2:-1]
                if env_var not in os.environ:
                    raise ConfigurationError(f"Environment variable not found: {env_var}")
                return os.environ[env_var]
            elif isinstance(obj, dict):
                return {k: replace_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_env_vars(v) for v in obj]
            return obj
            
        return replace_env_vars(result)
        
    def _validate_config(self, config: Config) -> None:
        """Validate configuration values"""
        if config.models.batch_size <= 0:
            raise ConfigurationError("Batch size must be positive")
            
        if config.processing.sampling_rate <= 0:
            raise ConfigurationError("Sampling rate must be positive")
        if config.processing.confidence_threshold < 0 or config.processing.confidence_threshold > 1:
            raise ConfigurationError("Confidence threshold must be between 0 and 1")
            
        if config.security.max_requests_per_minute <= 0:
            raise ConfigurationError("Max requests per minute must be positive")
            
        if not config.cache_dir.exists():
            try:
                config.cache_dir.mkdir(parents=True)
            except Exception as e:
                raise ConfigurationError(f"Cannot create cache directory: {str(e)}")
                
    def get_config(self) -> Config:
        """Get current configuration"""
        if self._config is None:
            raise ConfigurationError("Configuration not loaded")
        return self._config
        
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        if self._config is None:
            raise ConfigurationError("Configuration not loaded")
            
        config_dict = asdict(self._config)
        config_dict = self._deep_merge(config_dict, updates)
        
        new_config = Config.from_dict(config_dict)
        self._validate_config(new_config)
        self._config = new_config
        
        if self._config_path:
            with open(self._config_path, 'w') as f:
                yaml.dump(asdict(self._config), f)
                
    def generate_default_config(self, output_path: str) -> None:
        """Generate default configuration file"""
        default_config = {
            "environment": "dev",
            "version": "1.0.0",
            "models": {
                "scene_model": "MCG-NJU/videomae-base-finetuned-kinetics",
                "object_model": "facebook/detr-resnet-50",
                "audio_model": "facebook/wav2vec2-base-960h",
                "alignment_model": "openai/clip-vit-base-patch32",
                "batch_size": 32,
                "device": None
            },
            "processing": {
                "max_video_size": 1_000_000_000,  # 1GB
                "sampling_rate": 1,
                "min_segment_frames": 30,
                "confidence_threshold": 0.7
            },
            "security": {
                "api_key": "${API_KEY}",
                "secret_key": secrets.token_hex(32),
                "allowed_origins": ["http://localhost:3000"],
                "max_requests_per_minute": 60
            },
            "cache_dir": str(Path.home() / ".video_ai_cache"),
            "log_level": "INFO",
            "enable_gpu": True
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(default_config, f, sort_keys=False) 
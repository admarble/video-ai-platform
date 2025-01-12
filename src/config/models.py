from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional
from datetime import datetime

class Environment(Enum):
    """Environment types for configuration"""
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"

@dataclass
class ConfigVersion:
    """Configuration version information"""
    version: str
    timestamp: float
    hash: str
    changes: List[str] = field(default_factory=list)

@dataclass
class SecretsConfig:
    """Secrets management configuration"""
    encryption_key: str
    key_rotation_days: int
    vault_path: str
    last_rotation: float

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
    allowed_origins: List[str]
    max_requests_per_minute: int

@dataclass
class Config:
    """Main configuration class with versioning"""
    environment: Environment
    version: str
    models: ModelSettings
    processing: ProcessingSettings
    security: SecuritySettings
    secrets: SecretsConfig
    cache_dir: Path
    log_level: str
    enable_gpu: bool = True
    version_info: Optional[ConfigVersion] = None 
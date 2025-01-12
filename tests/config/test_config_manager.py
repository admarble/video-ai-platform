import pytest
from pathlib import Path
import yaml
import tempfile
import shutil
from src.config import (
    ConfigManager,
    ConfigurationError,
    Environment,
    Config
)

@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for config files"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def base_config(temp_config_dir):
    """Create a base config file for testing"""
    config = {
        'environment': 'dev',
        'version': '1.0.0',
        'models': {
            'scene_model': 'model1',
            'object_model': 'model2',
            'audio_model': 'model3',
            'alignment_model': 'model4',
            'batch_size': 32
        },
        'processing': {
            'max_video_size': 1920,
            'sampling_rate': 30,
            'min_segment_frames': 60,
            'confidence_threshold': 0.8
        },
        'security': {
            'api_key': 'secret://api_key',
            'secret_key': 'secret://secret_key',
            'allowed_origins': ['localhost'],
            'max_requests_per_minute': 100
        },
        'secrets': {
            'encryption_key': 'test-key',
            'key_rotation_days': 30,
            'vault_path': str(temp_config_dir / 'secrets'),
            'last_rotation': 0
        },
        'cache_dir': str(temp_config_dir / 'cache'),
        'log_level': 'INFO',
        'enable_gpu': True
    }
    
    config_path = temp_config_dir / 'config.base.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path

@pytest.fixture
def config_manager():
    """Create a ConfigManager instance"""
    return ConfigManager()

def test_load_config(config_manager, base_config):
    """Test basic config loading"""
    config = config_manager.load_config(str(base_config), Environment.DEV)
    assert isinstance(config, Config)
    assert config.environment == Environment.DEV
    assert config.version == '1.0.0'
    assert config.models.batch_size == 32

def test_environment_specific_config(config_manager, temp_config_dir, base_config):
    """Test loading environment-specific config"""
    env_config = {
        'models': {
            'batch_size': 64
        },
        'processing': {
            'sampling_rate': 60
        }
    }
    
    env_config_path = temp_config_dir / 'config.prod.yaml'
    with open(env_config_path, 'w') as f:
        yaml.dump(env_config, f)
    
    config = config_manager.load_config(str(base_config), Environment.PROD)
    assert config.models.batch_size == 64
    assert config.processing.sampling_rate == 60

def test_secret_management(config_manager, base_config):
    """Test secret storage and retrieval"""
    config_manager.load_config(str(base_config), Environment.DEV)
    
    # Store some secrets
    config_manager.store_secret('api_key', 'test-api-key')
    config_manager.store_secret('secret_key', 'test-secret-key')
    
    # Reload config to test secret injection
    config = config_manager.load_config(str(base_config), Environment.DEV)
    assert config.security.api_key == 'test-api-key'
    assert config.security.secret_key == 'test-secret-key'

def test_config_validation(config_manager, temp_config_dir):
    """Test configuration validation"""
    invalid_config = {
        'environment': 'invalid',
        'version': '1.0.0'
    }
    
    config_path = temp_config_dir / 'config.invalid.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(invalid_config, f)
    
    with pytest.raises(ConfigurationError):
        config_manager.load_config(str(config_path), Environment.DEV)

def test_observer_pattern(config_manager, base_config):
    """Test configuration change notifications"""
    called = {'value': False}
    
    def callback(new_config):
        called['value'] = True
    
    config_manager.add_observer(callback)
    config_manager.load_config(str(base_config), Environment.DEV)
    
    # Simulate config change
    config_manager.reload_config()
    assert called['value'] is True

def test_version_history(config_manager, base_config):
    """Test version history tracking"""
    config_manager.load_config(str(base_config), Environment.DEV)
    history = config_manager.get_version_history()
    
    assert len(history) == 1
    assert history[0].version == '1.0.0'
import pytest
import tempfile
from pathlib import Path
import time
import yaml
from src.config.watcher import ConfigObserver

class MockConfigManager:
    def __init__(self):
        self.reload_called = False
        
    def reload_config(self):
        self.reload_called = True

@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for config files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def config_observer(temp_config_dir):
    """Create a ConfigObserver instance"""
    manager = MockConfigManager()
    observer = ConfigObserver(str(temp_config_dir), manager)
    observer.start()
    yield observer, manager
    observer.stop()

def test_file_change_detection(config_observer, temp_config_dir):
    """Test that file changes trigger config reload"""
    observer, manager = config_observer
    
    # Create a config file
    config_path = temp_config_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump({'test': 'value'}, f)
    
    # Wait for file system events
    time.sleep(1)
    
    # Modify the file
    with open(config_path, 'w') as f:
        yaml.dump({'test': 'new_value'}, f)
    
    # Wait for notification
    time.sleep(1)
    
    assert manager.reload_called 
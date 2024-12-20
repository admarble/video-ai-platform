import pytest
import tempfile
from pathlib import Path
from src.config.secrets import SecretsManager
from cryptography.fernet import Fernet

@pytest.fixture
def temp_vault():
    """Create a temporary directory for secrets vault"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def secrets_manager(temp_vault):
    """Create a SecretsManager instance"""
    key = Fernet.generate_key().decode()
    return SecretsManager(str(temp_vault), key)

def test_store_and_retrieve_secret(secrets_manager):
    """Test basic secret storage and retrieval"""
    secrets_manager.store_secret('test_key', 'test_value')
    value = secrets_manager.get_secret('test_key')
    assert value == 'test_value'

def test_key_rotation(secrets_manager):
    """Test encryption key rotation"""
    # Store a secret with original key
    secrets_manager.store_secret('test_key', 'test_value')
    
    # Rotate to new key
    new_key = Fernet.generate_key().decode()
    secrets_manager.rotate_encryption_key(new_key)
    
    # Verify secret can still be retrieved
    value = secrets_manager.get_secret('test_key')
    assert value == 'test_value'

def test_nonexistent_secret(secrets_manager):
    """Test retrieving nonexistent secret"""
    value = secrets_manager.get_secret('nonexistent')
    assert value is None 
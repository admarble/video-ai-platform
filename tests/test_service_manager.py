import pytest
from pathlib import Path
import torch

from src.services.ai.service_manager import ServiceManager, ModelConfig
from src.exceptions import ServiceInitializationError

def test_service_manager_initialization():
    config = ModelConfig(
        device="cpu",
        batch_size=2,
        num_workers=1
    )
    manager = ServiceManager(config)
    assert manager.device == "cpu"
    assert manager.config.batch_size == 2
    assert manager.config.num_workers == 1

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_service_manager_gpu():
    manager = ServiceManager()
    assert manager.device == "cuda"

def test_invalid_service():
    manager = ServiceManager()
    with pytest.raises(ValueError):
        manager._initialize_service("invalid_service") 
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import logging
import asyncio

logger = logging.getLogger(__name__)

class CircuitBreakerStorageError(Exception):
    """Base exception for circuit breaker storage errors."""
    pass

class StorageBackend(ABC):
    """Abstract base class for circuit breaker storage backends."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._connection = None
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the storage backend."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the storage backend."""
        pass
    
    @abstractmethod
    async def get_state(self, key: str) -> Optional[Dict[str, Any]]:
        """Get state data as a dictionary."""
        pass
    
    @abstractmethod
    async def set_state(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Set state data from a dictionary with optional TTL."""
        pass
    
    @abstractmethod
    async def acquire_lock(self, lock_key: str, lock_value: str, ttl: int) -> bool:
        """Acquire a distributed lock with value and TTL."""
        pass
    
    @abstractmethod
    async def release_lock(self, lock_key: str, lock_value: str) -> None:
        """Release a distributed lock with value."""
        pass
    
    @abstractmethod
    async def register_instance(self, instance_id: str, ttl: int) -> None:
        """Register a circuit breaker instance with TTL."""
        pass
    
    @abstractmethod
    async def unregister_instance(self, instance_id: str) -> None:
        """Unregister a circuit breaker instance."""
        pass
    
    @abstractmethod
    async def get_instances(self, prefix: str) -> List[str]:
        """Get list of registered instances."""
        pass
    
    def _format_key(self, name: str, suffix: str) -> str:
        """Format a storage key with consistent prefix."""
        return f"circuit_breaker:{name}:{suffix}"
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect() 
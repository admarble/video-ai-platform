import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass
from enum import Enum

from .storage import StorageBackend, CircuitBreakerStorageError
from .redis_storage import RedisStorageBackend
from .s3_storage import S3StorageBackend
from .elasticache_storage import ElastiCacheStorageBackend

logger = logging.getLogger(__name__)

@dataclass
class HealthCheckConfig:
    check_interval: int = 5
    timeout: int = 3
    threshold: int = 3
    recovery_threshold: int = 2

class FailoverStrategy(Enum):
    LATENCY_BASED = "latency_based"
    ROUND_ROBIN = "round_robin"
    PRIORITY = "priority"

class FailoverStorageBackend(StorageBackend):
    """Storage backend that implements failover between multiple backends."""
    
    def __init__(self, storage_configs: List[Dict[str, Any]], 
                 strategy: FailoverStrategy = FailoverStrategy.LATENCY_BASED,
                 health_config: Optional[HealthCheckConfig] = None):
        self.storage_configs = storage_configs
        self.strategy = strategy
        self.health_config = health_config or HealthCheckConfig()
        self.backends: List[StorageBackend] = []
        self.health_status: Dict[str, Dict[str, Any]] = {}
        self._health_check_task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger(__name__)
        
    async def connect(self) -> None:
        """Connect to all storage backends."""
        for config in self.storage_configs:
            backend = self._create_backend(config)
            try:
                await backend.connect()
                self.backends.append(backend)
                self.health_status[config['name']] = {
                    'healthy': True,
                    'failures': 0,
                    'successes': 0,
                    'latency': 0,
                    'last_check': time.time(),
                    'last_failure': None
                }
            except Exception as e:
                self.logger.error(f"Failed to connect to backend {config['name']}: {str(e)}")
        
        if not self.backends:
            raise CircuitBreakerStorageError("No storage backends available")
            
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
    def _create_backend(self, config: Dict[str, Any]) -> StorageBackend:
        """Create a storage backend instance based on config."""
        backend_type = config['type'].lower()
        backend_config = config['config']
        
        if backend_type == 'redis':
            return RedisStorageBackend(backend_config)
        elif backend_type == 'elasticache':
            return ElastiCacheStorageBackend(backend_config)
        elif backend_type == 's3':
            return S3StorageBackend(backend_config)
        else:
            raise ValueError(f"Unsupported storage type: {backend_type}")
            
    async def disconnect(self) -> None:
        """Disconnect from all storage backends."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
                
        for backend in self.backends:
            try:
                await backend.disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting backend: {str(e)}")
                
    async def _health_check_loop(self) -> None:
        """Background task to check health of all backends."""
        while True:
            try:
                await asyncio.sleep(self.health_config.check_interval)
                for backend, config in zip(self.backends, self.storage_configs):
                    name = config['name']
                    status = self.health_status[name]
                    
                    try:
                        start_time = time.time()
                        await asyncio.wait_for(
                            self._check_backend_health(backend),
                            timeout=self.health_config.timeout
                        )
                        latency = time.time() - start_time
                        
                        status['latency'] = latency
                        status['successes'] += 1
                        status['failures'] = 0
                        
                        if not status['healthy'] and status['successes'] >= self.health_config.recovery_threshold:
                            status['healthy'] = True
                            self.logger.info(f"Backend {name} recovered")
                            
                    except Exception as e:
                        status['failures'] += 1
                        status['successes'] = 0
                        status['last_failure'] = time.time()
                        
                        if status['healthy'] and status['failures'] >= self.health_config.threshold:
                            status['healthy'] = False
                            self.logger.warning(f"Backend {name} marked unhealthy: {str(e)}")
                            
                    status['last_check'] = time.time()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {str(e)}")
                await asyncio.sleep(1)
                
    async def _check_backend_health(self, backend: StorageBackend) -> None:
        """Check health of a single backend."""
        # Try to write and read a test value
        test_key = "health_check"
        test_value = {"timestamp": time.time()}
        
        await backend.set_state(test_key, test_value)
        result = await backend.get_state(test_key)
        
        if not result or result.get('timestamp') != test_value['timestamp']:
            raise CircuitBreakerStorageError("Health check failed: inconsistent data")
            
    def _select_backend(self) -> StorageBackend:
        """Select a backend based on the chosen strategy."""
        available = [(name, status, backend) 
                    for (name, status), backend in zip(self.health_status.items(), self.backends)
                    if status['healthy']]
                    
        if not available:
            raise CircuitBreakerStorageError("No healthy backends available")
            
        if self.strategy == FailoverStrategy.LATENCY_BASED:
            return min(available, key=lambda x: x[1]['latency'])[2]
        elif self.strategy == FailoverStrategy.ROUND_ROBIN:
            # Simple round-robin using the first healthy backend
            return available[0][2]
        else:  # PRIORITY
            # Return the first healthy backend in the original order
            return available[0][2]
            
    async def get_state(self, key: str) -> Optional[Dict[str, Any]]:
        """Get state from the selected backend."""
        backend = self._select_backend()
        return await backend.get_state(key)
        
    async def set_state(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Set state in the selected backend."""
        backend = self._select_backend()
        await backend.set_state(key, value, ttl)
        
    async def acquire_lock(self, lock_key: str, lock_value: str, ttl: int) -> bool:
        """Acquire lock from the selected backend."""
        backend = self._select_backend()
        return await backend.acquire_lock(lock_key, lock_value, ttl)
        
    async def release_lock(self, lock_key: str, lock_value: str) -> None:
        """Release lock from the selected backend."""
        backend = self._select_backend()
        await backend.release_lock(lock_key, lock_value)
        
    async def register_instance(self, instance_id: str, ttl: int) -> None:
        """Register instance with the selected backend."""
        backend = self._select_backend()
        await backend.register_instance(instance_id, ttl)
        
    async def unregister_instance(self, instance_id: str) -> None:
        """Unregister instance from the selected backend."""
        backend = self._select_backend()
        await backend.unregister_instance(instance_id)
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all backends."""
        return {
            name: {
                'healthy': status['healthy'],
                'latency': status['latency'],
                'last_check': status['last_check'],
                'last_failure': status['last_failure']
            }
            for name, status in self.health_status.items()
        } 
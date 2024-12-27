import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional, Callable, TypeVar, Awaitable, List
from dataclasses import dataclass
from functools import wraps
import uuid

from .storage import StorageBackend
from .redis_storage import RedisStorageBackend
from .etcd_storage import EtcdStorageBackend
from .consul_storage import ConsulStorageBackend
from .zookeeper_storage import ZooKeeperStorageBackend
from .dynamodb_storage import DynamoDBStorageBackend
from .s3_storage import S3StorageBackend
from .elasticache_storage import ElastiCacheStorageBackend

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int
    reset_timeout: int
    sync_interval: int
    state_ttl: Optional[int] = None
    corruption_threshold: Optional[int] = None
    model_error_threshold: Optional[int] = None

class CircuitBreakerState:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreakerError(Exception):
    """Exception raised when the circuit breaker prevents an operation."""
    pass

class DistributedCircuitBreaker:
    def __init__(
        self,
        name: str,
        storage: StorageBackend,
        config: CircuitBreakerConfig
    ):
        self.name = name
        self.storage = storage
        self.config = config
        self.instance_id = str(uuid.uuid4())
        self._sync_task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger(__name__)
    
    async def start(self) -> None:
        """Initialize the circuit breaker and start background sync."""
        await self.storage.connect()
        await self.storage.register_instance(self.instance_id, self.config.sync_interval * 2)
        
        # Initialize state if not exists
        state = await self.storage.get_state(f"{self.name}:state")
        if not state:
            await self.storage.set_state(
                f"{self.name}:state",
                {
                    "state": CircuitBreakerState.CLOSED,
                    "failures": 0,
                    "corruption_count": 0,
                    "model_error_count": 0,
                    "success_count": 0,
                    "last_failure": 0,
                    "last_success": 0,
                    "last_updated": int(time.time()),
                    "instance_id": self.instance_id
                },
                ttl=self.config.state_ttl
            )
        
        self._sync_task = asyncio.create_task(self._sync_loop())
    
    async def cleanup(self) -> None:
        """Clean up resources and stop background sync."""
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        
        await self.storage.unregister_instance(self.instance_id)
        await self.storage.disconnect()
    
    async def get_instances(self) -> List[str]:
        """Get list of registered circuit breaker instances."""
        return await self.storage.get_instances(self.name)
    
    async def _sync_loop(self) -> None:
        """Background task to sync circuit breaker state."""
        while True:
            try:
                await asyncio.sleep(self.config.sync_interval)
                await self.storage.register_instance(
                    self.instance_id,
                    self.config.sync_interval * 2
                )
                
                # Check if we should reset to half-open state
                state_data = await self.storage.get_state(f"{self.name}:state")
                if state_data and state_data["state"] == CircuitBreakerState.OPEN:
                    if (time.time() - state_data["last_failure"]) >= self.config.reset_timeout:
                        await self._try_transition_to_half_open()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in circuit breaker sync loop: {str(e)}")
                await asyncio.sleep(1)  # Avoid tight loop on persistent errors
    
    async def _try_transition_to_half_open(self) -> None:
        """Attempt to transition from OPEN to HALF_OPEN state."""
        lock_acquired = await self.storage.acquire_lock(
            f"{self.name}:transition_lock",
            self.instance_id,
            self.config.sync_interval
        )
        
        if lock_acquired:
            try:
                state_data = await self.storage.get_state(f"{self.name}:state")
                if state_data["state"] == CircuitBreakerState.OPEN:
                    state_data["state"] = CircuitBreakerState.HALF_OPEN
                    state_data["last_updated"] = int(time.time())
                    state_data["instance_id"] = self.instance_id
                    await self.storage.set_state(
                        f"{self.name}:state",
                        state_data,
                        ttl=self.config.state_ttl
                    )
            finally:
                await self.storage.release_lock(
                    f"{self.name}:transition_lock",
                    self.instance_id
                )
    
    async def _record_success(self) -> None:
        """Record a successful execution."""
        lock_acquired = await self.storage.acquire_lock(
            f"{self.name}:state_lock",
            self.instance_id,
            self.config.sync_interval
        )
        
        if lock_acquired:
            try:
                state_data = await self.storage.get_state(f"{self.name}:state")
                current_time = int(time.time())
                state_data["last_success"] = current_time
                state_data["success_count"] += 1
                state_data["last_updated"] = current_time
                state_data["instance_id"] = self.instance_id
                
                if state_data["state"] in [CircuitBreakerState.HALF_OPEN, CircuitBreakerState.OPEN]:
                    state_data["state"] = CircuitBreakerState.CLOSED
                    state_data["failures"] = 0
                    state_data["corruption_count"] = 0
                    state_data["model_error_count"] = 0
                
                await self.storage.set_state(
                    f"{self.name}:state",
                    state_data,
                    ttl=self.config.state_ttl
                )
            finally:
                await self.storage.release_lock(
                    f"{self.name}:state_lock",
                    self.instance_id
                )
    
    async def _record_failure(self, error_type: Optional[str] = None) -> None:
        """Record a failed execution with optional error type."""
        lock_acquired = await self.storage.acquire_lock(
            f"{self.name}:state_lock",
            self.instance_id,
            self.config.sync_interval
        )
        
        if lock_acquired:
            try:
                state_data = await self.storage.get_state(f"{self.name}:state")
                current_time = int(time.time())
                state_data["last_failure"] = current_time
                state_data["last_updated"] = current_time
                state_data["instance_id"] = self.instance_id
                
                # Update specific error counters
                if error_type == "corruption":
                    state_data["corruption_count"] += 1
                elif error_type == "model_error":
                    state_data["model_error_count"] += 1
                else:
                    state_data["failures"] += 1
                
                # Check thresholds
                should_open = False
                if (
                    state_data["state"] == CircuitBreakerState.CLOSED
                    and (
                        state_data["failures"] >= self.config.failure_threshold
                        or (self.config.corruption_threshold and state_data["corruption_count"] >= self.config.corruption_threshold)
                        or (self.config.model_error_threshold and state_data["model_error_count"] >= self.config.model_error_threshold)
                    )
                ):
                    should_open = True
                elif state_data["state"] == CircuitBreakerState.HALF_OPEN:
                    should_open = True
                
                if should_open:
                    state_data["state"] = CircuitBreakerState.OPEN
                
                await self.storage.set_state(
                    f"{self.name}:state",
                    state_data,
                    ttl=self.config.state_ttl
                )
            finally:
                await self.storage.release_lock(
                    f"{self.name}:state_lock",
                    self.instance_id
                )
    
    def __call__(self, func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        """Decorator for protecting async functions with the circuit breaker."""
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            state_data = await self.storage.get_state(f"{self.name}:state")
            current_state = state_data["state"]
            
            if current_state == CircuitBreakerState.OPEN:
                raise CircuitBreakerError(
                    f"Circuit is OPEN (failures={state_data['failures']}, "
                    f"corruption={state_data['corruption_count']}, "
                    f"model_errors={state_data['model_error_count']})"
                )
            
            try:
                result = await func(*args, **kwargs)
                await self._record_success()
                return result
            except Exception as e:
                error_type = None
                if "corruption" in str(e).lower():
                    error_type = "corruption"
                elif "model" in str(e).lower():
                    error_type = "model_error"
                
                await self._record_failure(error_type)
                raise CircuitBreakerError(f"Operation failed: {str(e)}") from e
        
        return wrapper

async def create_circuit_breaker_with_storage(
    name: str,
    storage_type: str,
    storage_config: Dict[str, Any],
    circuit_config: CircuitBreakerConfig
) -> DistributedCircuitBreaker:
    """Factory function to create a circuit breaker with the specified storage backend."""
    storage_backends = {
        "redis": RedisStorageBackend,
        "etcd": EtcdStorageBackend,
        "consul": ConsulStorageBackend,
        "zookeeper": ZooKeeperStorageBackend,
        "dynamodb": DynamoDBStorageBackend,
        "s3": S3StorageBackend,
        "elasticache": ElastiCacheStorageBackend
    }
    
    if storage_type not in storage_backends:
        raise ValueError(f"Unsupported storage type: {storage_type}")
    
    storage = storage_backends[storage_type](storage_config)
    circuit = DistributedCircuitBreaker(name, storage, circuit_config)
    await circuit.start()
    return circuit 
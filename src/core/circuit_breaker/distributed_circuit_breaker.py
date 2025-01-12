from typing import Dict, Any, Optional, List, Union, Callable
import json
import time
import asyncio
import logging
from datetime import datetime
import aioredis
from dataclasses import dataclass
from src.core.circuit_breaker import CircuitBreaker, CircuitState, CircuitConfig, CircuitOpenError

@dataclass
class DistributedCircuitConfig(CircuitConfig):
    """Configuration for distributed circuit breaker"""
    sync_interval: int = 5           # State sync interval in seconds
    state_ttl: int = 300            # State TTL in Redis (seconds)
    quorum_percent: float = 0.5     # Percent of instances needed for state change
    lock_timeout: int = 10          # Lock timeout in seconds

class RedisCircuitBreaker(CircuitBreaker):
    """Circuit breaker with Redis-based distributed state"""
    
    def __init__(
        self,
        name: str,
        config: DistributedCircuitConfig,
        redis_url: str,
        instance_id: Optional[str] = None
    ):
        super().__init__(name, config)
        self.config: DistributedCircuitConfig = config
        self.redis: aioredis.Redis = None
        self.redis_url = redis_url
        self.instance_id = instance_id or f"{name}-{time.time()}"
        self.state_key = f"circuit:{name}:state"
        self.instances_key = f"circuit:{name}:instances"
        self.lock_key = f"circuit:{name}:lock"
        self._sync_task = None

    async def init(self):
        """Initialize Redis connection and start sync"""
        self.redis = await aioredis.from_url(self.redis_url)
        await self._register_instance()
        self._sync_task = asyncio.create_task(self._sync_state_loop())

    async def _register_instance(self):
        """Register this instance in Redis"""
        await self.redis.sadd(self.instances_key, self.instance_id)
        await self.redis.expire(self.instances_key, self.config.state_ttl)

    async def _sync_state_loop(self):
        """Background task to sync state with Redis"""
        while True:
            try:
                await self._sync_state()
                await asyncio.sleep(self.config.sync_interval)
            except Exception as e:
                self.logger.error(f"Error syncing state: {str(e)}")
                await asyncio.sleep(1)

    async def _acquire_lock(self) -> bool:
        """Acquire distributed lock for state changes"""
        locked = await self.redis.set(
            self.lock_key,
            self.instance_id,
            ex=self.config.lock_timeout,
            nx=True
        )
        return bool(locked)

    async def _release_lock(self):
        """Release distributed lock"""
        # Only release if we hold the lock
        if await self.redis.get(self.lock_key) == self.instance_id:
            await self.redis.delete(self.lock_key)

    async def _sync_state(self):
        """Sync local state with Redis"""
        try:
            # Get current distributed state
            state_data = await self.redis.get(self.state_key)
            if state_data:
                distributed_state = json.loads(state_data)
                await self._update_local_state(distributed_state)
            else:
                # No distributed state - publish local state
                await self._publish_state()

        except Exception as e:
            self.logger.error(f"Error syncing state: {str(e)}")

    async def _update_local_state(self, distributed_state: Dict[str, Any]):
        """Update local state from distributed state"""
        with self.lock:
            # Check if state should change
            new_state = CircuitState(distributed_state['state'])
            if new_state != self.state:
                if new_state == CircuitState.OPEN:
                    await self._handle_remote_open(distributed_state)
                elif new_state == CircuitState.HALF_OPEN:
                    await self._handle_remote_half_open(distributed_state)
                elif new_state == CircuitState.CLOSED:
                    await self._handle_remote_close(distributed_state)

            # Update counters
            self.failure_timestamps = distributed_state.get('failure_timestamps', [])
            self.success_count = distributed_state.get('success_count', 0)
            self.half_open_count = distributed_state.get('half_open_count', 0)

    async def _publish_state(self):
        """Publish local state to Redis"""
        state_data = {
            'state': self.state.value,
            'failure_timestamps': self.failure_timestamps,
            'success_count': self.success_count,
            'half_open_count': self.half_open_count,
            'last_state_change': self.last_state_change,
            'instance_id': self.instance_id
        }
        
        await self.redis.set(
            self.state_key,
            json.dumps(state_data),
            ex=self.config.state_ttl
        )

    async def _get_active_instances(self) -> List[str]:
        """Get list of active instances"""
        return [i.decode() for i in await self.redis.smembers(self.instances_key)]

    async def _check_quorum(self) -> bool:
        """Check if enough instances agree on state change"""
        active_instances = await self._get_active_instances()
        quorum_size = max(1, int(len(active_instances) * self.config.quorum_percent))
        state_data = await self.redis.get(self.state_key)
        
        if not state_data:
            return False
            
        distributed_state = json.loads(state_data)
        agreeing_instances = 0
        
        for instance in active_instances:
            instance_state = await self.redis.get(f"circuit:{self.name}:instance:{instance}")
            if instance_state:
                instance_data = json.loads(instance_state)
                if instance_data['state'] == distributed_state['state']:
                    agreeing_instances += 1
                    
        return agreeing_instances >= quorum_size

    async def _handle_remote_open(self, state_data: Dict[str, Any]):
        """Handle circuit being opened by another instance"""
        self.state = CircuitState.OPEN
        self.last_state_change = state_data.get('last_state_change', time.time())
        self.logger.warning(
            f"Circuit opened by instance {state_data.get('instance_id')}"
        )

    async def _handle_remote_half_open(self, state_data: Dict[str, Any]):
        """Handle circuit being half-opened by another instance"""
        self.state = CircuitState.HALF_OPEN
        self.last_state_change = state_data.get('last_state_change', time.time())
        self.success_count = 0
        self.half_open_count = 0
        self.logger.info(
            f"Circuit half-opened by instance {state_data.get('instance_id')}"
        )

    async def _handle_remote_close(self, state_data: Dict[str, Any]):
        """Handle circuit being closed by another instance"""
        self.state = CircuitState.CLOSED
        self.last_state_change = state_data.get('last_state_change', time.time())
        self.failure_timestamps = []
        self.success_count = 0
        self.half_open_count = 0
        self.logger.info(
            f"Circuit closed by instance {state_data.get('instance_id')}"
        )

    async def can_execute(self) -> bool:
        """Check if request can be executed (distributed version)"""
        with self.lock:
            current_time = time.time()

            # Clean up old failure timestamps
            self._clean_failure_window(current_time)

            if self.state == CircuitState.OPEN:
                if current_time - self.last_state_change >= self.config.reset_timeout:
                    # Try to acquire lock for state change
                    if await self._acquire_lock():
                        try:
                            if await self._check_quorum():
                                await self._transition_to_half_open()
                                return True
                        finally:
                            await self._release_lock()
                return False
                
            if self.state == CircuitState.HALF_OPEN:
                return self.half_open_count < self.config.half_open_limit
                
            return True

    async def record_success(self):
        """Record successful execution (distributed version)"""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    # Try to acquire lock for state change
                    if await self._acquire_lock():
                        try:
                            if await self._check_quorum():
                                await self._transition_to_closed()
                            else:
                                self.success_count -= 1
                        finally:
                            await self._release_lock()
                else:
                    self.half_open_count += 1
                await self._publish_state()

    async def record_failure(self):
        """Record failed execution (distributed version)"""
        with self.lock:
            current_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                if await self._acquire_lock():
                    try:
                        await self._transition_to_open(current_time)
                    finally:
                        await self._release_lock()
                return

            # Add failure timestamp
            self.failure_timestamps.append(current_time)
            self._clean_failure_window(current_time)

            # Check if threshold is exceeded
            if len(self.failure_timestamps) >= self.config.failure_threshold:
                if await self._acquire_lock():
                    try:
                        if await self._check_quorum():
                            await self._transition_to_open(current_time)
                    finally:
                        await self._release_lock()
                        
            await self._publish_state()

    async def cleanup(self):
        """Cleanup resources"""
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
                
        if self.redis:
            await self.redis.srem(self.instances_key, self.instance_id)
            await self.redis.close()

async def create_distributed_circuit_breaker(
    name: str,
    redis_url: str,
    config: Optional[DistributedCircuitConfig] = None
) -> RedisCircuitBreaker:
    """Create distributed circuit breaker instance"""
    circuit = RedisCircuitBreaker(
        name=name,
        config=config or DistributedCircuitConfig(),
        redis_url=redis_url
    )
    await circuit.init()
    return circuit

def distributed_circuit_breaker(
    circuit: RedisCircuitBreaker,
    fallback: Optional[Callable] = None
):
    """Decorator for using distributed circuit breaker"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            if not await circuit.can_execute():
                if fallback:
                    return await fallback(*args, **kwargs)
                raise CircuitOpenError(f"Circuit {circuit.name} is open")
                
            try:
                result = await func(*args, **kwargs)
                await circuit.record_success()
                return result
            except Exception as e:
                await circuit.record_failure()
                raise
                
        return wrapper
    return decorator 
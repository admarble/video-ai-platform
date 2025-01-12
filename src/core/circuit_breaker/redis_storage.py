import aioredis
from typing import Any, Dict, Optional, List
import json
import time
from .storage import StorageBackend, CircuitBreakerStorageError

class RedisStorageBackend(StorageBackend):
    """Redis implementation of circuit breaker storage backend."""
    
    async def connect(self) -> None:
        try:
            self._connection = await aioredis.from_url(
                self.config['url'],
                encoding='utf-8',
                decode_responses=True
            )
        except Exception as e:
            raise CircuitBreakerStorageError(f"Failed to connect to Redis: {str(e)}")
    
    async def disconnect(self) -> None:
        if self._connection:
            await self._connection.close()
    
    async def get_state(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            data = await self._connection.get(self._format_key(key, "state"))
            return json.loads(data) if data else None
        except Exception as e:
            self.logger.error(f"Failed to get state from Redis: {str(e)}")
            return None
    
    async def set_state(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> None:
        try:
            formatted_key = self._format_key(key, "state")
            data = json.dumps(value)
            if ttl:
                await self._connection.setex(formatted_key, ttl, data)
            else:
                await self._connection.set(formatted_key, data)
        except Exception as e:
            self.logger.error(f"Failed to set state in Redis: {str(e)}")
    
    async def acquire_lock(self, lock_key: str, lock_value: str, ttl: int) -> bool:
        try:
            formatted_key = self._format_key(lock_key, "lock")
            return await self._connection.set(formatted_key, lock_value, nx=True, ex=ttl)
        except Exception as e:
            self.logger.error(f"Failed to acquire lock in Redis: {str(e)}")
            return False
    
    async def release_lock(self, lock_key: str, lock_value: str) -> None:
        try:
            formatted_key = self._format_key(lock_key, "lock")
            # Only release if we hold the lock
            current = await self._connection.get(formatted_key)
            if current == lock_value:
                await self._connection.delete(formatted_key)
        except Exception as e:
            self.logger.error(f"Failed to release lock in Redis: {str(e)}")
    
    async def register_instance(self, instance_id: str, ttl: int) -> None:
        try:
            instance_key = self._format_key(instance_id, "instance")
            instance_data = json.dumps({
                "instance_id": instance_id,
                "registered_at": int(time.time()),
                "last_heartbeat": int(time.time())
            })
            await self._connection.setex(instance_key, ttl, instance_data)
            
            # Add to instance set
            instances_key = self._format_key(instance_id.split(":")[0], "instances")
            await self._connection.sadd(instances_key, instance_id)
            await self._connection.expire(instances_key, ttl)
        except Exception as e:
            self.logger.error(f"Failed to register instance in Redis: {str(e)}")
    
    async def unregister_instance(self, instance_id: str) -> None:
        try:
            instance_key = self._format_key(instance_id, "instance")
            await self._connection.delete(instance_key)
            
            # Remove from instance set
            instances_key = self._format_key(instance_id.split(":")[0], "instances")
            await self._connection.srem(instances_key, instance_id)
        except Exception as e:
            self.logger.error(f"Failed to unregister instance from Redis: {str(e)}")
    
    async def get_instances(self, prefix: str) -> List[str]:
        try:
            instances_key = self._format_key(prefix, "instances")
            return list(await self._connection.smembers(instances_key))
        except Exception as e:
            self.logger.error(f"Failed to get instances from Redis: {str(e)}")
            return [] 
import json
import time
import logging
from typing import Any, Dict, Optional, List
import aioredis
import aioboto3

from .storage import StorageBackend, CircuitBreakerStorageError

class ElastiCacheStorageBackend(StorageBackend):
    """ElastiCache Redis implementation of circuit breaker storage backend."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.endpoint = config['endpoint']
        self.port = config.get('port', 6379)
        self.password = config.get('password')
        self.ssl = config.get('ssl', True)
        self.cluster_mode = config.get('cluster_mode', False)
        self.region = config.get('region', 'us-west-2')
        self.aws_access_key_id = config.get('aws_access_key_id')
        self.aws_secret_access_key = config.get('aws_secret_access_key')
        self.redis = None
        
    async def _get_auth_token(self) -> Optional[str]:
        """Get auth token from AWS Secrets Manager."""
        try:
            session = aioboto3.Session(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region
            )
            
            async with session.client('secretsmanager') as client:
                response = await client.get_secret_value(
                    SecretId=f"elasticache/{self.endpoint}"
                )
                return json.loads(response['SecretString'])['password']
                
        except Exception as e:
            self.logger.error(f"Failed to get auth token: {str(e)}")
            return None
        
    async def connect(self) -> None:
        """Establish connection to ElastiCache."""
        try:
            # If using auth token from AWS Secrets Manager
            if not self.password and (self.aws_access_key_id and self.aws_secret_access_key):
                self.password = await self._get_auth_token()
                
            url = f"redis{'s' if self.ssl else ''}://"
            if self.password:
                url += f":{self.password}@"
            url += f"{self.endpoint}:{self.port}"
            
            self.redis = await aioredis.from_url(
                url,
                ssl=self.ssl,
                encoding='utf-8',
                decode_responses=True
            )
            
        except Exception as e:
            raise CircuitBreakerStorageError(f"Failed to connect to ElastiCache: {str(e)}")
    
    async def disconnect(self) -> None:
        """Close connection to ElastiCache."""
        if self.redis:
            await self.redis.close()
            
    async def get_state(self, key: str) -> Optional[Dict[str, Any]]:
        """Get state from ElastiCache."""
        try:
            data = await self.redis.get(self._format_key(key, "state"))
            return json.loads(data) if data else None
            
        except Exception as e:
            self.logger.error(f"Failed to get state from ElastiCache: {str(e)}")
            return None
            
    async def set_state(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Set state in ElastiCache."""
        try:
            key = self._format_key(key, "state")
            data = json.dumps(value)
            if ttl:
                await self.redis.setex(key, ttl, data)
            else:
                await self.redis.set(key, data)
                
        except Exception as e:
            self.logger.error(f"Failed to set state in ElastiCache: {str(e)}")
            
    async def acquire_lock(self, lock_key: str, lock_value: str, ttl: int) -> bool:
        """Acquire distributed lock using ElastiCache."""
        try:
            lock_key = self._format_key(lock_key, "lock")
            return await self.redis.set(lock_key, lock_value, ex=ttl, nx=True)
            
        except Exception as e:
            self.logger.error(f"Failed to acquire lock in ElastiCache: {str(e)}")
            return False
            
    async def release_lock(self, lock_key: str, lock_value: str) -> None:
        """Release distributed lock."""
        try:
            lock_key = self._format_key(lock_key, "lock")
            # Only release if we hold the lock
            current = await self.redis.get(lock_key)
            if current == lock_value:
                await self.redis.delete(lock_key)
                
        except Exception as e:
            self.logger.error(f"Failed to release lock in ElastiCache: {str(e)}")
            
    async def register_instance(self, instance_id: str, ttl: int) -> None:
        """Register circuit breaker instance."""
        try:
            key = self._format_key(instance_id.split(":")[0], "instances")
            await self.redis.sadd(key, instance_id)
            await self.redis.expire(key, ttl)
            
            # Store instance metadata
            instance_key = self._format_key(instance_id, "instance")
            instance_data = json.dumps({
                "instance_id": instance_id,
                "registered_at": int(time.time()),
                "last_heartbeat": int(time.time())
            })
            await self.redis.setex(instance_key, ttl, instance_data)
            
        except Exception as e:
            self.logger.error(f"Failed to register instance in ElastiCache: {str(e)}")
            
    async def unregister_instance(self, instance_id: str) -> None:
        """Unregister circuit breaker instance."""
        try:
            key = self._format_key(instance_id.split(":")[0], "instances")
            await self.redis.srem(key, instance_id)
            
            # Remove instance metadata
            instance_key = self._format_key(instance_id, "instance")
            await self.redis.delete(instance_key)
            
        except Exception as e:
            self.logger.error(f"Failed to unregister instance in ElastiCache: {str(e)}")
            
    async def get_instances(self, prefix: str) -> List[str]:
        """Get list of registered instances."""
        try:
            key = self._format_key(prefix, "instances")
            return list(await self.redis.smembers(key))
            
        except Exception as e:
            self.logger.error(f"Failed to get instances from ElastiCache: {str(e)}")
            return [] 
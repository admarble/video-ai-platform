import json
import logging
import asyncio
from typing import Any, Dict, Optional, List
import aiozk
from aiozk.exc import NoNode as NoNodeError, NodeExists as NodeExistsError

from .storage import StorageBackend, CircuitBreakerStorageError

logger = logging.getLogger(__name__)

class ZooKeeperStorageBackend(StorageBackend):
    """ZooKeeper implementation of circuit breaker storage backend."""
    
    async def connect(self) -> None:
        try:
            hosts = self.config.get('hosts', 'localhost:2181')
            timeout = self.config.get('timeout', 10.0)
            root_path = self.config.get('root_path', '/circuit_breaker')
            
            self._connection = aiozk.ZKClient(
                hosts,
                timeout=timeout
            )
            await self._connection.start()
            
            # Ensure root path exists
            await self._ensure_path(root_path)
            self.root_path = root_path
            
        except Exception as e:
            raise CircuitBreakerStorageError(f"Failed to connect to ZooKeeper: {str(e)}")
    
    async def disconnect(self) -> None:
        if self._connection:
            await self._connection.close()
    
    async def _ensure_path(self, path: str) -> None:
        """Ensure ZooKeeper path exists"""
        try:
            await self._connection.ensure_path(path)
        except Exception as e:
            self.logger.error(f"Error ensuring path {path}: {str(e)}")
            raise
    
    async def get_state(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            path = f"{self.root_path}/{key}"
            try:
                data = await self._connection.get_data(path)
                return json.loads(data.decode()) if data else None
            except NoNodeError:
                return None
        except Exception as e:
            self.logger.error(f"Failed to get state from ZooKeeper: {str(e)}")
            return None
    
    async def set_state(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> None:
        try:
            path = f"{self.root_path}/{key}"
            data = json.dumps(value).encode()
            
            exists = await self._connection.exists(path)
            if exists:
                await self._connection.set_data(path, data)
            else:
                await self._connection.create(path, data, ephemeral=bool(ttl))
                
        except Exception as e:
            self.logger.error(f"Failed to set state in ZooKeeper: {str(e)}")
    
    async def acquire_lock(self, lock_key: str, lock_value: str, ttl: int) -> bool:
        try:
            lock_path = f"{self.root_path}/locks/{lock_key}"
            try:
                await self._connection.create(
                    lock_path,
                    lock_value.encode(),
                    ephemeral=True
                )
                return True
            except NodeExistsError:
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to acquire lock in ZooKeeper: {str(e)}")
            return False
    
    async def release_lock(self, lock_key: str, lock_value: str) -> None:
        try:
            lock_path = f"{self.root_path}/locks/{lock_key}"
            try:
                data = await self._connection.get_data(lock_path)
                if data.decode() == lock_value:
                    await self._connection.delete(lock_path)
            except NoNodeError:
                pass
        except Exception as e:
            self.logger.error(f"Failed to release lock in ZooKeeper: {str(e)}")
    
    async def register_instance(self, instance_id: str, ttl: int) -> None:
        try:
            path = f"{self.root_path}/instances/{instance_id}"
            data = json.dumps({
                'instance_id': instance_id,
                'timestamp': int(asyncio.get_event_loop().time())
            }).encode()
            
            exists = await self._connection.exists(path)
            if exists:
                await self._connection.set_data(path, data)
            else:
                await self._connection.create(path, data, ephemeral=True)
                
        except Exception as e:
            self.logger.error(f"Failed to register instance in ZooKeeper: {str(e)}")
            
    async def get_instances(self, key: str) -> List[str]:
        """Get registe`red` instances"""
        try:
            instances_path = f"{self.root_path}/instances"
            try:
                instances = await self._connection.get_children(instances_path)
                return instances
            except NoNodeError:
                return []
        except Exception as e:
            self.logger.error(f"Failed to get instances from ZooKeeper: {str(e)}")
            return [] 
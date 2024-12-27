import etcd3
import asyncio
from typing import Any, Dict, Optional, List
import json
import time
from .storage import StorageBackend, CircuitBreakerStorageError

class EtcdStorageBackend(StorageBackend):
    """etcd implementation of circuit breaker storage backend."""
    
    async def connect(self) -> None:
        try:
            self._connection = etcd3.client(
                host=self.config.get('host', 'localhost'),
                port=self.config.get('port', 2379)
            )
        except Exception as e:
            raise CircuitBreakerStorageError(f"Failed to connect to etcd: {str(e)}")
    
    async def disconnect(self) -> None:
        if self._connection:
            self._connection.close()
    
    async def get_state(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            value, _ = await asyncio.get_event_loop().run_in_executor(
                None, self._connection.get, self._format_key(key, "state")
            )
            return json.loads(value.decode('utf-8')) if value else None
        except Exception as e:
            self.logger.error(f"Failed to get state from etcd: {str(e)}")
            return None
    
    async def set_state(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> None:
        try:
            formatted_key = self._format_key(key, "state")
            data = json.dumps(value).encode('utf-8')
            
            if ttl:
                lease = self._connection.lease(ttl)
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._connection.put,
                    formatted_key,
                    data,
                    lease=lease
                )
            else:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._connection.put,
                    formatted_key,
                    data
                )
        except Exception as e:
            self.logger.error(f"Failed to set state in etcd: {str(e)}")
    
    async def acquire_lock(self, lock_key: str, lock_value: str, ttl: int) -> bool:
        try:
            formatted_key = self._format_key(lock_key, "lock")
            lease = self._connection.lease(ttl)
            success = await asyncio.get_event_loop().run_in_executor(
                None,
                self._connection.transaction,
                [self._connection.transactions.put(formatted_key, lock_value.encode('utf-8'), lease=lease)],
                [self._connection.transactions.compare.version(formatted_key, '=', 0)]
            )
            return success
        except Exception as e:
            self.logger.error(f"Failed to acquire lock in etcd: {str(e)}")
            return False
    
    async def release_lock(self, lock_key: str, lock_value: str) -> None:
        try:
            formatted_key = self._format_key(lock_key, "lock")
            # Only release if we hold the lock
            value, _ = await asyncio.get_event_loop().run_in_executor(
                None, self._connection.get, formatted_key
            )
            if value and value.decode('utf-8') == lock_value:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._connection.delete,
                    formatted_key
                )
        except Exception as e:
            self.logger.error(f"Failed to release lock in etcd: {str(e)}")
    
    async def register_instance(self, instance_id: str, ttl: int) -> None:
        try:
            instance_key = self._format_key(instance_id, "instance")
            instance_data = json.dumps({
                "instance_id": instance_id,
                "registered_at": int(time.time()),
                "last_heartbeat": int(time.time())
            }).encode('utf-8')
            
            # Create instance directory
            instances_prefix = self._format_key(instance_id.split(":")[0], "instances")
            instance_path = f"{instances_prefix}/{instance_id}"
            
            lease = self._connection.lease(ttl)
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._connection.put,
                instance_path,
                instance_data,
                lease=lease
            )
        except Exception as e:
            self.logger.error(f"Failed to register instance in etcd: {str(e)}")
    
    async def unregister_instance(self, instance_id: str) -> None:
        try:
            instances_prefix = self._format_key(instance_id.split(":")[0], "instances")
            instance_path = f"{instances_prefix}/{instance_id}"
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._connection.delete,
                instance_path
            )
        except Exception as e:
            self.logger.error(f"Failed to unregister instance from etcd: {str(e)}")
    
    async def get_instances(self, prefix: str) -> List[str]:
        try:
            instances_prefix = self._format_key(prefix, "instances")
            _, kvs = await asyncio.get_event_loop().run_in_executor(
                None,
                self._connection.get_prefix,
                instances_prefix
            )
            return [
                key.split("/")[-1]
                for key, _ in kvs
            ]
        except Exception as e:
            self.logger.error(f"Failed to get instances from etcd: {str(e)}")
            return [] 
            raise CircuitBreakerStorageError(f"Failed to unregister instance from etcd: {str(e)}") 
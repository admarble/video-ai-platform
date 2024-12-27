import aiohttp
import asyncio
from typing import Any, Dict, Optional, List
import json
import base64
import time
from .storage import StorageBackend, CircuitBreakerStorageError

class ConsulStorageBackend(StorageBackend):
    """Consul implementation of circuit breaker storage backend."""
    
    async def connect(self) -> None:
        try:
            self.base_url = self.config.get('url', 'http://localhost:8500')
            self._session = aiohttp.ClientSession()
            self._consul_session = None
            self._health_check_tasks = {}
        except Exception as e:
            raise CircuitBreakerStorageError(f"Failed to connect to Consul: {str(e)}")
    
    async def disconnect(self) -> None:
        if self._session:
            # Cancel all health check tasks
            for task in self._health_check_tasks.values():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await self._session.close()
    
    async def get_state(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            async with self._session.get(
                f"{self.base_url}/v1/kv/{self._format_key(key, 'state')}"
            ) as response:
                if response.status == 404:
                    return None
                if response.status != 200:
                    raise CircuitBreakerStorageError(f"Consul returned status {response.status}")
                
                data = await response.json()
                if not data:
                    return None
                
                value = base64.b64decode(data[0]['Value']).decode('utf-8')
                return json.loads(value)
        except Exception as e:
            self.logger.error(f"Failed to get state from Consul: {str(e)}")
            return None
    
    async def set_state(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> None:
        try:
            formatted_key = self._format_key(key, "state")
            data = json.dumps(value)
            params = {}
            
            if ttl:
                if not self._consul_session:
                    # Create session with TTL
                    async with self._session.put(
                        f"{self.base_url}/v1/session/create",
                        json={"TTL": f"{ttl}s"}
                    ) as response:
                        if response.status != 200:
                            raise CircuitBreakerStorageError(f"Failed to create Consul session")
                        session_data = await response.json()
                        self._consul_session = session_data['ID']
                
                params['acquire'] = self._consul_session
            
            async with self._session.put(
                f"{self.base_url}/v1/kv/{formatted_key}",
                params=params,
                data=data
            ) as response:
                if response.status != 200:
                    raise CircuitBreakerStorageError(f"Consul returned status {response.status}")
        except Exception as e:
            self.logger.error(f"Failed to set state in Consul: {str(e)}")
    
    async def acquire_lock(self, lock_key: str, lock_value: str, ttl: int) -> bool:
        try:
            # Create session for lock
            async with self._session.put(
                f"{self.base_url}/v1/session/create",
                json={"TTL": f"{ttl}s", "Behavior": "delete"}
            ) as response:
                if response.status != 200:
                    raise CircuitBreakerStorageError("Failed to create Consul session for lock")
                session_data = await response.json()
                session_id = session_data['ID']
            
            # Try to acquire lock
            formatted_key = self._format_key(lock_key, "lock")
            async with self._session.put(
                f"{self.base_url}/v1/kv/{formatted_key}",
                params={"acquire": session_id},
                data=lock_value
            ) as response:
                if response.status != 200:
                    raise CircuitBreakerStorageError(f"Consul returned status {response.status}")
                success = await response.text()
                return success.strip().lower() == "true"
        except Exception as e:
            self.logger.error(f"Failed to acquire lock in Consul: {str(e)}")
            return False
    
    async def release_lock(self, lock_key: str, lock_value: str) -> None:
        try:
            formatted_key = self._format_key(lock_key, "lock")
            # Only release if we hold the lock
            async with self._session.get(f"{self.base_url}/v1/kv/{formatted_key}") as response:
                if response.status == 200:
                    data = await response.json()
                    if data and base64.b64decode(data[0]['Value']).decode('utf-8') == lock_value:
                        async with self._session.delete(
                            f"{self.base_url}/v1/kv/{formatted_key}"
                        ) as del_response:
                            if del_response.status not in [200, 404]:
                                raise CircuitBreakerStorageError(f"Consul returned status {del_response.status}")
        except Exception as e:
            self.logger.error(f"Failed to release lock in Consul: {str(e)}")
    
    async def register_instance(self, instance_id: str, ttl: int) -> None:
        try:
            service_id = f"circuit-breaker-{instance_id}"
            service_name = instance_id.split(":")[0]
            
            # Register service with health check
            service = {
                'ID': service_id,
                'Name': service_name,
                'Check': {
                    'TTL': f"{ttl}s",
                    'DeregisterCriticalServiceAfter': f"{ttl*2}s"
                }
            }
            
            async with self._session.put(
                f"{self.base_url}/v1/agent/service/register",
                json=service
            ) as response:
                if response.status != 200:
                    raise CircuitBreakerStorageError(f"Failed to register service: {response.status}")
            
            # Start health check task
            if service_id not in self._health_check_tasks:
                self._health_check_tasks[service_id] = asyncio.create_task(
                    self._health_check_loop(service_id, ttl)
                )
            
            # Store instance metadata
            instance_key = self._format_key(instance_id, "instance")
            instance_data = json.dumps({
                "instance_id": instance_id,
                "registered_at": int(time.time()),
                "last_heartbeat": int(time.time())
            })
            
            async with self._session.put(
                f"{self.base_url}/v1/kv/{instance_key}",
                data=instance_data
            ) as response:
                if response.status != 200:
                    raise CircuitBreakerStorageError(f"Failed to store instance metadata")
                    
        except Exception as e:
            self.logger.error(f"Failed to register instance in Consul: {str(e)}")
    
    async def unregister_instance(self, instance_id: str) -> None:
        try:
            service_id = f"circuit-breaker-{instance_id}"
            
            # Cancel health check task
            if service_id in self._health_check_tasks:
                self._health_check_tasks[service_id].cancel()
                try:
                    await self._health_check_tasks[service_id]
                except asyncio.CancelledError:
                    pass
                del self._health_check_tasks[service_id]
            
            # Deregister service
            async with self._session.put(
                f"{self.base_url}/v1/agent/service/deregister/{service_id}"
            ) as response:
                if response.status not in [200, 404]:
                    raise CircuitBreakerStorageError(f"Failed to deregister service: {response.status}")
            
            # Remove instance metadata
            instance_key = self._format_key(instance_id, "instance")
            async with self._session.delete(
                f"{self.base_url}/v1/kv/{instance_key}"
            ) as response:
                if response.status not in [200, 404]:
                    raise CircuitBreakerStorageError(f"Failed to remove instance metadata")
                    
        except Exception as e:
            self.logger.error(f"Failed to unregister instance from Consul: {str(e)}")
    
    async def get_instances(self, prefix: str) -> List[str]:
        try:
            service_name = prefix.split(":")[-1]
            async with self._session.get(
                f"{self.base_url}/v1/health/service/{service_name}"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return [
                        service['Service']['ID'].replace('circuit-breaker-', '')
                        for service in data
                        if all(check['Status'] == 'passing' for check in service['Checks'])
                    ]
                return []
        except Exception as e:
            self.logger.error(f"Failed to get instances from Consul: {str(e)}")
            return []
    
    async def _health_check_loop(self, service_id: str, ttl: int) -> None:
        """Background task to update TTL health check."""
        while True:
            try:
                await asyncio.sleep(ttl // 2)
                async with self._session.put(
                    f"{self.base_url}/v1/agent/check/pass/service:{service_id}"
                ) as response:
                    if response.status != 200:
                        self.logger.error(f"Failed to update health check: {response.status}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {str(e)}")
                await asyncio.sleep(1)  # Avoid tight loop on persistent errors 
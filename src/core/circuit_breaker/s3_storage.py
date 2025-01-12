import json
import time
import logging
from typing import Any, Dict, Optional, List
import aioboto3
from botocore.exceptions import ClientError

from .storage import StorageBackend, CircuitBreakerStorageError

class S3StorageBackend(StorageBackend):
    """S3 implementation of circuit breaker storage backend."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bucket_name = config['bucket_name']
        self.prefix = config.get('prefix', 'circuit-breaker/')
        self.region = config.get('region', 'us-west-2')
        self.aws_access_key_id = config.get('aws_access_key_id')
        self.aws_secret_access_key = config.get('aws_secret_access_key')
        self.session = None
        
    async def connect(self) -> None:
        """Establish connection to S3."""
        try:
            self.session = aioboto3.Session(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region
            )
            
            # Ensure bucket exists
            async with self.session.client('s3') as s3:
                try:
                    await s3.head_bucket(Bucket=self.bucket_name)
                except ClientError as e:
                    if e.response['Error']['Code'] == '404':
                        await s3.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={
                                'LocationConstraint': self.region
                            }
                        )
                    else:
                        raise CircuitBreakerStorageError(f"Failed to connect to S3: {str(e)}")
                        
        except Exception as e:
            raise CircuitBreakerStorageError(f"Failed to connect to S3: {str(e)}")
    
    async def disconnect(self) -> None:
        """Close connection to S3."""
        if self.session:
            await self.session.close()
            
    def _get_object_key(self, key: str) -> str:
        """Get full S3 object key with prefix."""
        return f"{self.prefix.rstrip('/')}/{key}"
            
    async def get_state(self, key: str) -> Optional[Dict[str, Any]]:
        """Get state from S3."""
        try:
            object_key = self._get_object_key(key)
            
            async with self.session.client('s3') as s3:
                try:
                    response = await s3.get_object(
                        Bucket=self.bucket_name,
                        Key=object_key
                    )
                    data = await response['Body'].read()
                    return json.loads(data)
                except ClientError as e:
                    if e.response['Error']['Code'] == 'NoSuchKey':
                        return None
                    raise
                    
        except Exception as e:
            self.logger.error(f"Failed to get state from S3: {str(e)}")
            return None
            
    async def set_state(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Set state in S3."""
        try:
            object_key = self._get_object_key(key)
            data = json.dumps(value)
            
            async with self.session.client('s3') as s3:
                metadata = {}
                if ttl:
                    metadata['expires'] = str(int(time.time() + ttl))
                    
                await s3.put_object(
                    Bucket=self.bucket_name,
                    Key=object_key,
                    Body=data,
                    ContentType='application/json',
                    Metadata=metadata
                )
                
        except Exception as e:
            self.logger.error(f"Failed to set state in S3: {str(e)}")
            
    async def acquire_lock(self, lock_key: str, lock_value: str, ttl: int) -> bool:
        """Acquire distributed lock using S3."""
        try:
            lock_key = self._get_object_key(f"locks/{lock_key}")
            
            async with self.session.client('s3') as s3:
                try:
                    await s3.put_object(
                        Bucket=self.bucket_name,
                        Key=lock_key,
                        Body=lock_value,
                        Metadata={'expires': str(int(time.time() + ttl))},
                        # Use versioning condition to ensure atomic operation
                        ExpectedBucketOwner=lock_value
                    )
                    return True
                except ClientError as e:
                    if e.response['Error']['Code'] == 'PreconditionFailed':
                        return False
                    raise
                    
        except Exception as e:
            self.logger.error(f"Failed to acquire lock in S3: {str(e)}")
            return False
            
    async def release_lock(self, lock_key: str, lock_value: str) -> None:
        """Release distributed lock."""
        try:
            lock_key = self._get_object_key(f"locks/{lock_key}")
            
            async with self.session.client('s3') as s3:
                try:
                    response = await s3.get_object(
                        Bucket=self.bucket_name,
                        Key=lock_key
                    )
                    if await response['Body'].read() == lock_value.encode():
                        await s3.delete_object(
                            Bucket=self.bucket_name,
                            Key=lock_key
                        )
                except ClientError:
                    pass
                    
        except Exception as e:
            self.logger.error(f"Failed to release lock in S3: {str(e)}")
            
    async def register_instance(self, instance_id: str, ttl: int) -> None:
        """Register circuit breaker instance."""
        try:
            instance_key = self._get_object_key(f"instances/{instance_id}")
            
            async with self.session.client('s3') as s3:
                await s3.put_object(
                    Bucket=self.bucket_name,
                    Key=instance_key,
                    Body=instance_id,
                    Metadata={'expires': str(int(time.time() + ttl))}
                )
                
        except Exception as e:
            self.logger.error(f"Failed to register instance in S3: {str(e)}")
            
    async def unregister_instance(self, instance_id: str) -> None:
        """Unregister circuit breaker instance."""
        try:
            instance_key = self._get_object_key(f"instances/{instance_id}")
            
            async with self.session.client('s3') as s3:
                await s3.delete_object(
                    Bucket=self.bucket_name,
                    Key=instance_key
                )
                
        except Exception as e:
            self.logger.error(f"Failed to unregister instance in S3: {str(e)}")
            
    async def get_instances(self, prefix: str) -> List[str]:
        """Get list of registered instances."""
        try:
            prefix = self._get_object_key(f"instances/{prefix}")
            
            async with self.session.client('s3') as s3:
                response = await s3.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=prefix
                )
                
                instances = []
                current_time = time.time()
                
                for obj in response.get('Contents', []):
                    try:
                        obj_response = await s3.get_object(
                            Bucket=self.bucket_name,
                            Key=obj['Key']
                        )
                        expires = int(obj_response['Metadata']['expires'])
                        if expires > current_time:
                            instances.append(await obj_response['Body'].read().decode())
                    except ClientError:
                        continue
                        
                return instances
                
        except Exception as e:
            self.logger.error(f"Failed to get instances from S3: {str(e)}")
            return []
import json
import time
import logging
import asyncio
from typing import Any, Dict, Optional, List
import aioboto3
from botocore.exceptions import ClientError

from .storage import StorageBackend, CircuitBreakerStorageError

logger = logging.getLogger(__name__)

class DynamoDBStorageBackend(StorageBackend):
    """DynamoDB implementation of circuit breaker storage backend."""
    
    async def connect(self) -> None:
        try:
            region = self.config.get('region', 'us-west-2')
            table_name = self.config.get('table_name')
            if not table_name:
                raise CircuitBreakerStorageError("DynamoDB table_name is required")
            
            # Optional AWS credentials
            credentials = {}
            if 'aws_access_key_id' in self.config:
                credentials['aws_access_key_id'] = self.config['aws_access_key_id']
            if 'aws_secret_access_key' in self.config:
                credentials['aws_secret_access_key'] = self.config['aws_secret_access_key']
            
            self.session = aioboto3.Session(
                region_name=region,
                **credentials
            )
            self.table_name = table_name
            
            # Ensure table exists
            async with self.session.client('dynamodb') as client:
                try:
                    await client.describe_table(TableName=table_name)
                except ClientError as e:
                    if e.response['Error']['Code'] == 'ResourceNotFoundException':
                        await self._create_table()
                    else:
                        raise
            
        except Exception as e:
            raise CircuitBreakerStorageError(f"Failed to connect to DynamoDB: {str(e)}")
    
    async def _create_table(self):
        """Create DynamoDB table if it doesn't exist"""
        async with self.session.client('dynamodb') as client:
            await client.create_table(
                TableName=self.table_name,
                KeySchema=[
                    {'AttributeName': 'key', 'KeyType': 'HASH'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'key', 'AttributeType': 'S'}
                ],
                ProvisionedThroughput={
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            )
            # Wait for table to be created
            while True:
                try:
                    await client.describe_table(TableName=self.table_name)
                    break
                except ClientError:
                    await asyncio.sleep(1)
    
    async def disconnect(self) -> None:
        if self.session:
            await self.session.close()
    
    async def get_state(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            async with self.session.client('dynamodb') as client:
                response = await client.get_item(
                    TableName=self.table_name,
                    Key={'key': {'S': key}},
                    ConsistentRead=True
                )
                
                if 'Item' not in response:
                    return None
                
                # Check if item is expired
                if 'ttl' in response['Item']:
                    ttl = int(response['Item']['ttl']['N'])
                    if ttl <= int(time.time()):
                        return None
                
                return json.loads(response['Item']['data']['S'])
            
        except Exception as e:
            self.logger.error(f"Failed to get state from DynamoDB: {str(e)}")
            return None
    
    async def set_state(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> None:
        try:
            item = {
                'key': {'S': key},
                'data': {'S': json.dumps(value)},
                'updated_at': {'N': str(int(time.time()))}
            }
            
            if ttl:
                item['ttl'] = {'N': str(int(time.time() + ttl))}
            
            async with self.session.client('dynamodb') as client:
                await client.put_item(
                    TableName=self.table_name,
                    Item=item
                )
        except Exception as e:
            self.logger.error(f"Failed to set state in DynamoDB: {str(e)}")
    
    async def acquire_lock(self, lock_key: str, lock_value: str, ttl: int) -> bool:
        try:
            lock_key = f"lock:{lock_key}"
            expiry = int(time.time() + ttl)
            
            async with self.session.client('dynamodb') as client:
                try:
                    await client.put_item(
                        TableName=self.table_name,
                        Item={
                            'key': {'S': lock_key},
                            'value': {'S': lock_value},
                            'ttl': {'N': str(expiry)}
                        },
                        ConditionExpression='attribute_not_exists(#k)',
                        ExpressionAttributeNames={'#k': 'key'}
                    )
                    return True
                except ClientError as e:
                    if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                        return False
                    raise
                
        except Exception as e:
            self.logger.error(f"Failed to acquire lock in DynamoDB: {str(e)}")
            return False
    
    async def release_lock(self, lock_key: str, lock_value: str) -> None:
        try:
            lock_key = f"lock:{lock_key}"
            
            async with self.session.client('dynamodb') as client:
                # Only delete if we own the lock
                await client.delete_item(
                    TableName=self.table_name,
                    Key={'key': {'S': lock_key}},
                    ConditionExpression='#v = :value',
                    ExpressionAttributeNames={'#v': 'value'},
                    ExpressionAttributeValues={':value': {'S': lock_value}}
                )
                
        except ClientError as e:
            if e.response['Error']['Code'] != 'ConditionalCheckFailedException':
                self.logger.error(f"Failed to release lock in DynamoDB: {str(e)}")
        except Exception as e:
            self.logger.error(f"Failed to release lock in DynamoDB: {str(e)}")
    
    async def register_instance(self, instance_id: str, ttl: int) -> None:
        try:
            instance_key = f"instance:{instance_id}"
            expiry = int(time.time() + ttl)
            
            async with self.session.client('dynamodb') as client:
                await client.put_item(
                    TableName=self.table_name,
                    Item={
                        'key': {'S': instance_key},
                        'value': {'S': instance_id},
                        'ttl': {'N': str(expiry)}
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Failed to register instance in DynamoDB: {str(e)}")
            
    async def get_instances(self, key: str) -> List[str]:
        """Get registered instances"""
        try:
            prefix = f"instance:"
            
            async with self.session.client('dynamodb') as client:
                response = await client.scan(
                    TableName=self.table_name,
                    FilterExpression='begins_with(#k, :prefix)',
                    ExpressionAttributeNames={'#k': 'key'},
                    ExpressionAttributeValues={':prefix': {'S': prefix}}
                )
                
                instances = []
                for item in response.get('Items', []):
                    instances.append(item['value']['S'])
                return instances
                
        except Exception as e:
            self.logger.error(f"Failed to get instances from DynamoDB: {str(e)}")
            return [] 
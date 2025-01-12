import pytest
import asyncio
from unittest.mock import MagicMock, patch
import time

from src.core.circuit_breaker import (
    HealthCheckConfig,
    FailoverStrategy,
    FailoverStorageBackend,
    create_failover_storage,
    CircuitBreakerStorageError
)

@pytest.fixture
def storage_configs():
    return [
        {
            'name': 'primary',
            'type': 'redis',
            'config': {
                'url': 'redis://primary:6379'
            }
        },
        {
            'name': 'secondary',
            'type': 'elasticache',
            'config': {
                'endpoint': 'secondary.cache.amazonaws.com',
                'port': 6379
            }
        },
        {
            'name': 'fallback',
            'type': 's3',
            'config': {
                'bucket_name': 'circuit-breaker-fallback'
            }
        }
    ]

@pytest.fixture
def health_config():
    return HealthCheckConfig(
        check_interval=1,
        timeout=1,
        threshold=2,
        recovery_threshold=1
    )

@pytest.mark.asyncio
async def test_failover_storage_creation(storage_configs, health_config):
    with patch('src.core.circuit_breaker.failover_storage.RedisStorageBackend') as mock_redis, \
         patch('src.core.circuit_breaker.failover_storage.ElastiCacheStorageBackend') as mock_elasticache, \
         patch('src.core.circuit_breaker.failover_storage.S3StorageBackend') as mock_s3:
        
        # Setup mock backends
        mock_backends = [mock_redis.return_value, mock_elasticache.return_value, mock_s3.return_value]
        for backend in mock_backends:
            backend.connect = AsyncMock()
            backend.disconnect = AsyncMock()
            backend.get_state = AsyncMock(return_value={'timestamp': time.time()})
            backend.set_state = AsyncMock()
        
        # Create failover storage
        storage = await create_failover_storage(
            storage_configs=storage_configs,
            strategy=FailoverStrategy.LATENCY_BASED,
            health_config=health_config
        )
        
        assert isinstance(storage, FailoverStorageBackend)
        assert len(storage.backends) == 3
        assert all(backend.connect.called for backend in mock_backends)

@pytest.mark.asyncio
async def test_health_check_failover(storage_configs, health_config):
    with patch('src.core.circuit_breaker.failover_storage.RedisStorageBackend') as mock_redis, \
         patch('src.core.circuit_breaker.failover_storage.ElastiCacheStorageBackend') as mock_elasticache, \
         patch('src.core.circuit_breaker.failover_storage.S3StorageBackend') as mock_s3:
        
        # Setup mock backends
        mock_redis = mock_redis.return_value
        mock_elasticache = mock_elasticache.return_value
        mock_s3 = mock_s3.return_value
        
        for backend in [mock_redis, mock_elasticache, mock_s3]:
            backend.connect = AsyncMock()
            backend.disconnect = AsyncMock()
            backend.get_state = AsyncMock(return_value={'timestamp': time.time()})
            backend.set_state = AsyncMock()
        
        # Create failover storage
        storage = await create_failover_storage(
            storage_configs=storage_configs,
            strategy=FailoverStrategy.PRIORITY,
            health_config=health_config
        )
        
        # Simulate primary backend failure
        mock_redis.get_state.side_effect = CircuitBreakerStorageError("Connection failed")
        
        # Wait for health check to detect failure
        await asyncio.sleep(health_config.check_interval * (health_config.threshold + 1))
        
        # Verify failover to secondary
        test_key = "test"
        test_value = {"data": "test"}
        await storage.set_state(test_key, test_value)
        
        assert not mock_redis.set_state.called
        assert mock_elasticache.set_state.called
        
        # Get health status
        health_status = storage.get_health_status()
        assert not health_status['primary']['healthy']
        assert health_status['secondary']['healthy']

@pytest.mark.asyncio
async def test_latency_based_selection(storage_configs, health_config):
    with patch('src.core.circuit_breaker.failover_storage.RedisStorageBackend') as mock_redis, \
         patch('src.core.circuit_breaker.failover_storage.ElastiCacheStorageBackend') as mock_elasticache, \
         patch('src.core.circuit_breaker.failover_storage.S3StorageBackend') as mock_s3:
        
        # Setup mock backends with different latencies
        mock_backends = {
            'primary': (mock_redis.return_value, 0.1),
            'secondary': (mock_elasticache.return_value, 0.05),
            'fallback': (mock_s3.return_value, 0.2)
        }
        
        for backend, latency in mock_backends.values():
            backend.connect = AsyncMock()
            backend.disconnect = AsyncMock()
            backend.get_state = AsyncMock(return_value={'timestamp': time.time()})
            backend.set_state = AsyncMock()
            
            # Simulate different latencies
            async def delayed_response(backend, latency):
                await asyncio.sleep(latency)
                return {'timestamp': time.time()}
            
            backend.get_state.side_effect = lambda key: delayed_response(backend, latency)
        
        # Create failover storage with latency-based strategy
        storage = await create_failover_storage(
            storage_configs=storage_configs,
            strategy=FailoverStrategy.LATENCY_BASED,
            health_config=health_config
        )
        
        # Wait for health checks to measure latencies
        await asyncio.sleep(health_config.check_interval * 2)
        
        # Verify that the fastest backend is selected
        test_key = "test"
        test_value = {"data": "test"}
        await storage.set_state(test_key, test_value)
        
        assert mock_backends['secondary'][0].set_state.called
        assert not mock_backends['primary'][0].set_state.called
        assert not mock_backends['fallback'][0].set_state.called

class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs) 
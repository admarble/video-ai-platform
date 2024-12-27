from typing import Dict, Any, List
from .failover_storage import FailoverStorageBackend, FailoverStrategy, HealthCheckConfig

async def create_failover_storage(
    storage_configs: List[Dict[str, Any]],
    strategy: FailoverStrategy = FailoverStrategy.LATENCY_BASED,
    health_config: HealthCheckConfig = None
) -> FailoverStorageBackend:
    """
    Create and initialize a failover storage backend.
    
    Args:
        storage_configs: List of storage backend configurations
        strategy: Failover strategy to use
        health_config: Health check configuration
        
    Returns:
        Initialized FailoverStorageBackend instance
    """
    storage = FailoverStorageBackend(
        storage_configs=storage_configs,
        strategy=strategy,
        health_config=health_config
    )
    await storage.connect()
    return storage 
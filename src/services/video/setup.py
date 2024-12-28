from typing import Dict, Any, Optional
from pathlib import Path

from ...monitoring import CostMonitor, CostConfig
from ..service_manager import ServiceManager
from .cost_aware_processor import CostAwareVideoProcessor

async def setup_cost_aware_processor(
    base_dir: Path,
    config: Optional[Dict[str, Any]] = None
) -> CostAwareVideoProcessor:
    """Setup cost-aware video processor with cost monitoring"""
    # Create base directories if they don't exist
    costs_dir = base_dir / "costs"
    costs_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize cost monitor with default or custom config
    cost_config = None
    if config and 'cost_tracking' in config:
        cost_config = CostConfig(
            compute_cost_per_hour=config['cost_tracking'].get('compute_cost_per_hour', 0.50),
            storage_cost_per_gb_month=config['cost_tracking'].get('storage_cost_per_gb_month', 0.02),
            bandwidth_cost_per_gb=config['cost_tracking'].get('bandwidth_cost_per_gb', 0.05),
            model_cost_per_1k=config['cost_tracking'].get('model_cost_per_1k', 0.10),
            cache_cost_per_gb_hour=config['cost_tracking'].get('cache_cost_per_gb_hour', 0.01)
        )
    
    cost_monitor = CostMonitor(
        config=cost_config,
        history_file=costs_dir / "history.json"
    )

    # Initialize service manager
    service_manager = ServiceManager(config)

    # Create and return cost-aware processor
    processor = CostAwareVideoProcessor(
        cost_monitor=cost_monitor,
        service_manager=service_manager,
        base_dir=base_dir,
        config=config
    )

    return processor 
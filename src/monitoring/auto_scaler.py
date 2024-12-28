from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import time
import asyncio
import logging
from datetime import datetime, timedelta
import psutil
import torch

from .cost_monitor import CostMonitor, ResourceType

class ScalingTrigger(Enum):
    """Triggers for auto-scaling decisions"""
    COST_THRESHOLD = "cost_threshold"      # Cost exceeds threshold
    UTILIZATION = "utilization"            # Resource utilization
    QUEUE_LENGTH = "queue_length"          # Processing queue length
    PROCESSING_TIME = "processing_time"    # Video processing duration
    ERROR_RATE = "error_rate"              # Processing error rate

@dataclass
class ScalingRule:
    """Defines a scaling rule"""
    trigger: ScalingTrigger
    threshold: float
    scale_factor: float
    cooldown_minutes: int = 10
    min_instances: int = 1
    max_instances: int = 10
    last_scaled: Optional[float] = None

@dataclass
class ResourceLimits:
    """Resource limits for scaling"""
    max_cost_per_hour: float
    max_instances: int
    min_free_memory: float  # GB
    max_gpu_memory: float   # GB
    max_storage: float      # GB

class AutoScaler:
    """Manages cost-based auto-scaling"""
    
    def __init__(
        self,
        cost_monitor: CostMonitor,
        resource_limits: ResourceLimits,
        check_interval: int = 60
    ):
        self.cost_monitor = cost_monitor
        self.resource_limits = resource_limits
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)
        
        # Initialize scaling rules
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self._setup_default_rules()
        
        # Track current state
        self.current_instances = 1
        self.is_running = False
        self._monitor_task = None

    def _setup_default_rules(self):
        """Setup default scaling rules"""
        self.scaling_rules = {
            "cost_limit": ScalingRule(
                trigger=ScalingTrigger.COST_THRESHOLD,
                threshold=self.resource_limits.max_cost_per_hour * 0.8,  # 80% of max
                scale_factor=0.5,  # Scale down by 50% when cost is too high
                cooldown_minutes=15
            ),
            "utilization": ScalingRule(
                trigger=ScalingTrigger.UTILIZATION,
                threshold=0.75,  # 75% utilization
                scale_factor=1.5,  # Scale up by 50% when busy
                cooldown_minutes=10
            ),
            "queue_length": ScalingRule(
                trigger=ScalingTrigger.QUEUE_LENGTH,
                threshold=5,  # Videos in queue per instance
                scale_factor=2.0,  # Double instances for long queues
                cooldown_minutes=5
            ),
            "processing_time": ScalingRule(
                trigger=ScalingTrigger.PROCESSING_TIME,
                threshold=300,  # 5 minutes
                scale_factor=1.25,  # Scale up by 25% when slow
                cooldown_minutes=10
            ),
            "error_rate": ScalingRule(
                trigger=ScalingTrigger.ERROR_RATE,
                threshold=0.1,  # 10% error rate
                scale_factor=0.75,  # Scale down by 25% when errors high
                cooldown_minutes=20
            )
        }

    async def start(self):
        """Start auto-scaling monitor"""
        if self.is_running:
            return

        self.is_running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("Auto-scaler started")

    async def stop(self):
        """Stop auto-scaling monitor"""
        self.is_running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Auto-scaler stopped")

    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                await self._check_scaling_rules()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in scaling monitor: {str(e)}")
                await asyncio.sleep(5)  # Short sleep on error

    async def _check_scaling_rules(self):
        """Check all scaling rules"""
        # Get current metrics
        metrics = await self._get_current_metrics()
        
        # Check each rule
        scale_factors = []
        for rule_name, rule in self.scaling_rules.items():
            # Skip if in cooldown
            if rule.last_scaled and \
               time.time() - rule.last_scaled < rule.cooldown_minutes * 60:
                continue

            # Get metric value for rule
            metric_value = metrics.get(rule.trigger.value)
            if metric_value is None:
                continue

            # Calculate scaling factor if threshold exceeded
            if self._should_scale(rule, metric_value):
                factor = self._calculate_scale_factor(rule, metric_value)
                scale_factors.append(factor)
                self.logger.info(
                    f"Rule {rule_name} triggered: metric={metric_value}, "
                    f"threshold={rule.threshold}, factor={factor}"
                )

        # Apply scaling if needed
        if scale_factors:
            # Use average of all triggered rules
            avg_factor = sum(scale_factors) / len(scale_factors)
            await self._apply_scaling(avg_factor)

    async def _get_current_metrics(self) -> Dict[str, float]:
        """Get current metrics for scaling decisions"""
        try:
            # Get cost metrics
            costs = self.cost_monitor.get_current_costs()
            hourly_cost = costs.total_cost

            # Get resource utilization
            resources = await self._get_resource_utilization()

            # Get processing metrics from cost monitor
            processing_metrics = self.cost_monitor.get_cost_summary(days=1)
            queue_length = processing_metrics.get('pending_tasks', 0)
            error_rate = processing_metrics.get('error_rate', 0.0)
            avg_processing_time = processing_metrics.get('avg_processing_time', 0.0)

            return {
                "cost_threshold": hourly_cost,
                "utilization": resources['cpu_utilization'],
                "queue_length": queue_length / max(self.current_instances, 1),
                "processing_time": avg_processing_time,
                "error_rate": error_rate
            }

        except Exception as e:
            self.logger.error(f"Error getting metrics: {str(e)}")
            return {}

    def _should_scale(self, rule: ScalingRule, metric_value: float) -> bool:
        """Check if scaling should be triggered"""
        if rule.trigger in [ScalingTrigger.COST_THRESHOLD, 
                          ScalingTrigger.UTILIZATION,
                          ScalingTrigger.ERROR_RATE]:
            return metric_value > rule.threshold
        else:
            return metric_value > rule.threshold

    def _calculate_scale_factor(
        self,
        rule: ScalingRule,
        metric_value: float
    ) -> float:
        """Calculate scaling factor based on metric value"""
        if rule.trigger in [ScalingTrigger.COST_THRESHOLD, 
                          ScalingTrigger.ERROR_RATE]:
            # Scale down when these metrics are high
            return min(1.0, rule.scale_factor)
        else:
            # Scale up when other metrics are high
            return max(1.0, rule.scale_factor)

    async def _apply_scaling(self, scale_factor: float):
        """Apply scaling decision"""
        try:
            # Calculate new instance count
            new_instances = int(self.current_instances * scale_factor)
            
            # Apply limits
            new_instances = max(
                min(new_instances, self.resource_limits.max_instances),
                self.scaling_rules["cost_limit"].min_instances
            )

            if new_instances == self.current_instances:
                return

            # Check resource limits
            if not await self._check_resource_limits(new_instances):
                self.logger.warning("Resource limits would be exceeded, skipping scaling")
                return

            # Apply scaling
            success = await self._scale_instances(new_instances)
            if success:
                self.logger.info(
                    f"Scaled from {self.current_instances} to {new_instances} instances"
                )
                self.current_instances = new_instances

                # Update last scaled timestamp for all rules
                current_time = time.time()
                for rule in self.scaling_rules.values():
                    rule.last_scaled = current_time

        except Exception as e:
            self.logger.error(f"Error applying scaling: {str(e)}")

    async def _check_resource_limits(self, new_instances: int) -> bool:
        """Check if scaling would exceed resource limits"""
        try:
            # Check cost projection
            current_costs = self.cost_monitor.get_current_costs()
            projected_cost = current_costs.total_cost * (new_instances / self.current_instances)
            if projected_cost > self.resource_limits.max_cost_per_hour:
                return False

            # Check resource availability
            resources = await self._get_resource_utilization()
            
            # Check memory
            projected_memory = resources['memory_usage'] * (new_instances / self.current_instances)
            if projected_memory > self.resource_limits.min_free_memory:
                return False

            # Check GPU if available
            if 'gpu_memory' in resources:
                projected_gpu = resources['gpu_memory'] * (new_instances / self.current_instances)
                if projected_gpu > self.resource_limits.max_gpu_memory:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking resource limits: {str(e)}")
            return False

    async def _get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            resources = {
                'cpu_utilization': cpu_percent / 100,
                'memory_usage': memory.used / (1024 * 1024 * 1024),  # GB
            }

            # Add GPU metrics if available
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # GB
                resources['gpu_memory'] = gpu_memory

            return resources

        except Exception as e:
            self.logger.error(f"Error getting resource utilization: {str(e)}")
            return {
                'cpu_utilization': 0,
                'memory_usage': 0
            }

    async def _scale_instances(self, new_count: int) -> bool:
        """Scale to specified number of instances"""
        # This is a placeholder - implement actual scaling logic
        # based on your infrastructure (e.g., Kubernetes, cloud provider)
        self.logger.info(f"Scaling to {new_count} instances")
        return True

async def create_auto_scaler(
    cost_monitor: CostMonitor,
    config: Optional[Dict[str, Any]] = None
) -> AutoScaler:
    """Create and initialize auto-scaler instance"""
    config = config or {}
    
    # Create resource limits
    limits = ResourceLimits(
        max_cost_per_hour=config.get('max_cost_per_hour', 10.0),
        max_instances=config.get('max_instances', 10),
        min_free_memory=config.get('min_free_memory', 2.0),  # 2GB
        max_gpu_memory=config.get('max_gpu_memory', 16.0),   # 16GB
        max_storage=config.get('max_storage', 1000.0)        # 1TB
    )

    # Create auto-scaler
    scaler = AutoScaler(
        cost_monitor=cost_monitor,
        resource_limits=limits,
        check_interval=config.get('check_interval', 60)
    )

    # Start monitoring
    await scaler.start()

    return scaler 
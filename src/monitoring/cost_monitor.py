from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import time
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path

class ResourceType(Enum):
    """Types of resources to monitor costs for"""
    COMPUTE = "compute"          # CPU/GPU compute time
    STORAGE = "storage"          # File storage costs
    BANDWIDTH = "bandwidth"      # Network transfer costs
    MODEL_INFERENCE = "model"    # ML model inference costs
    CACHE = "cache"             # Cache storage costs

@dataclass
class CostConfig:
    """Cost configuration for different resources"""
    compute_cost_per_hour: float = 0.50      # Cost per hour of compute
    storage_cost_per_gb_month: float = 0.02  # Cost per GB per month
    bandwidth_cost_per_gb: float = 0.05      # Cost per GB transferred
    model_cost_per_1k: float = 0.10          # Cost per 1000 model inferences
    cache_cost_per_gb_hour: float = 0.01     # Cost per GB-hour of cache

@dataclass
class ResourceUsage:
    """Tracked resource usage"""
    compute_hours: float = 0.0
    storage_gb: float = 0.0
    bandwidth_gb: float = 0.0
    model_inferences: int = 0
    cache_gb_hours: float = 0.0

class CostBreakdown:
    """Detailed cost breakdown by resource"""
    def __init__(self, usage: ResourceUsage, config: CostConfig):
        self.compute_cost = usage.compute_hours * config.compute_cost_per_hour
        self.storage_cost = usage.storage_gb * config.storage_cost_per_gb_month
        self.bandwidth_cost = usage.bandwidth_gb * config.bandwidth_cost_per_gb
        self.model_cost = (usage.model_inferences / 1000) * config.model_cost_per_1k
        self.cache_cost = usage.cache_gb_hours * config.cache_cost_per_gb_hour
        self.total_cost = (self.compute_cost + self.storage_cost + 
                          self.bandwidth_cost + self.model_cost + self.cache_cost)

    def to_dict(self) -> Dict[str, float]:
        return {
            'compute_cost': self.compute_cost,
            'storage_cost': self.storage_cost,
            'bandwidth_cost': self.bandwidth_cost,
            'model_cost': self.model_cost,
            'cache_cost': self.cache_cost,
            'total_cost': self.total_cost
        }

class CostMonitor:
    """Monitors and tracks resource usage and costs"""
    
    def __init__(
        self,
        config: Optional[CostConfig] = None,
        history_file: Optional[Path] = None
    ):
        self.config = config or CostConfig()
        self.history_file = history_file
        self.current_usage = ResourceUsage()
        self.logger = logging.getLogger(__name__)
        
        # Track usage by video ID
        self.video_usage: Dict[str, ResourceUsage] = {}
        
        # Load historical data if available
        self._load_history()
        
    def _load_history(self):
        """Load historical cost data"""
        if self.history_file and self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    self.cost_history = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading cost history: {str(e)}")
                self.cost_history = []
        else:
            self.cost_history = []
            
    def track_resource_usage(
        self,
        resource_type: ResourceType,
        amount: float,
        video_id: Optional[str] = None
    ):
        """Track usage of a specific resource"""
        # Initialize video usage if needed
        if video_id and video_id not in self.video_usage:
            self.video_usage[video_id] = ResourceUsage()
            
        # Update usage based on resource type
        if resource_type == ResourceType.COMPUTE:
            self.current_usage.compute_hours += amount
            if video_id:
                self.video_usage[video_id].compute_hours += amount
                
        elif resource_type == ResourceType.STORAGE:
            self.current_usage.storage_gb += amount
            if video_id:
                self.video_usage[video_id].storage_gb += amount
                
        elif resource_type == ResourceType.BANDWIDTH:
            self.current_usage.bandwidth_gb += amount
            if video_id:
                self.video_usage[video_id].bandwidth_gb += amount
                
        elif resource_type == ResourceType.MODEL_INFERENCE:
            self.current_usage.model_inferences += int(amount)
            if video_id:
                self.video_usage[video_id].model_inferences += int(amount)
                
        elif resource_type == ResourceType.CACHE:
            self.current_usage.cache_gb_hours += amount
            if video_id:
                self.video_usage[video_id].cache_gb_hours += amount
                
    def get_current_costs(self) -> CostBreakdown:
        """Get current cost breakdown"""
        return CostBreakdown(self.current_usage, self.config)
        
    def get_video_costs(self, video_id: str) -> Optional[CostBreakdown]:
        """Get cost breakdown for specific video"""
        if video_id in self.video_usage:
            return CostBreakdown(self.video_usage[video_id], self.config)
        return None
        
    def save_cost_snapshot(self, metadata: Optional[Dict[str, Any]] = None):
        """Save current cost snapshot to history"""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'costs': self.get_current_costs().to_dict(),
            'video_costs': {
                video_id: CostBreakdown(usage, self.config).to_dict()
                for video_id, usage in self.video_usage.items()
            }
        }
        
        if metadata:
            snapshot['metadata'] = metadata
            
        self.cost_history.append(snapshot)
        
        # Save to file if configured
        if self.history_file:
            try:
                with open(self.history_file, 'w') as f:
                    json.dump(self.cost_history, f, indent=2)
            except Exception as e:
                self.logger.error(f"Error saving cost history: {str(e)}")
                
    def get_cost_summary(
        self,
        days: int = 30,
        video_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get cost summary for specified period"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter relevant snapshots
        snapshots = [
            s for s in self.cost_history
            if datetime.fromisoformat(s['timestamp']) >= cutoff_date
        ]
        
        if not snapshots:
            return {}
            
        # Calculate statistics
        if video_id:
            costs = [
                s['video_costs'].get(video_id, {'total_cost': 0.0})
                for s in snapshots
            ]
        else:
            costs = [s['costs'] for s in snapshots]
            
        return {
            'total_cost': sum(c['total_cost'] for c in costs),
            'avg_daily_cost': sum(c['total_cost'] for c in costs) / days,
            'cost_breakdown': {
                'compute': sum(c.get('compute_cost', 0.0) for c in costs),
                'storage': sum(c.get('storage_cost', 0.0) for c in costs),
                'bandwidth': sum(c.get('bandwidth_cost', 0.0) for c in costs),
                'model': sum(c.get('model_cost', 0.0) for c in costs),
                'cache': sum(c.get('cache_cost', 0.0) for c in costs)
            }
        }
        
    def analyze_cost_trends(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """Analyze cost trends over time"""
        summary = self.get_cost_summary(days)
        if not summary:
            return {}
            
        # Calculate daily costs
        daily_costs = {}
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for snapshot in self.cost_history:
            date = datetime.fromisoformat(snapshot['timestamp']).date()
            if date >= cutoff_date.date():
                if date not in daily_costs:
                    daily_costs[date.isoformat()] = 0.0
                daily_costs[date.isoformat()] += snapshot['costs']['total_cost']
                
        # Calculate trends
        daily_values = list(daily_costs.values())
        if len(daily_values) >= 2:
            cost_trend = (daily_values[-1] - daily_values[0]) / len(daily_values)
        else:
            cost_trend = 0.0
            
        return {
            'summary': summary,
            'daily_costs': daily_costs,
            'cost_trend': cost_trend,
            'projected_monthly_cost': summary['avg_daily_cost'] * 30
        } 
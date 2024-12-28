from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import logging
import time
from datetime import datetime, timedelta

from ...monitoring import CostMonitor, ResourceType, CostConfig

class OptimizationStrategy(Enum):
    """Available cost optimization strategies"""
    ADAPTIVE_QUALITY = "adaptive_quality"      # Adjust quality based on cost
    BATCH_PROCESSING = "batch_processing"      # Optimize batch sizes
    CACHE_OPTIMIZATION = "cache_optimization"  # Optimize cache usage
    MODEL_SELECTION = "model_selection"        # Choose cheaper models when possible
    RESOURCE_SCALING = "resource_scaling"      # Scale resources based on cost

@dataclass
class OptimizationConfig:
    """Configuration for cost optimization"""
    target_cost_per_minute: float = 0.50      # Target processing cost per minute
    max_cost_per_video: float = 10.0          # Maximum cost per video
    quality_cost_ratio: float = 0.8           # Quality vs cost balance (0-1)
    minimum_quality: float = 0.6              # Minimum acceptable quality
    cache_cost_threshold: float = 0.05        # Cost threshold for caching
    enabled_strategies: List[OptimizationStrategy] = None

class ProcessingProfile:
    """Processing parameters that can be optimized"""
    def __init__(
        self,
        frame_sample_rate: int = 1,
        batch_size: int = 32,
        model_quality: str = "high",
        cache_policy: str = "all",
        target_fps: Optional[int] = None
    ):
        self.frame_sample_rate = frame_sample_rate
        self.batch_size = batch_size
        self.model_quality = model_quality
        self.cache_policy = cache_policy
        self.target_fps = target_fps

    def to_dict(self) -> Dict[str, Any]:
        return {
            'frame_sample_rate': self.frame_sample_rate,
            'batch_size': self.batch_size,
            'model_quality': self.model_quality,
            'cache_policy': self.cache_policy,
            'target_fps': self.target_fps
        }

class CostOptimizer:
    """Optimizes processing parameters based on cost analysis"""
    
    def __init__(
        self,
        config: OptimizationConfig,
        cost_monitor: CostMonitor
    ):
        self.config = config
        self.cost_monitor = cost_monitor
        self.logger = logging.getLogger(__name__)
        
        # Initialize optimization history
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Set default strategies if none provided
        if not self.config.enabled_strategies:
            self.config.enabled_strategies = [
                OptimizationStrategy.ADAPTIVE_QUALITY,
                OptimizationStrategy.BATCH_PROCESSING,
                OptimizationStrategy.CACHE_OPTIMIZATION
            ]

    def optimize_processing_params(
        self,
        video_metadata: Dict[str, Any],
        current_costs: Optional[Dict[str, float]] = None
    ) -> ProcessingProfile:
        """Optimize processing parameters based on costs and video metadata"""
        profile = ProcessingProfile()
        
        try:
            # Get current costs if not provided
            if not current_costs:
                current_costs = self.cost_monitor.get_current_costs().to_dict()

            # Apply enabled optimization strategies
            for strategy in self.config.enabled_strategies:
                if strategy == OptimizationStrategy.ADAPTIVE_QUALITY:
                    self._optimize_quality(profile, current_costs, video_metadata)
                elif strategy == OptimizationStrategy.BATCH_PROCESSING:
                    self._optimize_batch_size(profile, current_costs)
                elif strategy == OptimizationStrategy.CACHE_OPTIMIZATION:
                    self._optimize_cache(profile, current_costs)
                elif strategy == OptimizationStrategy.MODEL_SELECTION:
                    self._optimize_model_selection(profile, current_costs)
                elif strategy == OptimizationStrategy.RESOURCE_SCALING:
                    self._optimize_resources(profile, current_costs)

            # Log optimization decision
            self._log_optimization(profile, current_costs, video_metadata)

            return profile

        except Exception as e:
            self.logger.error(f"Error during optimization: {str(e)}")
            return ProcessingProfile()  # Return default profile on error

    def _optimize_quality(
        self,
        profile: ProcessingProfile,
        costs: Dict[str, float],
        metadata: Dict[str, Any]
    ):
        """Optimize quality vs cost tradeoff"""
        # Calculate current cost per minute
        video_duration = metadata.get('duration', 0)
        if video_duration > 0:
            cost_per_minute = costs['total_cost'] / (video_duration / 60)
        else:
            cost_per_minute = float('inf')

        # Adjust frame sampling rate based on cost
        if cost_per_minute > self.config.target_cost_per_minute:
            # Increase sampling rate to reduce cost
            profile.frame_sample_rate = min(
                profile.frame_sample_rate * 2,
                10  # Maximum sampling rate
            )
        elif cost_per_minute < self.config.target_cost_per_minute * 0.5:
            # Decrease sampling rate to improve quality
            profile.frame_sample_rate = max(
                profile.frame_sample_rate // 2,
                1  # Minimum sampling rate
            )

        # Adjust target FPS based on content
        if 'average_motion' in metadata:
            if metadata['average_motion'] > 0.8:
                profile.target_fps = max(30, profile.target_fps or 0)
            elif metadata['average_motion'] < 0.2:
                profile.target_fps = min(15, profile.target_fps or float('inf'))

    def _optimize_batch_size(
        self,
        profile: ProcessingProfile,
        costs: Dict[str, float]
    ):
        """Optimize batch processing parameters"""
        # Check if compute costs are too high
        if costs['compute_cost'] > costs['total_cost'] * 0.7:
            # Increase batch size to improve throughput
            profile.batch_size = min(
                profile.batch_size * 2,
                128  # Maximum batch size
            )
        elif costs['memory_cost'] > costs['total_cost'] * 0.3:
            # Decrease batch size to reduce memory usage
            profile.batch_size = max(
                profile.batch_size // 2,
                8  # Minimum batch size
            )

    def _optimize_cache(
        self,
        profile: ProcessingProfile,
        costs: Dict[str, float]
    ):
        """Optimize cache usage"""
        cache_cost_ratio = costs['cache_cost'] / costs['total_cost']
        
        if cache_cost_ratio > self.config.cache_cost_threshold:
            # Reduce caching
            if profile.cache_policy == "all":
                profile.cache_policy = "selective"
            elif profile.cache_policy == "selective":
                profile.cache_policy = "minimal"
        elif cache_cost_ratio < self.config.cache_cost_threshold * 0.5:
            # Increase caching if beneficial
            if profile.cache_policy == "minimal":
                profile.cache_policy = "selective"
            elif profile.cache_policy == "selective":
                profile.cache_policy = "all"

    def _optimize_model_selection(
        self,
        profile: ProcessingProfile,
        costs: Dict[str, float]
    ):
        """Optimize model selection based on costs"""
        model_cost_ratio = costs['model_cost'] / costs['total_cost']
        
        # Adjust model quality based on cost ratio
        if model_cost_ratio > 0.4:  # Models are too expensive
            if profile.model_quality == "high":
                profile.model_quality = "medium"
            elif profile.model_quality == "medium":
                profile.model_quality = "low"
        elif model_cost_ratio < 0.2:  # Can afford better models
            if profile.model_quality == "low":
                profile.model_quality = "medium"
            elif profile.model_quality == "medium":
                profile.model_quality = "high"

    def _optimize_resources(
        self,
        profile: ProcessingProfile,
        costs: Dict[str, float]
    ):
        """Optimize resource allocation"""
        # Calculate cost efficiency
        total_cost = costs['total_cost']
        compute_ratio = costs['compute_cost'] / total_cost
        memory_ratio = costs.get('cache_cost', 0) / total_cost

        # Adjust batch size based on resource efficiency
        if compute_ratio > 0.6:  # Compute-bound
            profile.batch_size = max(profile.batch_size // 2, 8)
        elif memory_ratio > 0.3:  # Memory-bound
            profile.batch_size = min(profile.batch_size * 2, 128)

    def _log_optimization(
        self,
        profile: ProcessingProfile,
        costs: Dict[str, float],
        metadata: Dict[str, Any]
    ):
        """Log optimization decision"""
        optimization_record = {
            'timestamp': datetime.now().isoformat(),
            'profile': profile.to_dict(),
            'costs': costs,
            'metadata': metadata
        }
        
        self.optimization_history.append(optimization_record)
        
        self.logger.info(
            f"Optimized processing profile: {profile.to_dict()}, "
            f"Current costs: {costs}"
        )

    def analyze_optimization_impact(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """Analyze impact of optimization strategies"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter relevant history
        recent_history = [
            record for record in self.optimization_history
            if datetime.fromisoformat(record['timestamp']) >= cutoff_date
        ]
        
        if not recent_history:
            return {}

        # Calculate optimization impact
        cost_savings = []
        quality_impact = []
        
        for i in range(1, len(recent_history)):
            prev = recent_history[i-1]
            curr = recent_history[i]
            
            # Calculate cost difference
            cost_diff = prev['costs']['total_cost'] - curr['costs']['total_cost']
            cost_savings.append(cost_diff)
            
            # Estimate quality impact based on profile changes
            quality_change = self._estimate_quality_change(
                prev['profile'],
                curr['profile']
            )
            quality_impact.append(quality_change)

        return {
            'total_cost_savings': sum(cost_savings),
            'average_cost_reduction': sum(cost_savings) / len(cost_savings) if cost_savings else 0,
            'quality_impact': sum(quality_impact) / len(quality_impact) if quality_impact else 0,
            'optimization_count': len(recent_history),
            'most_common_changes': self._analyze_common_changes(recent_history)
        }

    def _estimate_quality_change(
        self,
        prev_profile: Dict[str, Any],
        curr_profile: Dict[str, Any]
    ) -> float:
        """Estimate quality impact of profile changes"""
        quality_change = 0.0
        
        # Frame sampling impact
        if curr_profile['frame_sample_rate'] != prev_profile['frame_sample_rate']:
            quality_change -= 0.1 * (
                curr_profile['frame_sample_rate'] - prev_profile['frame_sample_rate']
            )
            
        # Model quality impact
        quality_levels = {'high': 1.0, 'medium': 0.7, 'low': 0.4}
        if curr_profile['model_quality'] != prev_profile['model_quality']:
            quality_change += quality_levels[curr_profile['model_quality']] - \
                            quality_levels[prev_profile['model_quality']]
            
        return quality_change

    def _analyze_common_changes(
        self,
        history: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Analyze most common optimization changes"""
        changes = {
            'frame_rate_changes': 0,
            'batch_size_changes': 0,
            'model_quality_changes': 0,
            'cache_policy_changes': 0
        }
        
        for i in range(1, len(history)):
            prev = history[i-1]['profile']
            curr = history[i]['profile']
            
            if curr['frame_sample_rate'] != prev['frame_sample_rate']:
                changes['frame_rate_changes'] += 1
            if curr['batch_size'] != prev['batch_size']:
                changes['batch_size_changes'] += 1
            if curr['model_quality'] != prev['model_quality']:
                changes['model_quality_changes'] += 1
            if curr['cache_policy'] != prev['cache_policy']:
                changes['cache_policy_changes'] += 1
                
        return changes

def create_cost_optimizer(
    cost_monitor: CostMonitor,
    config: Optional[Dict[str, Any]] = None
) -> CostOptimizer:
    """Create cost optimizer instance"""
    optimization_config = OptimizationConfig(
        target_cost_per_minute=config.get('target_cost_per_minute', 0.50),
        max_cost_per_video=config.get('max_cost_per_video', 10.0),
        quality_cost_ratio=config.get('quality_cost_ratio', 0.8),
        minimum_quality=config.get('minimum_quality', 0.6),
        cache_cost_threshold=config.get('cache_cost_threshold', 0.05)
    )
    
    return CostOptimizer(optimization_config, cost_monitor) 
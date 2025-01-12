from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

@dataclass
class ProcessingMetrics:
    """Metrics for individual video processing tasks"""
    video_id: str
    start_time: float
    end_time: Optional[float] = None
    frames_processed: int = 0
    frames_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_memory_mb: Optional[float] = None
    scene_count: int = 0
    object_count: int = 0
    processing_errors: List[str] = field(default_factory=list)

@dataclass
class SystemMetrics:
    """System-wide performance metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    gpu_utilization: Optional[float] = None
    gpu_memory_used: Optional[float] = None
    disk_usage_percent: float = 0.0
    active_processes: int = 0 
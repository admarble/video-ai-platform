from dataclasses import dataclass, field
from typing import Callable, Awaitable, Any, Dict
from .priority import CleanupPriority
import time

@dataclass
class TaskMetrics:
    """Metrics for cleanup task execution"""
    total_runs: int = 0
    total_failures: int = 0
    total_bytes_cleaned: int = 0
    last_execution_time: float = 0
    avg_execution_time: float = 0

@dataclass
class CleanupTask:
    """Represents a cleanup task"""
    name: str
    priority: CleanupPriority
    interval: int  # seconds
    func: Callable[..., Awaitable[Any]]
    last_run: float = 0
    enabled: bool = True
    retry_count: int = 0
    max_retries: int = 3
    progress: float = 0  # 0-100
    metrics: TaskMetrics = field(default_factory=TaskMetrics)
    _is_running: bool = False

    def __lt__(self, other):
        if not isinstance(other, CleanupTask):
            return NotImplemented
        return self.priority < other.priority

    def update_metrics(self, execution_time: float, bytes_cleaned: int = 0, failed: bool = False):
        """Update task metrics after execution"""
        self.metrics.total_runs += 1
        if failed:
            self.metrics.total_failures += 1
        
        self.metrics.total_bytes_cleaned += bytes_cleaned
        self.metrics.last_execution_time = execution_time
        
        # Update running average execution time
        self.metrics.avg_execution_time = (
            (self.metrics.avg_execution_time * (self.metrics.total_runs - 1) + execution_time)
            / self.metrics.total_runs
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get task metrics"""
        return {
            'total_runs': self.metrics.total_runs,
            'total_failures': self.metrics.total_failures,
            'total_bytes_cleaned': self.metrics.total_bytes_cleaned,
            'last_execution_time': self.metrics.last_execution_time,
            'avg_execution_time': self.metrics.avg_execution_time,
            'progress': self.progress,
            'is_running': self._is_running
        } 
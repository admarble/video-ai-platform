from pathlib import Path
from typing import Dict, Any, Optional
import time
import json
import threading
from datetime import datetime, timedelta
import psutil
import torch
import logging
from .logging_manager import ComponentLogger
from .alert_manager import AlertManager, AlertSeverity, AlertChannel
from .monitoring_system import MonitoringSystem, MonitoringMetric

class ProcessingMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.frames_processed = 0
        self.scene_count = 0
        self.object_count = 0
        self.memory_usage_mb = 0
        self.gpu_memory_mb = 0
        self.error = None
        self.completed = False

class MetricsCollector:
    def __init__(self, config):
        self.config = config
        self.logger = ComponentLogger("metrics_collector")
        self.metrics_dir = Path(config.get('metrics_dir', '.'))
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.current_metrics: Dict[str, ProcessingMetrics] = {}
        self.lock = threading.Lock()
        self.alert_manager = AlertManager(
            self,  # MetricsCollector acts as log_aggregator
            config.get('alert_config', {}),
            Path(config.get('alert_history_path', 'alert_history.json'))
        )
        self.monitoring = MonitoringSystem(
            self,  # MetricsCollector acts as log_aggregator
            config.get('monitoring_config', {}),
            alert_history_file=str(Path(config.get('alert_history_path', 'data/alert_history.json')))
        )

    def start_processing_metrics(self, video_id: str) -> None:
        with self.lock:
            self.current_metrics[video_id] = ProcessingMetrics()
            self._save_metrics(video_id)

    def update_processing_metrics(self, video_id: str, **kwargs) -> None:
        with self.lock:
            if video_id not in self.current_metrics:
                return

            metrics = self.current_metrics[video_id]
            for key, value in kwargs.items():
                setattr(metrics, key, value)

            # Update system metrics
            metrics.memory_usage_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            if torch.cuda.is_available():
                metrics.gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)

            self._save_metrics(video_id)

    def complete_processing_metrics(self, video_id: str) -> None:
        with self.lock:
            if video_id in self.current_metrics:
                self.current_metrics[video_id].completed = True
                self._save_metrics(video_id)
                del self.current_metrics[video_id]

    def get_system_metrics_summary(self, duration: Optional[timedelta] = None) -> Dict[str, Any]:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        metrics = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'disk_percent': disk.percent,
            'memory_available_mb': memory.available / (1024 * 1024),
            'disk_free_mb': disk.free / (1024 * 1024)
        }

        if torch.cuda.is_available():
            metrics['gpu_memory_used_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
            metrics['gpu_memory_cached_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)

        return metrics

    def _save_metrics(self, video_id: str) -> None:
        metrics = self.current_metrics[video_id]
        metrics_file = self.metrics_dir / f"{video_id}_{int(metrics.start_time)}.json"
        
        data = {
            'start_time': metrics.start_time,
            'frames_processed': metrics.frames_processed,
            'scene_count': metrics.scene_count,
            'object_count': metrics.object_count,
            'memory_usage_mb': metrics.memory_usage_mb,
            'gpu_memory_mb': metrics.gpu_memory_mb,
            'error': metrics.error,
            'completed': metrics.completed,
            'duration': time.time() - metrics.start_time
        }

        with open(metrics_file, 'w') as f:
            json.dump(data, f)

    def cleanup_old_metrics(self, max_age_days: int = 7) -> None:
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        
        for metrics_file in self.metrics_dir.glob("*.json"):
            try:
                if metrics_file.stat().st_mtime < cutoff_time:
                    metrics_file.unlink()
            except Exception as e:
                self.logger.error(f"Error cleaning up metrics file {metrics_file}: {e}") 

    def record_metric(self, metric_name: str, value: float, tags: dict):
        self.logger.info(f"Recording metric: {metric_name}",
                        extra={
                            "metric_name": metric_name,
                            "value": value,
                            "tags": tags
                        })
        # Existing metric recording code...

    def start(self):
        # Start monitoring system
        self.monitoring.start_monitoring()
        
    def stop(self):
        # Cleanup and shutdown monitoring
        if hasattr(self, 'monitoring'):
            self.monitoring.shutdown()
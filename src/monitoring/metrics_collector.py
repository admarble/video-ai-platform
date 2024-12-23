from typing import Dict, Any
import json
import time
from pathlib import Path
import psutil
import logging
from datetime import datetime, timedelta

class MetricsCollector:
    """Collects and stores system metrics"""
    
    def __init__(self, metrics_dir: Path):
        self.metrics_dir = metrics_dir
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'percent': psutil.cpu_percent(interval=1),
                'count': psutil.cpu_count(),
                'freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent
            },
            'disk': {
                'total': psutil.disk_usage('/').total,
                'used': psutil.disk_usage('/').used,
                'free': psutil.disk_usage('/').free,
                'percent': psutil.disk_usage('/').percent
            },
            'network': {
                'bytes_sent': psutil.net_io_counters().bytes_sent,
                'bytes_recv': psutil.net_io_counters().bytes_recv
            }
        }
        return metrics
        
    def update_processing_metrics(self, video_id: str, **metrics: Any) -> None:
        """Update metrics for a specific video processing job"""
        metrics_file = self.metrics_dir / f"{video_id}_metrics.json"
        
        current_metrics = {
            'timestamp': datetime.now().isoformat(),
            'video_id': video_id,
            **metrics
        }
        
        try:
            if metrics_file.exists():
                with metrics_file.open('r') as f:
                    history = json.load(f)
                if not isinstance(history, list):
                    history = [history]
            else:
                history = []
                
            history.append(current_metrics)
            
            with metrics_file.open('w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to update metrics for video {video_id}: {str(e)}")
            raise
            
    def cleanup_old_metrics(self, max_age_days: int = 30) -> None:
        """Remove metrics files older than specified days"""
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        
        try:
            for metrics_file in self.metrics_dir.glob("*_metrics.json"):
                if metrics_file.stat().st_mtime < cutoff_time.timestamp():
                    metrics_file.unlink()
                    self.logger.info(f"Removed old metrics file: {metrics_file}")
                    
        except Exception as e:
            self.logger.error(f"Error during metrics cleanup: {str(e)}")
            raise
            
    def is_healthy(self) -> bool:
        """Check if metrics collector is functioning properly"""
        try:
            # Try to collect metrics and write to a test file
            metrics = self.collect_system_metrics()
            test_file = self.metrics_dir / "health_check.json"
            
            with test_file.open('w') as f:
                json.dump(metrics, f)
                
            test_file.unlink()  # Clean up test file
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False 
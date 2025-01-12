import time
import threading
import logging
import psutil
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, Optional, Any
import torch

from .metrics import ProcessingMetrics, SystemMetrics

class MetricsCollector:
    """Collects and manages system and processing metrics"""
    
    def __init__(self, metrics_dir: Path, retention_days: int = 7):
        self.metrics_dir = metrics_dir
        self.retention_days = retention_days
        self.current_metrics: Dict[str, ProcessingMetrics] = {}
        self.system_metrics: deque = deque(maxlen=1000)
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self._start_system_monitoring()

    # Copy all the methods from your original implementation here
    # The implementation remains the same, just moved to this file

def setup_monitoring(base_dir: Path) -> MetricsCollector:
    """Setup monitoring system"""
    metrics_dir = base_dir / "metrics"
    collector = MetricsCollector(metrics_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return collector 
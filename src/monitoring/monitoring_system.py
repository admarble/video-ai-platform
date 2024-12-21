from typing import Dict, Any, List, Optional, Callable
import time
from datetime import datetime, timedelta
import threading
import queue
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from dataclasses import dataclass
from enum import Enum
import logging
import psutil
import torch
import os
from pathlib import Path

class MonitoringMetric(Enum):
    """Types of metrics to monitor"""
    ERROR_RATE = "error_rate"
    PROCESSING_TIME = "processing_time"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    GPU_USAGE = "gpu_usage"
    QUEUE_SIZE = "queue_size"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertChannel(Enum):
    """Available alert notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"

@dataclass
class Alert:
    """Represents an alert that was triggered"""
    metric: MonitoringMetric
    severity: AlertSeverity
    message: str
    timestamp: float
    value: float
    threshold: float
    details: Optional[Dict[str, Any]] = None

@dataclass
class AlertRule:
    """Defines conditions for triggering alerts"""
    metric: MonitoringMetric
    severity: AlertSeverity
    threshold: float
    window_minutes: int = 5
    cooldown_minutes: int = 15
    channels: List[AlertChannel] = None
    last_triggered: Optional[float] = None

class MonitoringSystem:
    """Monitoring and alerting system for video processing platform"""
    
    def __init__(
        self,
        log_aggregator,
        config: Dict[str, Any],
        alert_history_file: Optional[str] = None
    ):
        self.log_aggregator = log_aggregator
        self.config = config
        self.alert_history_file = alert_history_file
        
        # Initialize metrics storage
        self.metrics: Dict[str, List[tuple]] = {
            metric.value: [] for metric in MonitoringMetric
        }
        
        # Setup alert rules
        self.alert_rules: Dict[MonitoringMetric, List[AlertRule]] = {}
        self._setup_default_rules()
        
        # Alert queue for async processing
        self.alert_queue = queue.Queue()
        
        # Start alert processor thread
        self._start_alert_processor()
        
        self.logger = logging.getLogger(__name__)

    # ... (rest of the implementation as provided) 
from typing import Dict, Any, List, Optional, Callable
import time
from datetime import datetime
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
from pathlib import Path

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
class AlertRule:
    """Defines an alert rule"""
    name: str
    description: str
    severity: AlertSeverity
    condition: Callable
    channels: List[AlertChannel]
    cooldown_minutes: int = 60
    last_triggered: Optional[float] = None

@dataclass
class Alert:
    """Represents a triggered alert"""
    rule_name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    details: Optional[Dict[str, Any]] = None

class AlertManager:
    """Manages alert rules and notifications"""
    
    def __init__(
        self,
        log_aggregator,
        config: Dict[str, Any],
        alert_history_path: Optional[Path] = None
    ):
        self.log_aggregator = log_aggregator
        self.config = config
        self.alert_history_path = alert_history_path
        self.rules: Dict[str, AlertRule] = {}
        self.alert_queue = queue.Queue()
        self.logger = logging.getLogger(__name__)
        self._start_alert_processor()
        self._init_default_rules()

    # ... (rest of the implementation as provided) 
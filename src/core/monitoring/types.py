from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

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
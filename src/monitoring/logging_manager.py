from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import logging.handlers
import queue
import threading
import time
import json
from datetime import datetime
import os
import sqlite3
from dataclasses import dataclass
from enum import Enum
import traceback

class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

@dataclass
class LogEntry:
    """Represents a structured log entry"""
    timestamp: float
    level: LogLevel
    component: str
    message: str
    video_id: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None
    exception: Optional[str] = None

# [Previous LogAggregator and ComponentLogger classes implementation...] 
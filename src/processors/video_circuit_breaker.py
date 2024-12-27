"""Video-specific circuit breaker implementation."""

import time
import logging
import threading
import asyncio
from enum import Enum
from typing import Dict, Any, Optional, Callable, Union
from datetime import datetime
import psutil
try:
    import torch
except ImportError:
    torch = None

from .video_exceptions import (
    CircuitOpenError,
    VideoProcessingError,
    VideoCorruptedError,
    ResourceExhaustedError,
    ModelError,
    ProcessingTimeoutError
)
from .video_circuit_config import CircuitConfig, VideoCircuitConfig

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation - requests allowed
    OPEN = "open"         # Failure threshold exceeded - requests blocked
    HALF_OPEN = "half_open"  # Testing if service is restored

class CircuitBreaker:
    """Base circuit breaker implementation."""
    
    def __init__(self, name: str, config: CircuitConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_state_change = time.time()
        self.half_open_count = 0
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        self.failure_timestamps = []

    def can_execute(self) -> bool:
        """Check if request can be executed."""
        with self.lock:
            current_time = time.time()
            self._clean_failure_window(current_time)

            if self.state == CircuitState.OPEN:
                if current_time - self.last_state_change >= self.config.reset_timeout:
                    self._transition_to_half_open()
                    return True
                return False
                
            if self.state == CircuitState.HALF_OPEN:
                return self.half_open_count < self.config.half_open_limit
                
            return True

    def record_success(self):
        """Record successful execution."""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
                else:
                    self.half_open_count += 1

    def record_failure(self):
        """Record failed execution."""
        with self.lock:
            current_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self._transition_to_open(current_time)
                return

            self.failure_timestamps.append(current_time)
            self._clean_failure_window(current_time)

            if len(self.failure_timestamps) >= self.config.failure_threshold:
                self._transition_to_open(current_time)

    def _clean_failure_window(self, current_time: float):
        """Remove failures outside the window."""
        window_start = current_time - self.config.window_size
        self.failure_timestamps = [
            t for t in self.failure_timestamps if t >= window_start
        ]

    def _transition_to_open(self, current_time: float):
        """Transition to open state."""
        self.state = CircuitState.OPEN
        self.last_state_change = current_time
        self.logger.warning(
            f"Circuit {self.name} opened after {len(self.failure_timestamps)} failures"
        )

    def _transition_to_half_open(self):
        """Transition to half-open state."""
        self.state = CircuitState.HALF_OPEN
        self.last_state_change = time.time()
        self.success_count = 0
        self.half_open_count = 0
        self.logger.info(f"Circuit {self.name} entering half-open state")

    def _transition_to_closed(self):
        """Transition to closed state."""
        self.state = CircuitState.CLOSED
        self.last_state_change = time.time()
        self.failure_timestamps = []
        self.success_count = 0
        self.half_open_count = 0
        self.logger.info(f"Circuit {self.name} closed after successful tests")

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        with self.lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': len(self.failure_timestamps),
                'success_count': self.success_count,
                'half_open_count': self.half_open_count,
                'last_failure': datetime.fromtimestamp(self.last_failure_time).isoformat() if self.last_failure_time else None,
                'last_state_change': datetime.fromtimestamp(self.last_state_change).isoformat()
            }

class VideoCircuitBreaker(CircuitBreaker):
    """Circuit breaker specialized for video processing."""
    
    def __init__(self, name: str, config: VideoCircuitConfig):
        super().__init__(name, config)
        self.corruption_count = 0
        self.model_error_count = 0
        self.last_resource_check = 0
        self.config: VideoCircuitConfig = config

    def handle_error(self, error: Exception) -> bool:
        """Handle specific video processing errors."""
        with self.lock:
            current_time = time.time()
            
            if isinstance(error, VideoCorruptedError):
                self.corruption_count += 1
                if self.corruption_count >= self.config.corruption_threshold:
                    self._transition_to_open(current_time)
                    return False
                    
            elif isinstance(error, ModelError):
                self.model_error_count += 1
                if self.model_error_count >= self.config.model_error_threshold:
                    self._transition_to_open(current_time)
                    return False
                    
            elif isinstance(error, ResourceExhaustedError):
                self._transition_to_open(current_time)
                return False
                
            self.record_failure()
            return True

    def check_resources(self) -> bool:
        """Check system resources before processing."""
        current_time = time.time()
        
        if current_time - self.last_resource_check < self.config.resource_check_interval:
            return True
            
        self.last_resource_check = current_time
        
        try:
            memory = psutil.virtual_memory()
            free_memory_mb = memory.available / (1024 * 1024)
            
            if free_memory_mb < self.config.min_free_memory:
                self.logger.warning(f"Insufficient memory: {free_memory_mb:.0f}MB available")
                return False
                
            if torch is not None and torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                if gpu_memory > torch.cuda.get_device_properties(0).total_memory * 0.9:
                    self.logger.warning("GPU memory nearly exhausted")
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking resources: {str(e)}")
            return True

    def reset_error_counts(self):
        """Reset error counters when closing circuit."""
        with self.lock:
            self.corruption_count = 0
            self.model_error_count = 0

    def _transition_to_closed(self):
        """Override to reset error counts."""
        super()._transition_to_closed()
        self.reset_error_counts()

    def get_state(self) -> Dict[str, Any]:
        """Get enhanced state information."""
        state = super().get_state()
        state.update({
            'corruption_count': self.corruption_count,
            'model_error_count': self.model_error_count,
            'last_resource_check': datetime.fromtimestamp(self.last_resource_check).isoformat() if self.last_resource_check else None
        })
        return state 
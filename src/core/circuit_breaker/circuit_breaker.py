from enum import Enum
from typing import Dict, Any, Optional, Callable, Union
import time
import logging
import threading
from dataclasses import dataclass
import asyncio
from datetime import datetime, timedelta

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation - requests allowed
    OPEN = "open"         # Failure threshold exceeded - requests blocked
    HALF_OPEN = "half_open"  # Testing if service is restored

@dataclass
class CircuitConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5        # Number of failures before opening
    reset_timeout: int = 60           # Seconds before attempting reset
    half_open_limit: int = 3         # Number of requests to test in half-open state
    window_size: int = 60            # Time window for failure counting (seconds)
    success_threshold: int = 2       # Successes needed to close circuit
    exceptions: tuple = (Exception,)  # Exception types to count as failures

class CircuitBreaker:
    """Implements circuit breaker pattern for video processing"""
    
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
        """Check if request can be executed"""
        with self.lock:
            current_time = time.time()

            # Clean up old failure timestamps
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
        """Record successful execution"""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
                else:
                    self.half_open_count += 1

    def record_failure(self):
        """Record failed execution"""
        with self.lock:
            current_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self._transition_to_open(current_time)
                return

            # Add failure timestamp
            self.failure_timestamps.append(current_time)
            self._clean_failure_window(current_time)

            # Check if threshold is exceeded
            if len(self.failure_timestamps) >= self.config.failure_threshold:
                self._transition_to_open(current_time)

    def _clean_failure_window(self, current_time: float):
        """Remove failures outside the window"""
        window_start = current_time - self.config.window_size
        self.failure_timestamps = [
            t for t in self.failure_timestamps if t >= window_start
        ]

    def _transition_to_open(self, current_time: float):
        """Transition to open state"""
        self.state = CircuitState.OPEN
        self.last_state_change = current_time
        self.logger.warning(
            f"Circuit {self.name} opened after {len(self.failure_timestamps)} failures"
        )

    def _transition_to_half_open(self):
        """Transition to half-open state"""
        self.state = CircuitState.HALF_OPEN
        self.last_state_change = time.time()
        self.success_count = 0
        self.half_open_count = 0
        self.logger.info(f"Circuit {self.name} entering half-open state")

    def _transition_to_closed(self):
        """Transition to closed state"""
        self.state = CircuitState.CLOSED
        self.last_state_change = time.time()
        self.failure_timestamps = []
        self.success_count = 0
        self.half_open_count = 0
        self.logger.info(f"Circuit {self.name} closed after successful tests")

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
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

class CircuitBreakerRegistry:
    """Manages multiple circuit breakers"""
    
    def __init__(self):
        self.circuits: Dict[str, CircuitBreaker] = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

    def get_circuit(
        self,
        name: str,
        config: Optional[CircuitConfig] = None
    ) -> CircuitBreaker:
        """Get or create circuit breaker"""
        with self.lock:
            if name not in self.circuits:
                self.circuits[name] = CircuitBreaker(
                    name,
                    config or CircuitConfig()
                )
            return self.circuits[name]

    def remove_circuit(self, name: str):
        """Remove circuit breaker"""
        with self.lock:
            if name in self.circuits:
                del self.circuits[name]

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states of all circuit breakers"""
        with self.lock:
            return {
                name: circuit.get_state()
                for name, circuit in self.circuits.items()
            }

def circuit_breaker(
    circuit_name: str,
    registry: CircuitBreakerRegistry,
    config: Optional[CircuitConfig] = None,
    fallback: Optional[Callable] = None
):
    """Decorator for adding circuit breaker to functions"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            circuit = registry.get_circuit(circuit_name, config)
            
            if not circuit.can_execute():
                if fallback:
                    return await fallback(*args, **kwargs)
                raise CircuitOpenError(f"Circuit {circuit_name} is open")
                
            try:
                result = await func(*args, **kwargs)
                circuit.record_success()
                return result
            except Exception as e:
                circuit.record_failure()
                raise

        def sync_wrapper(*args, **kwargs):
            circuit = registry.get_circuit(circuit_name, config)
            
            if not circuit.can_execute():
                if fallback:
                    return fallback(*args, **kwargs)
                raise CircuitOpenError(f"Circuit {circuit_name} is open")
                
            try:
                result = func(*args, **kwargs)
                circuit.record_success()
                return result
            except Exception as e:
                circuit.record_failure()
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

class CircuitOpenError(Exception):
    """Raised when circuit is open"""
    pass 
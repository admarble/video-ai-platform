"""Decorators for video circuit breaker pattern."""

import time
import asyncio
from typing import Optional, Callable, Dict, Any, Union
from functools import wraps

from .video_exceptions import (
    CircuitOpenError,
    ResourceExhaustedError,
    ProcessingTimeoutError
)
from .video_circuit_config import VideoCircuitConfig
from .video_circuit_breaker import VideoCircuitBreaker

class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    _instance = None
    _circuits: Dict[str, VideoCircuitBreaker] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CircuitBreakerRegistry, cls).__new__(cls)
        return cls._instance

    def get_circuit(
        self,
        name: str,
        config: Optional[VideoCircuitConfig] = None
    ) -> VideoCircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self._circuits:
            self._circuits[name] = VideoCircuitBreaker(
                name,
                config or VideoCircuitConfig()
            )
        return self._circuits[name]

    def remove_circuit(self, name: str):
        """Remove a circuit breaker."""
        if name in self._circuits:
            del self._circuits[name]

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states of all circuit breakers."""
        return {
            name: circuit.get_state()
            for name, circuit in self._circuits.items()
        }

def video_circuit_breaker(
    circuit_name: str,
    registry: Optional[CircuitBreakerRegistry] = None,
    config: Optional[VideoCircuitConfig] = None,
    fallback: Optional[Callable] = None
):
    """Decorator for adding circuit breaker to video processing functions."""
    
    if registry is None:
        registry = CircuitBreakerRegistry()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            circuit = registry.get_circuit(circuit_name, config)
            
            if not circuit.can_execute():
                if fallback:
                    return await fallback(*args, **kwargs)
                raise CircuitOpenError(f"Circuit {circuit_name} is open")
                
            if not circuit.check_resources():
                if fallback:
                    return await fallback(*args, **kwargs)
                raise ResourceExhaustedError("Insufficient system resources")
                
            try:
                start_time = time.time()
                result = await func(*args, **kwargs)
                
                if time.time() - start_time > circuit.config.max_processing_time:
                    circuit.record_failure()
                    raise ProcessingTimeoutError("Video processing timeout")
                    
                circuit.record_success()
                return result
                
            except Exception as e:
                circuit.handle_error(e)
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            circuit = registry.get_circuit(circuit_name, config)
            
            if not circuit.can_execute():
                if fallback:
                    return fallback(*args, **kwargs)
                raise CircuitOpenError(f"Circuit {circuit_name} is open")
                
            if not circuit.check_resources():
                if fallback:
                    return fallback(*args, **kwargs)
                raise ResourceExhaustedError("Insufficient system resources")
                
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                
                if time.time() - start_time > circuit.config.max_processing_time:
                    circuit.record_failure()
                    raise ProcessingTimeoutError("Video processing timeout")
                    
                circuit.record_success()
                return result
                
            except Exception as e:
                circuit.handle_error(e)
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator 
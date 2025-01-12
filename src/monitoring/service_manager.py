from typing import Dict, Any, List, Optional
from pathlib import Path
from .logging_manager import setup_logging, ComponentLogger
from .utils import setup_video_processing_with_tracing

class ServiceManager:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.base_dir = Path(config.get('base_dir', '.'))
        
        # Initialize logging
        self.log_aggregator = setup_logging(self.base_dir)
        self.logger = ComponentLogger("service_manager")
        
        self.services: Dict[str, Any] = {}

    def initialize_services(self, service_names: List[str]) -> None:
        for service_name in service_names:
            try:
                service_config = self.config.get(service_name, {})
                self.services[service_name] = self._create_service(service_name, service_config)
            except Exception as e:
                self.logger.error(f"Failed to initialize service {service_name}: {e}")
                raise

    def get_service(self, service_name: str) -> Any:
        if service_name not in self.services:
            raise KeyError(f"Service {service_name} not initialized")
        return self.services[service_name]

    def cleanup(self) -> None:
        for service in self.services.values():
            if hasattr(service, 'cleanup'):
                try:
                    service.cleanup()
                except Exception as e:
                    self.logger.error(f"Error cleaning up service: {e}")
        self.services.clear()

    def _create_service(self, service_name: str, config: Dict[str, Any]) -> Any:
        # Implementation would depend on your specific services
        # This is a placeholder for service initialization logic
        pass 

    def shutdown(self):
        # Existing shutdown code...
        self.log_aggregator.shutdown()

class VideoProcessingService:
    def __init__(self, config):
        self.config = config
        self.processor = setup_video_processing_with_tracing(
            service_name=config.service_name,
            otlp_endpoint=config.otlp_endpoint
        )
    
    # ... rest of the service implementation
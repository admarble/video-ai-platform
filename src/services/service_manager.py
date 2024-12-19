from src.services.scene_processor import SceneProcessor
from src.services.object_detector import ObjectDetector
from src.exceptions.service_exceptions import ServiceInitializationError

class ServiceManager:
    def __init__(self):
        self._services = {}
        
    def initialize_services(self, service_names):
        for name in service_names:
            if name not in self._services:
                self._services[name] = self._create_service(name)
                
    def _create_service(self, name):
        if name == 'scene_processor':
            return SceneProcessor()
        elif name == 'object_detector':
            return ObjectDetector()
        raise ServiceInitializationError(f"Unknown service: {name}")
        
    def batch_process_parallel(self, items, service_name, method_name, batch_size=2):
        service = self._services.get(service_name)
        if not service:
            raise ServiceInitializationError(f"Service {service_name} not initialized")
            
        method = getattr(service, method_name)
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            results.extend(method(batch))
            
        return results 
import psutil
import logging
import torch

class ResourceMonitor:
    """Monitor system resource usage"""
    def __init__(self):
        self.process = psutil.Process()
        
    def log_usage(self):
        """Log current resource usage"""
        memory_info = self.process.memory_info()
        cpu_percent = self.process.cpu_percent()
        
        logging.info(
            f"Resource Usage - Memory: {memory_info.rss / 1024 / 1024:.2f}MB, "
            f"CPU: {cpu_percent}%"
        )
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            logging.info(f"GPU Memory: {gpu_memory:.2f}MB") 
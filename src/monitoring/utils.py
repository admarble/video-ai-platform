from typing import Optional
from .tracing import TracingConfig, TracingSystem
from .video_processor import TracedVideoProcessor

def setup_video_processing_with_tracing(
    service_name: str = "video-processing-service",
    otlp_endpoint: Optional[str] = None
) -> TracedVideoProcessor:
    """Setup video processor with tracing enabled"""
    
    config = TracingConfig(
        service_name=service_name,
        otlp_endpoint=otlp_endpoint,
        enabled=True,
        sampling_rate=1.0
    )
    
    tracing = TracingSystem(config)
    processor = TracedVideoProcessor(tracing)
    
    return processor 
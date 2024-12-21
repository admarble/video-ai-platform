import logging
from typing import Optional, Dict, Any, Union, Type, cast, TypeVar, Generator, TYPE_CHECKING
from contextlib import contextmanager

if TYPE_CHECKING:
    from opentelemetry.trace import Span as OTelSpan
else:
    OTelSpan = Any

# OpenTelemetry imports with type hints
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.trace import Span, Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    # Create dummy classes for type hints
    class DummySpan:
        def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None: pass
        def record_exception(self, exception: Exception, attributes: Optional[Dict[str, Any]] = None) -> None: pass
        def set_status(self, status: Any, description: Optional[str] = None) -> None: pass
    
    Span = DummySpan
    Status = type('Status', (), {'ERROR': 'ERROR'})
    StatusCode = type('StatusCode', (), {'ERROR': 'ERROR'})
    TracerProvider = type('TracerProvider', (), {})
    OTLPSpanExporter = type('OTLPSpanExporter', (), {})
    Resource = type('Resource', (), {'create': staticmethod(lambda x: None)})
    SERVICE_NAME = 'service.name'
    SERVICE_VERSION = 'service.version'
    trace = None

# Type variable for the context manager
T = TypeVar('T', bound=Optional[OTelSpan])

class TracingConfig:
    """Configuration for tracing system"""
    def __init__(
        self,
        service_name: str = "video-processing-service",
        otlp_endpoint: Optional[str] = None,
        enabled: bool = True,
        sampling_rate: float = 1.0,
        export_timeout_ms: int = 30000
    ):
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint
        self.enabled = enabled and OPENTELEMETRY_AVAILABLE
        self.sampling_rate = sampling_rate
        self.export_timeout_ms = export_timeout_ms

class TracingSystem:
    """Manages distributed tracing for the video processing pipeline"""
    
    def __init__(self, config: TracingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tracer = None
        
        if self.config.enabled:
            self._initialize_tracing()
    
    def _initialize_tracing(self) -> None:
        """Initialize OpenTelemetry tracing"""
        try:
            # Create resource attributes
            resource = Resource.create({
                SERVICE_NAME: self.config.service_name,
                SERVICE_VERSION: "1.0.0",
                "environment": "production"
            })
            
            # Initialize tracer provider
            provider = TracerProvider(resource=resource)
            
            # Configure OTLP exporter if endpoint is provided
            if self.config.otlp_endpoint:
                otlp_exporter = OTLPSpanExporter(
                    endpoint=self.config.otlp_endpoint,
                    timeout=self.config.export_timeout_ms
                )
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                
            # Set global tracer provider
            trace.set_tracer_provider(provider)
            
            # Create tracer
            self.tracer = trace.get_tracer(__name__)
            
            self.logger.info("Tracing system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize tracing: {str(e)}")
            self.config.enabled = False

    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        parent: Optional[OTelSpan] = None
    ) -> Generator[Optional[OTelSpan], None, None]:
        """Start a new trace span"""
        if not self.config.enabled or not self.tracer:
            yield None
            return
        
        try:
            context = trace.set_span_in_context(parent) if parent else None
            with self.tracer.start_as_current_span(
                name,
                context=context,
                attributes=attributes or {}
            ) as span:
                yield span
        except Exception as e:
            self.logger.error(f"Error in span {name}: {str(e)}")
            yield None

    def add_event(
        self,
        span: Optional[OTelSpan],
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add event to span"""
        if span and self.config.enabled:
            try:
                span.add_event(name, attributes=attributes or {})
            except Exception as e:
                self.logger.error(f"Error adding event to span: {str(e)}")

    def record_exception(
        self,
        span: Optional[OTelSpan],
        exception: Exception,
        attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record exception in span"""
        if span and self.config.enabled:
            try:
                span.record_exception(
                    exception,
                    attributes=attributes or {}
                )
                if OPENTELEMETRY_AVAILABLE:
                    span.set_status(
                        Status(StatusCode.ERROR, str(exception))
                    )
            except Exception as e:
                self.logger.error(f"Error recording exception in span: {str(e)}")
    
    # ... [Previous TracingSystem methods remain the same] 
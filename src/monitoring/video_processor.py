from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union
import time
import logging
import cv2
import numpy as np
from datetime import timedelta

# OpenTelemetry import with fallback
try:
    from opentelemetry import trace
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None

from .metrics_collector import MetricsCollector
from .service_manager import ServiceManager
from .logging_manager import ComponentLogger
from .tracing import TracingSystem, Span

def _extract_frames(video_path: str) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()
        
    return frames, fps

class TracedVideoProcessor:
    """Video processor with tracing instrumentation"""
    
    def __init__(self, tracing: TracingSystem):
        self.tracing = tracing
        self.logger = logging.getLogger(__name__)
    
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """Process video with distributed tracing"""
        with self.tracing.start_span("process_video") as span:
            try:
                # Add video metadata to span
                video_file = Path(video_path)
                self.tracing.add_event(span, "video_metadata", {
                    "video_path": str(video_file),
                    "video_size": video_file.stat().st_size,
                    "video_name": video_file.name
                })
                
                # Extract frames
                with self.tracing.start_span("extract_frames", parent=span) as frames_span:
                    frames, fps = self._extract_frames(video_path)
                    self.tracing.add_event(frames_span, "frames_extracted", {
                        "frame_count": len(frames),
                        "fps": fps
                    })
                
                # Create results dictionary
                results = {
                    "video_path": str(video_file),
                    "frame_count": len(frames),
                    "fps": fps,
                    "status": "success"
                }
                
                return results
                
            except Exception as e:
                self.tracing.record_exception(span, e)
                raise

    # ... (rest of the implementation remains the same as in your original code) 
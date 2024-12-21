from pathlib import Path
from typing import Optional, Dict, Any
import time
import logging
from datetime import timedelta
import torch

from .alert_manager import AlertManager, AlertSeverity, AlertChannel
from .metrics_collector import MetricsCollector
from ..services.service_manager import ServiceManager
from .monitoring_system import MonitoringSystem, MonitoringMetric

class MonitoredVideoProcessor:
    """Video processor with integrated performance monitoring and alerting"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        base_dir: Optional[Path] = None
    ):
        self.config = config
        self.base_dir = base_dir or Path("./data")
        
        # Initialize components
        self.service_manager = ServiceManager(config.get('service_config', {}))
        self.metrics_collector = MetricsCollector(config.get('metrics_config', {
            'metrics_dir': str(self.base_dir / "metrics"),
            'alert_config': config.get('alert_config', {}),
            'alert_history_path': str(self.base_dir / "alert_history.json")
        }))
        
        # Setup monitoring system
        self.monitoring = MonitoringSystem(
            self.metrics_collector,
            config.get('monitoring_config', {}),
            alert_history_file=str(self.base_dir / "alert_history.json")
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Start monitoring
        self.metrics_collector.start()
        
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """Process video with performance monitoring and alerting"""
        video_id = Path(video_path).stem
        processing_start = time.time()
        
        try:
            # Start monitoring
            self.metrics_collector.start_processing_metrics(video_id)
            
            # Initialize services
            self.service_manager.initialize_services([
                'scene_processor',
                'object_detector',
                'audio_processor'
            ])
            
            # Extract frames with monitoring
            frames, fps = self.service_manager.extract_frames(video_path)
            frame_metrics = {
                'frames_processed': len(frames),
                'frames_per_second': len(frames) / (time.time() - processing_start)
            }
            
            self.metrics_collector.update_processing_metrics(video_id, **frame_metrics)
            
            # Process scenes
            scene_processor = self.service_manager.get_service('scene_processor')
            scenes = scene_processor.process_scenes(frames)
            scene_metrics = {
                'scene_count': len(scenes),
                'model_confidence': sum(s.confidence for s in scenes) / len(scenes)
            }
            
            self.metrics_collector.update_processing_metrics(video_id, **scene_metrics)
            
            # Detect objects
            object_detector = self.service_manager.get_service('object_detector')
            detections = object_detector.process_frames(frames, enable_tracking=True)
            
            object_metrics = {
                'object_count': sum(len(frame_dets) for frame_dets in detections),
                'processing_time': time.time() - processing_start
            }
            
            self.metrics_collector.update_processing_metrics(video_id, **object_metrics)
            
            # Process audio
            audio_processor = self.service_manager.get_service('audio_processor')
            audio_segments = audio_processor.process_audio(video_path)
            
            # Prepare results
            results = {
                'video_id': video_id,
                'duration': time.time() - processing_start,
                'frame_count': len(frames),
                'fps': fps,
                'scene_count': len(scenes),
                'object_count': object_metrics['object_count'],
                'audio_segments': len(audio_segments),
                'performance_metrics': self._get_performance_summary(video_id)
            }
            
            # Final metrics check
            self.metrics_collector.update_processing_metrics(
                video_id,
                frames_processed=len(frames),
                scene_count=len(scenes),
                object_count=object_metrics['object_count']
            )
            
            # Complete monitoring
            self.metrics_collector.complete_processing_metrics(video_id)
            
            return results
            
        except Exception as e:
            # Record error and trigger alert
            error_details = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'processing_time': time.time() - processing_start
            }
            
            self.metrics_collector.update_processing_metrics(
                video_id,
                error=str(e)
            )
            
            self.metrics_collector.record_metric(
                'video_processing_error',
                1.0,
                {
                    'video_id': video_id,
                    'error_type': error_details['error_type']
                }
            )
            
            self.metrics_collector.complete_processing_metrics(video_id)
            raise
            
        finally:
            # Cleanup
            self.service_manager.cleanup()
            
    def _get_performance_summary(self, video_id: str) -> Dict[str, Any]:
        """Get performance metrics summary"""
        # Get system metrics for the processing duration
        system_metrics = self.metrics_collector.get_system_metrics_summary(
            duration=timedelta(hours=1)
        )
        
        # Get processing-specific metrics
        with self.metrics_collector.lock:
            if video_id in self.metrics_collector.current_metrics:
                metrics = self.metrics_collector.current_metrics[video_id]
                
                return {
                    'frames_per_second': metrics.frames_processed / (time.time() - metrics.start_time),
                    'memory_usage_mb': metrics.memory_usage_mb,
                    'gpu_memory_mb': metrics.gpu_memory_mb,
                    'system_metrics': system_metrics
                }
                
        return {}
        
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        health_metrics = {
            'system_metrics': self.metrics_collector.get_system_metrics_summary(),
            'active_processes': len(self.metrics_collector.current_metrics),
            'gpu_available': torch.cuda.is_available(),
            'gpu_memory_used': (
                torch.cuda.memory_allocated() / 1024 / 1024
                if torch.cuda.is_available() else None
            )
        }
        
        self.metrics_collector.record_metric(
            'system_health',
            1.0 if health_metrics['gpu_available'] else 0.0,
            {'gpu_available': health_metrics['gpu_available']}
        )
        
        return health_metrics
        
    def cleanup(self):
        """Cleanup monitoring and alerting systems"""
        self.metrics_collector.cleanup_old_metrics()
        self.metrics_collector.stop() 
from pathlib import Path
from typing import Dict, Any
import time
from src.core.logging import LoggingManager, LogLevel

class MonitoredVideoProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.service_manager = ServiceManager()
        self.metrics_collector = MetricsCollector()
        
        # Initialize logging manager
        self.logging_manager = LoggingManager(
            base_dir=Path(__file__).parent.parent.parent,
            config={
                "system": {
                    "level": "INFO",
                    "format": "%(asctime)s - [%(levelname)s] %(message)s - %(video_id)s - %(metric)s: %(value)s"
                }
            }
        )
        self.logger = self.logging_manager.get_logger("system")
        
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """Process video with monitoring"""
        processing_start = time.time()
        video_id = self._generate_video_id(video_path)
        
        try:
            self.logger.info(
                "Starting video processing",
                extra={
                    "video_id": video_id,
                    "metric": "status",
                    "value": "started"
                }
            )
            
            # Extract frames
            frame_extractor = self.service_manager.get_service('frame_extractor')
            frames, fps = frame_extractor._extract_frames(video_path)
            
            self.logger.info(
                "Frames extracted",
                extra={
                    "video_id": video_id,
                    "metric": "frames",
                    "value": len(frames)
                }
            )
            
            # Scene analysis
            scene_processor = self.service_manager.get_service('scene_processor')
            scenes = scene_processor.process_scenes(frames)
            
            self.logger.info(
                "Scene analysis completed",
                extra={
                    "video_id": video_id,
                    "metric": "scenes",
                    "value": len(scenes)
                }
            )
            
            # Object detection
            object_detector = self.service_manager.get_service('object_detector')
            detections = object_detector.process_frames(frames, enable_tracking=True)
            
            object_metrics = {
                'object_count': sum(len(frame_dets) for frame_dets in detections),
                'processing_time': time.time() - processing_start
            }
            
            self.logger.info(
                "Object detection completed",
                extra={
                    "video_id": video_id,
                    "metric": "objects",
                    "value": object_metrics['object_count']
                }
            )
            
            self.metrics_collector.update_processing_metrics(video_id, **object_metrics)
            
            # Process audio
            audio_processor = self.service_manager.get_service('audio_processor')
            audio_segments = audio_processor.process_audio(video_path)
            
            self.logger.info(
                "Audio processing completed",
                extra={
                    "video_id": video_id,
                    "metric": "audio_segments",
                    "value": len(audio_segments)
                }
            )
            
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
            
            self.logger.info(
                "Video processing completed",
                extra={
                    "video_id": video_id,
                    "metric": "duration",
                    "value": results['duration']
                }
            )
            
            return results
            
        except Exception as e:
            self.logger.error(
                f"Video processing failed: {str(e)}",
                extra={
                    "video_id": video_id,
                    "metric": "status",
                    "value": "failed"
                },
                exc_info=True
            )
            raise 
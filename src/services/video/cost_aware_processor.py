from typing import Dict, Any, Optional, List
from pathlib import Path
import time
import logging
import numpy as np
from datetime import datetime

from ...monitoring import CostMonitor, ResourceType, CostConfig
from ...monitoring.auto_scaler import AutoScaler, create_auto_scaler
from ..service_manager import ServiceManager
from .cost_optimizer import CostOptimizer, create_cost_optimizer

class CostAwareVideoProcessor:
    """Video processor with integrated cost monitoring and auto-scaling"""
    
    def __init__(
        self,
        cost_monitor: CostMonitor,
        service_manager: ServiceManager,
        base_dir: Path,
        config: Optional[Dict[str, Any]] = None
    ):
        self.cost_monitor = cost_monitor
        self.service_manager = service_manager
        self.base_dir = base_dir
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize cost optimizer
        self.optimizer = create_cost_optimizer(
            cost_monitor=cost_monitor,
            config=self.config.get('optimization', {})
        )
        
        # Initialize auto-scaler
        self.auto_scaler = None
        self._initialize_auto_scaler()

    async def _initialize_auto_scaler(self):
        """Initialize the auto-scaler"""
        try:
            self.auto_scaler = await create_auto_scaler(
                cost_monitor=self.cost_monitor,
                config=self.config.get('auto_scaling', {
                    'max_cost_per_hour': 10.0,
                    'max_instances': 10,
                    'min_free_memory': 2.0,  # GB
                    'max_gpu_memory': 16.0,  # GB
                    'check_interval': 60     # seconds
                })
            )
            self.logger.info("Auto-scaler initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize auto-scaler: {str(e)}")
            self.auto_scaler = None

    async def process_video(
        self,
        video_path: str,
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process video with cost tracking and auto-scaling"""
        start_time = time.time()
        video_id = task_id or Path(video_path).stem

        try:
            # Get video metadata
            metadata = await self._analyze_video_metadata(video_path)
            
            # Get optimized processing parameters
            profile = self.optimizer.optimize_processing_params(
                video_metadata=metadata,
                current_costs=self.cost_monitor.get_current_costs().to_dict()
            )
            
            # Initialize services with cost tracking
            await self._initialize_services_with_cost(video_id, profile)
            
            # Process video with cost tracking
            frames, frame_cost = await self._process_frames_with_cost(
                video_path, video_id, profile
            )
            
            scenes = await self._process_scenes_with_cost(
                frames, video_id, profile
            )
            
            objects = await self._process_objects_with_cost(
                frames, video_id, profile
            )
            
            audio = await self._process_audio_with_cost(
                video_path, video_id, profile
            )
            
            # Calculate total processing time
            processing_time = time.time() - start_time
            
            # Track processing metrics for auto-scaling
            self._update_processing_metrics(processing_time)
            
            return {
                'video_id': video_id,
                'scenes': scenes,
                'objects': objects,
                'audio': audio,
                'processing_time': processing_time,
                'costs': self.cost_monitor.get_video_costs(video_id)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing video {video_id}: {str(e)}")
            # Track error for auto-scaling
            self._update_error_metrics()
            raise

    def _update_processing_metrics(self, processing_time: float):
        """Update processing metrics for auto-scaling"""
        try:
            metrics = self.cost_monitor.get_cost_summary(days=1)
            current_metrics = metrics.get('processing_metrics', {})
            
            # Update processing times
            times = current_metrics.get('processing_times', [])
            times.append(processing_time)
            # Keep last 100 processing times
            times = times[-100:]
            
            # Calculate average
            avg_time = sum(times) / len(times)
            
            # Update metrics
            current_metrics['processing_times'] = times
            current_metrics['avg_processing_time'] = avg_time
            metrics['processing_metrics'] = current_metrics
            
            # Save updated metrics
            self.cost_monitor.save_cost_snapshot({
                'processing_metrics': current_metrics
            })
            
        except Exception as e:
            self.logger.error(f"Error updating processing metrics: {str(e)}")

    def _update_error_metrics(self):
        """Update error metrics for auto-scaling"""
        try:
            metrics = self.cost_monitor.get_cost_summary(days=1)
            current_metrics = metrics.get('processing_metrics', {})
            
            # Update error count
            error_count = current_metrics.get('error_count', 0) + 1
            total_processed = current_metrics.get('total_processed', 0) + 1
            
            # Calculate error rate
            error_rate = error_count / total_processed if total_processed > 0 else 0
            
            # Update metrics
            current_metrics.update({
                'error_count': error_count,
                'total_processed': total_processed,
                'error_rate': error_rate
            })
            metrics['processing_metrics'] = current_metrics
            
            # Save updated metrics
            self.cost_monitor.save_cost_snapshot({
                'processing_metrics': current_metrics
            })
            
        except Exception as e:
            self.logger.error(f"Error updating error metrics: {str(e)}")

    async def _analyze_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Analyze video to extract metadata for optimization"""
        try:
            # Get basic video info using service manager
            video_info = self.service_manager.get_video_info(video_path)
            
            # Calculate motion metrics if supported
            motion_stats = self.service_manager.analyze_motion(video_path)
            
            return {
                'duration': video_info.get('duration', 0),
                'resolution': video_info.get('resolution', ''),
                'average_motion': motion_stats.get('average_motion', 0.5),
                'motion_complexity': motion_stats.get('complexity', 0.5),
                'size_gb': Path(video_path).stat().st_size / (1024 * 1024 * 1024)
            }
        except Exception as e:
            self.logger.warning(f"Error analyzing video metadata: {str(e)}")
            return {
                'duration': 0,
                'resolution': '',
                'average_motion': 0.5,
                'size_gb': Path(video_path).stat().st_size / (1024 * 1024 * 1024)
            }

    async def _initialize_services_with_cost(
        self,
        video_id: str,
        profile: 'ProcessingProfile'
    ):
        """Initialize services with cost tracking"""
        start_time = time.time()

        try:
            # Initialize services with optimized parameters
            self.service_manager.initialize_services(
                [
                    'scene_processor',
                    'object_detector',
                    'audio_processor'
                ],
                config={
                    'model_quality': profile.model_quality,
                    'batch_size': profile.batch_size,
                    'cache_policy': profile.cache_policy
                }
            )

            # Track compute time for initialization
            compute_hours = (time.time() - start_time) / 3600
            self.cost_monitor.track_resource_usage(
                ResourceType.COMPUTE,
                amount=compute_hours,
                video_id=video_id
            )

            # Track GPU memory usage if applicable
            if hasattr(self.service_manager, 'get_gpu_memory_usage'):
                gpu_memory_gb = self.service_manager.get_gpu_memory_usage()
                if gpu_memory_gb > 0:
                    self.cost_monitor.track_resource_usage(
                        ResourceType.CACHE,
                        amount=gpu_memory_gb,
                        video_id=video_id
                    )

        except Exception as e:
            self.logger.error(f"Error initializing services: {str(e)}")
            raise

    async def _process_frames_with_cost(
        self,
        video_path: str,
        video_id: str,
        profile: 'ProcessingProfile'
    ) -> tuple[np.ndarray, float]:
        """Process frames with cost tracking"""
        start_time = time.time()

        try:
            # Extract frames with optimized parameters
            frames, fps = self.service_manager.extract_frames(
                video_path,
                sample_rate=profile.frame_sample_rate,
                target_fps=profile.target_fps
            )

            # Track compute time
            compute_hours = (time.time() - start_time) / 3600
            self.cost_monitor.track_resource_usage(
                ResourceType.COMPUTE,
                amount=compute_hours,
                video_id=video_id
            )

            # Track storage for extracted frames
            frames_size_gb = frames.nbytes / (1024 * 1024 * 1024)
            self.cost_monitor.track_resource_usage(
                ResourceType.STORAGE,
                amount=frames_size_gb,
                video_id=video_id
            )

            return frames, fps

        except Exception as e:
            self.logger.error(f"Error extracting frames: {str(e)}")
            raise

    async def _process_scenes_with_cost(
        self,
        frames: np.ndarray,
        video_id: str,
        profile: 'ProcessingProfile'
    ) -> List[Dict[str, Any]]:
        """Process scenes with cost tracking"""
        start_time = time.time()

        try:
            scene_processor = self.service_manager.get_service('scene_processor')
            scenes = scene_processor.process_scenes(
                frames,
                batch_size=profile.batch_size
            )

            # Track compute time
            compute_hours = (time.time() - start_time) / 3600
            self.cost_monitor.track_resource_usage(
                ResourceType.COMPUTE,
                amount=compute_hours,
                video_id=video_id
            )

            # Track model inferences
            self.cost_monitor.track_resource_usage(
                ResourceType.MODEL_INFERENCE,
                amount=len(frames),  # One inference per frame
                video_id=video_id
            )

            return scenes

        except Exception as e:
            self.logger.error(f"Error processing scenes: {str(e)}")
            raise

    async def _process_objects_with_cost(
        self,
        frames: np.ndarray,
        video_id: str,
        profile: 'ProcessingProfile'
    ) -> List[Dict[str, Any]]:
        """Process objects with cost tracking"""
        start_time = time.time()

        try:
            object_detector = self.service_manager.get_service('object_detector')
            detections = object_detector.process_frames(
                frames,
                batch_size=profile.batch_size,
                model_quality=profile.model_quality
            )

            # Track compute time
            compute_hours = (time.time() - start_time) / 3600
            self.cost_monitor.track_resource_usage(
                ResourceType.COMPUTE,
                amount=compute_hours,
                video_id=video_id
            )

            # Track model inferences
            self.cost_monitor.track_resource_usage(
                ResourceType.MODEL_INFERENCE,
                amount=len(frames),  # One inference per frame
                video_id=video_id
            )

            return detections

        except Exception as e:
            self.logger.error(f"Error detecting objects: {str(e)}")
            raise

    async def _process_audio_with_cost(
        self,
        video_path: str,
        video_id: str,
        profile: 'ProcessingProfile'
    ) -> List[Dict[str, Any]]:
        """Process audio with cost tracking"""
        start_time = time.time()

        try:
            audio_processor = self.service_manager.get_service('audio_processor')
            segments = audio_processor.process_audio(
                video_path,
                model_quality=profile.model_quality
            )

            # Track compute time
            compute_hours = (time.time() - start_time) / 3600
            self.cost_monitor.track_resource_usage(
                ResourceType.COMPUTE,
                amount=compute_hours,
                video_id=video_id
            )

            # Track model inferences (one per audio segment)
            self.cost_monitor.track_resource_usage(
                ResourceType.MODEL_INFERENCE,
                amount=len(segments),
                video_id=video_id
            )

            return segments

        except Exception as e:
            self.logger.error(f"Error processing audio: {str(e)}")
            raise

    def get_processing_costs(
        self,
        video_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get processing cost summary with optimization impact"""
        cost_summary = self.cost_monitor.get_cost_summary(days, video_id)
        optimization_impact = self.optimizer.analyze_optimization_impact(days)
        
        return {
            'cost_summary': cost_summary,
            'optimization_impact': optimization_impact
        }

    def analyze_processing_efficiency(
        self,
        video_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Analyze processing efficiency and optimization effectiveness"""
        cost_trends = self.cost_monitor.analyze_cost_trends(days)
        optimization_impact = self.optimizer.analyze_optimization_impact(days)
        
        # Calculate efficiency metrics
        total_cost = cost_trends['summary']['total_cost']
        total_videos = len(cost_trends.get('daily_costs', {}))
        
        if total_videos > 0:
            avg_cost_per_video = total_cost / total_videos
            cost_reduction = optimization_impact.get('average_cost_reduction', 0)
            quality_impact = optimization_impact.get('quality_impact', 0)
            
            efficiency_score = (1 - cost_reduction) * (1 + quality_impact)
        else:
            avg_cost_per_video = 0
            efficiency_score = 0
        
        return {
            'cost_trends': cost_trends,
            'optimization_impact': optimization_impact,
            'efficiency_metrics': {
                'avg_cost_per_video': avg_cost_per_video,
                'efficiency_score': efficiency_score,
                'quality_adjusted_savings': optimization_impact.get('total_cost_savings', 0) * 
                                         (1 + optimization_impact.get('quality_impact', 0))
            }
        } 
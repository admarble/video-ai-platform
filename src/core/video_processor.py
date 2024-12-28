from pathlib import Path
from typing import Dict, Optional, List, Any, Union
from dataclasses import dataclass
import json
import hashlib
from datetime import datetime
import os
import shutil

from src.services.service_manager import ServiceManager, ModelConfig
from src.exceptions import VideoProcessingError, ServiceInitializationError
from src.core.config import get_config
from src.core.logging import LoggingManager, LogLevel

@dataclass
class VideoMetadata:
    """Metadata for a video file"""
    video_id: str
    filename: str
    file_path: str
    file_size: int
    duration: float
    width: int
    height: int
    fps: float
    created_at: datetime
    format: str
    status: str = "pending"

@dataclass
class ProcessingResults:
    """Results from video processing pipeline"""
    metadata: VideoMetadata
    scenes: List[Dict]
    objects: List[Dict]
    audio_segments: List[Dict]
    frame_embeddings: Optional[List[float]] = None

class VideoProcessor:
    """Main video processing orchestration"""
    
    def __init__(
        self,
        upload_dir: Optional[str] = None,
        processing_dir: Optional[str] = None,
        model_config: Optional[ModelConfig] = None
    ):
        """
        Initialize video processor.
        
        Args:
            upload_dir: Directory for uploaded videos
            processing_dir: Directory for processing artifacts
            model_config: Configuration for ML models
        """
        config = get_config()
        self.upload_dir = Path(upload_dir or config.UPLOAD_DIR)
        self.processing_dir = Path(processing_dir or config.PROCESSING_DIR)
        
        # Create directories
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processing_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize service manager
        self.service_manager = ServiceManager(model_config)
        self.service_manager.initialize_services(parallel=True)
        
        # Initialize logging manager
        self.logging_manager = LoggingManager(
            base_dir=Path(__file__).parent.parent.parent,
            config={
                "video_processing": {
                    "level": "INFO",
                    "format": "%(asctime)s - [%(levelname)s] %(message)s - %(video_id)s"
                }
            }
        )
        self.logger = self.logging_manager.get_logger("video_processing")
        self.logger.info("Initialized VideoProcessor")
        
    def init_video(
        self,
        video_upload: Union[str, Path, bytes],
        filename: Optional[str] = None
    ) -> VideoMetadata:
        """
        Initialize video processing from upload.
        
        Args:
            video_upload: Path to video file or bytes
            filename: Original filename if video_upload is bytes
            
        Returns:
            VideoMetadata object
        """
        try:
            # Handle different upload types
            if isinstance(video_upload, (str, Path)):
                video_path = Path(video_upload)
                if not video_path.exists():
                    raise VideoProcessingError(f"Video file not found: {video_path}")
                filename = filename or video_path.name
            else:
                # Handle bytes upload
                if not filename:
                    raise VideoProcessingError("Filename required for bytes upload")
                temp_path = self.upload_dir / filename
                with open(temp_path, 'wb') as f:
                    f.write(video_upload)
                video_path = temp_path
                
            # Generate unique video ID
            video_id = self._generate_video_id(video_path)
            
            # Create processing directory for this video
            video_processing_dir = self.processing_dir / video_id
            video_processing_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy video to processing directory
            dest_path = video_processing_dir / filename
            shutil.copy2(video_path, dest_path)
            
            # Extract basic metadata using ffprobe
            metadata = self._extract_metadata(dest_path)
            
            # Create metadata object
            video_metadata = VideoMetadata(
                video_id=video_id,
                filename=filename,
                file_path=str(dest_path),
                file_size=dest_path.stat().st_size,
                duration=metadata['duration'],
                width=metadata['width'],
                height=metadata['height'],
                fps=metadata['fps'],
                created_at=datetime.now(),
                format=metadata['format']
            )
            
            # Save metadata
            self._save_metadata(video_metadata)
            
            logging.info(f"Initialized video processing for {filename}")
            return video_metadata
            
        except Exception as e:
            raise VideoProcessingError(f"Failed to initialize video: {str(e)}")
            
    def _generate_video_id(self, video_path: Path) -> str:
        """Generate unique ID for video based on content and timestamp"""
        timestamp = datetime.now().isoformat()
        content_hash = hashlib.md5(open(video_path, 'rb').read()).hexdigest()
        return f"{content_hash[:10]}_{timestamp}"
        
    def _extract_metadata(self, video_path: Path) -> Dict:
        """Extract video metadata using ffprobe"""
        import ffmpeg
        try:
            probe = ffmpeg.probe(str(video_path))
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            
            return {
                'duration': float(probe['format']['duration']),
                'width': int(video_info['width']),
                'height': int(video_info['height']),
                'fps': eval(video_info['r_frame_rate']),  # Convert fraction to float
                'format': probe['format']['format_name']
            }
        except Exception as e:
            raise VideoProcessingError(f"Failed to extract metadata: {str(e)}")
            
    def _save_metadata(self, metadata: VideoMetadata) -> None:
        """Save video metadata to JSON file"""
        metadata_path = self.processing_dir / metadata.video_id / "metadata.json"
        
        with open(metadata_path, 'w') as f:
            # Convert datetime to string for JSON serialization
            metadata_dict = {
                **metadata.__dict__,
                'created_at': metadata.created_at.isoformat()
            }
            json.dump(metadata_dict, f, indent=2)
            
    def process_video(
        self,
        video_metadata: VideoMetadata,
        extract_frames: bool = True,
        analyze_scenes: bool = True,
        detect_objects: bool = True,
        process_audio: bool = True,
        generate_embeddings: bool = True
    ) -> ProcessingResults:
        """
        Process video through all enabled components.
        
        Args:
            video_metadata: Video metadata object
            extract_frames: Whether to extract video frames
            analyze_scenes: Whether to perform scene analysis
            detect_objects: Whether to perform object detection
            process_audio: Whether to process audio
            generate_embeddings: Whether to generate frame embeddings
            
        Returns:
            ProcessingResults object with all enabled results
        """
        try:
            self.logger.info(
                "Starting video processing",
                extra={"video_id": video_metadata.video_id}
            )
            
            results = ProcessingResults(
                metadata=video_metadata,
                scenes=[],
                objects=[],
                audio_segments=[],
                frame_embeddings=None
            )
            
            # Extract frames if needed or if required by other components
            frames = None
            if extract_frames or analyze_scenes or detect_objects or generate_embeddings:
                frame_extractor = self.service_manager.get_service("frame_extractor")
                frames, fps = frame_extractor._extract_frames(
                    video_metadata.file_path,
                    sampling_rate=2  # Can be configurable
                )
                self.logger.info(
                    f"Extracted {len(frames)} frames from video",
                    extra={"video_id": video_metadata.video_id}
                )
                
            # Scene analysis
            if analyze_scenes and frames is not None:
                scene_processor = self.service_manager.get_service("scene_processor")
                results.scenes = scene_processor.process_scenes(frames)
                self.logger.info(
                    f"Analyzed {len(results.scenes)} scenes",
                    extra={"video_id": video_metadata.video_id}
                )
                
            # Object detection
            if detect_objects and frames is not None:
                object_detector = self.service_manager.get_service("object_detector")
                results.objects = object_detector.process_frames(frames)
                self.logger.info(
                    f"Detected objects in {len(frames)} frames",
                    extra={"video_id": video_metadata.video_id}
                )
                
            # Audio processing
            if process_audio:
                audio_processor = self.service_manager.get_service("audio_processor")
                results.audio_segments = audio_processor.process_audio(
                    video_metadata.file_path
                )
                self.logger.info(
                    f"Processed {len(results.audio_segments)} audio segments",
                    extra={"video_id": video_metadata.video_id}
                )
                
            # Generate embeddings
            if generate_embeddings and frames is not None:
                text_aligner = self.service_manager.get_service("text_aligner")
                results.frame_embeddings = text_aligner._generate_embeddings(frames)
                self.logger.info(
                    "Generated frame embeddings",
                    extra={"video_id": video_metadata.video_id}
                )
                
            # Save results
            self._save_results(video_metadata.video_id, results)
            
            # Update status
            video_metadata.status = "completed"
            self._save_metadata(video_metadata)
            
            self.logger.info(
                "Video processing completed successfully",
                extra={"video_id": video_metadata.video_id}
            )
            
            return results
            
        except Exception as e:
            # Update status on failure
            video_metadata.status = "failed"
            self._save_metadata(video_metadata)
            
            self.logger.error(
                f"Failed to process video: {str(e)}",
                extra={"video_id": video_metadata.video_id},
                exc_info=True
            )
            raise VideoProcessingError(f"Failed to process video: {str(e)}")
            
    def _save_results(self, video_id: str, results: ProcessingResults) -> None:
        """Save processing results to files"""
        results_dir = self.processing_dir / video_id
        
        # Save scenes
        if results.scenes:
            with open(results_dir / "scenes.json", 'w') as f:
                json.dump([s.__dict__ for s in results.scenes], f, indent=2)
                
        # Save object detections
        if results.objects:
            with open(results_dir / "objects.json", 'w') as f:
                json.dump([o.__dict__ for o in results.objects], f, indent=2)
                
        # Save audio segments
        if results.audio_segments:
            with open(results_dir / "audio_segments.json", 'w') as f:
                json.dump([s.__dict__ for s in results.audio_segments], f, indent=2)
                
        # Save embeddings
        if results.frame_embeddings is not None:
            np.save(
                results_dir / "frame_embeddings.npy",
                results.frame_embeddings
            )
            
    def cleanup(self) -> None:
        """Cleanup resources"""
        self.service_manager.cleanup()

# Helper function for easy initialization
def create_video_processor(
    upload_dir: Optional[str] = None,
    processing_dir: Optional[str] = None,
    model_config: Optional[ModelConfig] = None
) -> VideoProcessor:
    """Create and initialize video processor with optional configuration"""
    try:
        return VideoProcessor(upload_dir, processing_dir, model_config)
    except Exception as e:
        raise ServiceInitializationError(f"Failed to create video processor: {str(e)}")
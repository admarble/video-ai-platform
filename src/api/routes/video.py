from fastapi import APIRouter
import uuid
import tempfile
import os
import cv2.headless as cv2
from datetime import datetime
from urllib.parse import urlparse
import aiohttp
from typing import Dict, Tuple, List
from pydantic import BaseModel
from enum import Enum
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Enums
class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Custom Exceptions
class VideoProcessingError(Exception):
    """Custom exception for video processing errors"""
    pass

# Pydantic Models
class VideoUpload(BaseModel):
    """Request model for video upload"""
    url: str
    title: str
    description: str | None = None

class VideoMetadata(BaseModel):
    """Model for video metadata"""
    id: str
    url: str
    status: ProcessingStatus
    duration: float | None = None
    frame_count: int | None = None
    fps: float | None = None
    resolution: Tuple[int, int] | None = None
    format: str | None = None
    size_bytes: int | None = None
    created_at: datetime
    updated_at: datetime
    title: str
    description: str | None = None

router = APIRouter()

@router.get("/")
async def list_videos():
    return {"message": "List of videos"}

class VideoRouter:
    # ... existing methods ...

    async def init_video(self, video_upload: VideoUpload) -> VideoMetadata:
        """
        Initialize video processing by validating URL, downloading sample frames,
        and extracting metadata.
        """
        try:
            # Generate unique video ID
            video_id = str(uuid.uuid4())
            logger.info(f"Initializing video processing for ID: {video_id}")

            # Validate and extract video information
            metadata = await self._validate_video_url(video_upload.url)
            
            # Create video metadata object
            video_metadata = VideoMetadata(
                id=video_id,
                url=video_upload.url,
                status=ProcessingStatus.PENDING,
                duration=metadata.get('duration'),
                frame_count=metadata.get('frame_count'),
                fps=metadata.get('fps'),
                resolution=metadata.get('resolution'),
                format=metadata.get('format'),
                size_bytes=metadata.get('size_bytes'),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                title=video_upload.title,
                description=video_upload.description
            )
            
            logger.info(f"Video metadata initialized: {video_metadata.model_dump_json()}")
            return video_metadata

        except Exception as e:
            error_msg = f"Failed to initialize video: {str(e)}"
            logger.error(error_msg)
            raise VideoProcessingError(error_msg)

    async def _validate_video_url(self, url: str) -> Dict:
        """Validate video URL and extract basic metadata."""
        try:
            # Parse URL
            parsed_url = urlparse(str(url))
            if not parsed_url.scheme or not parsed_url.netloc:
                raise VideoProcessingError("Invalid URL format")

            # Create temporary file for video sample
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                temp_path = tmp_file.name

            # Download sample of video file
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise VideoProcessingError(f"Failed to access video URL: {response.status}")
                    
                    # Download initial bytes for validation
                    with open(temp_path, 'wb') as f:
                        f.write(await response.content.read(10 * 1024 * 1024))

            # Extract metadata
            metadata = await self._extract_video_metadata(temp_path)
            os.unlink(temp_path)
            return metadata

        except Exception as e:
            raise VideoProcessingError(f"Video validation failed: {str(e)}")

    async def _extract_video_metadata(self, video_path: str) -> Dict:
        """Extract comprehensive metadata from video file."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise VideoProcessingError("Failed to open video file")

            metadata = {
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
                'resolution': (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                ),
                'size_bytes': os.path.getsize(video_path),
                'format': self._determine_video_format(video_path)
            }

            # Validate metadata
            if metadata['frame_count'] <= 0 or metadata['fps'] <= 0:
                raise VideoProcessingError("Invalid video format or corrupted file")

            if not self._meets_minimum_requirements(metadata):
                raise VideoProcessingError("Video does not meet minimum requirements")

            cap.release()
            return metadata

        except Exception as e:
            raise VideoProcessingError(f"Failed to extract video metadata: {str(e)}")

    def _meets_minimum_requirements(self, metadata: Dict) -> bool:
        """Check if video meets minimum processing requirements."""
        min_duration = 1.0  # minimum 1 second
        min_resolution = (32, 32)  # minimum resolution
        max_resolution = (7680, 4320)  # 8K resolution limit
        
        return (
            metadata['duration'] >= min_duration and
            metadata['resolution'][0] >= min_resolution[0] and
            metadata['resolution'][1] >= min_resolution[1] and
            metadata['resolution'][0] <= max_resolution[0] and
            metadata['resolution'][1] <= max_resolution[1] and
            metadata['frame_count'] > 0 and
            metadata['fps'] > 0
        )

    def _determine_video_format(self, video_path: str) -> str:
        """Determine video format from file extension."""
        # Simple implementation - can be enhanced with actual format detection
        return os.path.splitext(video_path)[1].lstrip('.')

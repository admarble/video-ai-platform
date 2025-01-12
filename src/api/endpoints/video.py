from fastapi import APIRouter, UploadFile, File
from pathlib import Path
from src.processors.video_processor import VideoProcessor
from src.utils.initialization import initialize_processing_environment
from src.models.video import VideoMetadata, ProcessingResults
from src.core.config import settings
from src.core.logging import LoggingManager, LogLevel

router = APIRouter()
processor = initialize_processing_environment()

# Initialize logging manager
logging_manager = LoggingManager(
    base_dir=Path(__file__).parent.parent.parent.parent,
    config={
        "access": {
            "level": "INFO",
            "format": "%(asctime)s - %(method)s %(path)s - %(status)s - %(message)s"
        }
    }
)
logger = logging_manager.get_logger("access")

@router.post("/upload", response_model=VideoMetadata)
async def upload_video(file: UploadFile = File(...)):
    """Upload and initialize video processing"""
    try:
        logger.info(
            "Video upload request",
            extra={
                "method": "POST",
                "path": "/upload",
                "status": "PROCESSING",
                "filename": file.filename
            }
        )
        
        content = await file.read()
        metadata = processor.init_video(content, filename=file.filename)
        
        logger.info(
            "Video upload successful",
            extra={
                "method": "POST",
                "path": "/upload",
                "status": "SUCCESS",
                "filename": file.filename,
                "video_id": metadata.video_id
            }
        )
        
        return metadata
        
    except Exception as e:
        logger.error(
            f"Video upload failed: {str(e)}",
            extra={
                "method": "POST",
                "path": "/upload",
                "status": "ERROR",
                "filename": file.filename
            },
            exc_info=True
        )
        raise

@router.post("/process/{video_id}", response_model=ProcessingResults)
async def process_video(video_id: str):
    """Process an uploaded video"""
    try:
        logger.info(
            "Video processing request",
            extra={
                "method": "POST",
                "path": f"/process/{video_id}",
                "status": "PROCESSING",
                "video_id": video_id
            }
        )
        
        metadata_path = Path(settings.PROCESSING_DIR) / video_id / "metadata.json"
        with open(metadata_path) as f:
            metadata = VideoMetadata.parse_raw(f.read())
        
        results = processor.process_video(metadata)
        
        logger.info(
            "Video processing successful",
            extra={
                "method": "POST",
                "path": f"/process/{video_id}",
                "status": "SUCCESS",
                "video_id": video_id
            }
        )
        
        return results
        
    except Exception as e:
        logger.error(
            f"Video processing failed: {str(e)}",
            extra={
                "method": "POST",
                "path": f"/process/{video_id}",
                "status": "ERROR",
                "video_id": video_id
            },
            exc_info=True
        )
        raise 
from fastapi import APIRouter, UploadFile, File
from pathlib import Path
from src.processors.video_processor import VideoProcessor
from src.utils.initialization import initialize_processing_environment
from src.models.video import VideoMetadata, ProcessingResults
from src.core.config import settings

router = APIRouter()
processor = initialize_processing_environment()

@router.post("/upload", response_model=VideoMetadata)
async def upload_video(file: UploadFile = File(...)):
    """Upload and initialize video processing"""
    content = await file.read()
    metadata = processor.init_video(content, filename=file.filename)
    return metadata

@router.post("/process/{video_id}", response_model=ProcessingResults)
async def process_video(video_id: str):
    """Process an uploaded video"""
    metadata_path = Path(settings.PROCESSING_DIR) / video_id / "metadata.json"
    with open(metadata_path) as f:
        metadata = VideoMetadata.parse_raw(f.read())
    
    results = processor.process_video(metadata)
    return results 
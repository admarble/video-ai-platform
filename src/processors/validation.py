from typing import List
from ..models.processing import ValidationLevel
from ..models.results import ProcessingResult

async def validate_processing_result(
    result: ProcessingResult,
    validation_level: ValidationLevel = ValidationLevel.THOROUGH
) -> List[str]:
    """Enhanced result validation with different levels"""
    validation_errors = []
    
    # Basic validation
    if validation_level >= ValidationLevel.BASIC:
        if not result.video_id:
            validation_errors.append("Missing video ID")
        if not result.scenes:
            validation_errors.append("No scenes detected")
        if not result.transcript:
            validation_errors.append("No transcript generated")

    # Thorough validation
    if validation_level >= ValidationLevel.THOROUGH:
        # Validate scene timestamps
        previous_end = 0
        for scene in result.scenes:
            if scene.start_time < previous_end:
                validation_errors.append(f"Invalid scene timing: {scene.start_time} < {previous_end}")
            previous_end = scene.end_time
            
        # Validate object detections
        for obj in result.objects:
            if not (0 <= obj.confidence <= 1):
                validation_errors.append(f"Invalid object confidence: {obj.confidence}")
            if not all(0 <= coord <= 1 for coord in obj.bbox):
                validation_errors.append(f"Invalid bounding box coordinates: {obj.bbox}")

    # Strict validation
    if validation_level >= ValidationLevel.STRICT:
        # Validate embeddings
        if not result.embeddings or not result.embeddings.get("frame_embeddings"):
            validation_errors.append("Missing frame embeddings")
        
        # Validate transcript timing
        transcript_times = sorted(result.transcript.keys())
        if transcript_times and (
            transcript_times[0] < 0 or 
            transcript_times[-1] > result.metadata.get("duration", float("inf"))
        ):
            validation_errors.append("Invalid transcript timestamps")

    return validation_errors 
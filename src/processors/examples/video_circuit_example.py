"""Example usage of video circuit breaker."""

import asyncio
from typing import Dict, Any
import logging

from ..video_circuit_config import VideoCircuitConfig
from ..video_circuit_decorator import video_circuit_breaker, CircuitBreakerRegistry
from ..video_exceptions import VideoCorruptedError, ModelError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create registry and config
registry = CircuitBreakerRegistry()
config = VideoCircuitConfig(
    failure_threshold=5,
    reset_timeout=60,
    max_processing_time=300,  # 5 minutes
    min_free_memory=2048,     # 2GB
    corruption_threshold=3,    # Open after 3 corrupt files
    model_error_threshold=5    # Open after 5 model errors
)

# Fallback function for when circuit is open
async def process_video_fallback(video_path: str) -> Dict[str, Any]:
    """Fallback handler for video processing."""
    return {
        "status": "queued",
        "message": "System under high load, video processing queued"
    }

@video_circuit_breaker(
    "video_processing",
    registry=registry,
    config=config,
    fallback=process_video_fallback
)
async def process_video(video_path: str) -> Dict[str, Any]:
    """Example video processing function with circuit breaker."""
    # Simulate video processing
    await asyncio.sleep(1)  # Simulate some work
    
    # Simulate different error scenarios
    if "corrupted" in video_path:
        raise VideoCorruptedError("Video file is corrupted")
    if "model_error" in video_path:
        raise ModelError("Model prediction failed")
        
    return {
        "status": "completed",
        "message": "Video processed successfully",
        "path": video_path
    }

async def main():
    """Example usage of video processing with circuit breaker."""
    # Process a normal video
    try:
        result = await process_video("normal_video.mp4")
        logger.info(f"Normal video result: {result}")
    except Exception as e:
        logger.error(f"Error processing normal video: {str(e)}")

    # Process corrupted videos to trigger corruption threshold
    for i in range(4):
        try:
            result = await process_video(f"corrupted_video_{i}.mp4")
            logger.info(f"Corrupted video result: {result}")
        except Exception as e:
            logger.error(f"Error processing corrupted video: {str(e)}")

    # Process with model errors to trigger model error threshold
    for i in range(6):
        try:
            result = await process_video(f"model_error_video_{i}.mp4")
            logger.info(f"Model error video result: {result}")
        except Exception as e:
            logger.error(f"Error processing model error video: {str(e)}")

    # After circuit is open, this should use fallback
    result = await process_video("another_video.mp4")
    logger.info(f"Fallback result: {result}")

    # Get circuit state
    circuit = registry.get_circuit("video_processing")
    logger.info(f"Circuit state: {circuit.get_state()}")

if __name__ == "__main__":
    asyncio.run(main()) 
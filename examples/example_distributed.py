import asyncio
import logging
from distributed_circuit_breaker import (
    DistributedCircuitConfig,
    create_distributed_circuit_breaker,
    distributed_circuit_breaker
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_video_fallback(video_path: str):
    """Fallback function when circuit is open"""
    return {"status": "queued", "message": "System under high load"}

async def main():
    # Create distributed circuit breaker config
    config = DistributedCircuitConfig(
        failure_threshold=5,
        reset_timeout=60,
        half_open_limit=3,
        window_size=60,
        success_threshold=2,
        sync_interval=5,
        state_ttl=300,
        quorum_percent=0.5,
        lock_timeout=10
    )

    # Initialize circuit breaker
    circuit = await create_distributed_circuit_breaker(
        name="video_processing",
        redis_url="redis://localhost:6379",
        config=config
    )

    # Define video processing function with circuit breaker
    @distributed_circuit_breaker(circuit, fallback=process_video_fallback)
    async def process_video(video_path: str):
        """Simulated video processing function"""
        # Simulate processing
        await asyncio.sleep(1)
        
        # Simulate random failures
        if video_path.endswith("fail"):
            raise Exception("Video processing failed")
            
        return {"status": "completed", "path": video_path}

    try:
        # Test successful processing
        result = await process_video("video1.mp4")
        logger.info(f"Success: {result}")

        # Test failures
        for i in range(6):
            try:
                await process_video(f"video{i}.fail")
            except Exception as e:
                logger.error(f"Failed: {str(e)}")

        # Circuit should be open now
        result = await process_video("video2.mp4")
        logger.info(f"Fallback: {result}")

        # Wait for reset timeout
        await asyncio.sleep(config.reset_timeout)

        # Try again - should be half-open
        result = await process_video("video3.mp4")
        logger.info(f"After reset: {result}")

    finally:
        # Cleanup
        await circuit.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 
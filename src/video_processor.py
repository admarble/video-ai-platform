from pathlib import Path
from monitoring import setup_monitoring

def process_video(video_path: str):
    # Initialize monitoring
    metrics_collector = setup_monitoring(Path("./data"))
    
    # Start processing metrics
    video_id = Path(video_path).stem
    metrics_collector.start_processing_metrics(video_id)
    
    try:
        # Your video processing logic here
        frames_processed = 0
        scene_count = 0
        object_count = 0
        
        # Update metrics periodically during processing
        metrics_collector.update_processing_metrics(
            video_id,
            frames_processed=frames_processed,
            scene_count=scene_count,
            object_count=object_count
        )
        
        # Complete processing
        metrics_collector.complete_processing_metrics(video_id)
        
    except Exception as e:
        # Record error
        metrics_collector.update_processing_metrics(
            video_id,
            error=str(e)
        )
        raise 
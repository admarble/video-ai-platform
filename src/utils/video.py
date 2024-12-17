import decord
import numpy as np
from typing import List, Optional, Tuple
import logging
from pathlib import Path
from ..exceptions import VideoProcessingError

def _calculate_adaptive_sampling_rate(
    duration: float,
    frame_height: int,
    frame_width: int,
    fps: float,
    target_memory_gb: float = 4.0
) -> int:
    """
    Calculate adaptive sampling rate based on video properties.
    
    Args:
        duration: Video duration in seconds
        frame_height: Height of video frames
        frame_width: Width of video frames
        fps: Frames per second
        target_memory_gb: Target memory usage in gigabytes
        
    Returns:
        int: Recommended sampling rate
    """
    # Calculate memory per frame (assuming 3 channels, uint8)
    bytes_per_frame = frame_height * frame_width * 3
    
    # Calculate total frames
    total_frames = int(duration * fps)
    
    # Calculate total memory needed without sampling
    total_memory_bytes = bytes_per_frame * total_frames
    
    # Calculate required sampling rate to meet target memory
    target_memory_bytes = target_memory_gb * (1024 ** 3)
    sampling_rate = max(1, int(np.ceil(total_memory_bytes / target_memory_bytes)))
    
    # Additional heuristics based on video properties
    if duration > 600:  # For videos longer than 10 minutes
        sampling_rate = max(sampling_rate, int(fps / 2))  # At least 2 frames per second
    elif duration > 3600:  # For videos longer than 1 hour
        sampling_rate = max(sampling_rate, int(fps))  # At least 1 frame per second
        
    # Adjust for high resolution videos
    if frame_height * frame_width > 1920 * 1080:
        sampling_rate = max(sampling_rate, 2)  # Sample less frequently for HD+ videos
        
    return sampling_rate

def _extract_frames(
    video_path: str,
    sampling_rate: Optional[int] = None,
    max_frames: Optional[int] = None,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    target_memory_gb: float = 4.0
) -> Tuple[np.ndarray, float]:
    """
    Extracts frames from a video file efficiently using Decord with adaptive sampling.
    
    Args:
        video_path (str): Path to the video file
        sampling_rate (int, optional): Extract every nth frame. If None, calculated automatically.
        max_frames (int, optional): Maximum number of frames to extract. Defaults to None.
        start_time (float, optional): Start time in seconds. Defaults to 0.0.
        end_time (float, optional): End time in seconds. Defaults to None.
        target_memory_gb (float, optional): Target memory usage in GB. Defaults to 4.0.
    
    Returns:
        Tuple[np.ndarray, float]: Tuple containing:
            - numpy array of frames with shape (num_frames, height, width, channels)
            - fps of the video
    
    Raises:
        VideoProcessingError: If video cannot be loaded or frames cannot be extracted
    """
    try:
        # Verify file exists
        if not Path(video_path).exists():
            raise VideoProcessingError(f"Video file not found: {video_path}")
            
        # Load video
        video_reader = decord.VideoReader(
            video_path,
            ctx=decord.cpu(0)  # Use CPU context - can be changed to GPU if available
        )
        
        # Get video metadata
        total_frames = len(video_reader)
        fps = video_reader.get_avg_fps()
        duration = total_frames / fps
        height, width = video_reader[0].shape[:2]
        
        # Calculate adaptive sampling rate if not provided
        if sampling_rate is None:
            sampling_rate = _calculate_adaptive_sampling_rate(
                duration=duration,
                frame_height=height,
                frame_width=width,
                fps=fps,
                target_memory_gb=target_memory_gb
            )
            logging.info(f"Using adaptive sampling rate: {sampling_rate}")
        
        # Calculate frame indices based on time range
        start_idx = int(start_time * fps) if start_time > 0 else 0
        end_idx = min(
            int(end_time * fps) if end_time is not None else total_frames,
            total_frames
        )
        
        # Generate frame indices with sampling rate
        frame_indices = list(range(start_idx, end_idx, sampling_rate))
        
        # Apply max_frames limit if specified
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]
        
        if not frame_indices:
            raise VideoProcessingError("No frames to extract with current parameters")
            
        # Extract frames
        frames = video_reader.get_batch(frame_indices)
        
        # Convert to numpy array and ensure correct format (uint8)
        frames_np = frames.asnumpy()
        
        # Verify extracted frames
        if frames_np.size == 0:
            raise VideoProcessingError("Failed to extract any frames")
            
        effective_fps = fps / sampling_rate
        logging.info(
            f"Successfully extracted {len(frame_indices)} frames from {video_path}\n"
            f"Original FPS: {fps:.2f}, Effective FPS: {effective_fps:.2f}\n"
            f"Sampling rate: {sampling_rate}, Memory usage estimate: "
            f"{(frames_np.nbytes / (1024**3)):.2f}GB"
        )
        
        return frames_np, fps
        
    except decord.DECORDError as e:
        raise VideoProcessingError(f"Decord error while processing video: {str(e)}")
    except Exception as e:
        raise VideoProcessingError(f"Unexpected error while extracting frames: {str(e)}") 
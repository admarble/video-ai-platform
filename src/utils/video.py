import cv2
import numpy as np
from typing import Tuple, List
from pathlib import Path

def extract_frames(
    video_path: str,
    sampling_rate: int = 1
) -> Tuple[List[np.ndarray], float]:
    """
    Extract frames from video at given sampling rate
    
    Args:
        video_path: Path to video file
        sampling_rate: Extract every nth frame
        
    Returns:
        Tuple of (list of frames, fps)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % sampling_rate == 0:
            frames.append(frame)
        frame_count += 1
        
    cap.release()
    return frames, fps

def save_frame(frame: np.ndarray, output_path: Path) -> None:
    """Save a single frame to disk"""
    cv2.imwrite(str(output_path), frame) 
import cv2
import numpy as np
from pathlib import Path
import sys
import logging
from typing import List, Tuple

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.ai.processors.integrated_processor import (
    IntegratedProcessor,
    IntegratedSettings,
    TimesformerSettings,
    YOLOSettings,
    WhisperSettings
)

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_video(video_path: str) -> Tuple[np.ndarray, str, float]:
    """Load video frames and extract audio."""
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read frames
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    # Extract audio using ffmpeg
    audio_path = str(Path(video_path).with_suffix('.wav'))
    import subprocess
    subprocess.run([
        'ffmpeg', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', '16000', '-ac', '1',
        audio_path
    ], check=True)
    
    return np.array(frames), audio_path, fps

def process_video(
    video_path: str,
    output_dir: str,
    device: str = 'cuda'
) -> None:
    """Process a video file using the integrated processor."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load video and extract audio
    logging.info(f"Loading video: {video_path}")
    frames, audio_path, fps = load_video(video_path)
    
    # Initialize settings
    settings = IntegratedSettings(
        timesformer=TimesformerSettings(
            num_frames=8,
            image_size=224
        ),
        yolo=YOLOSettings(
            model_size='x',
            conf_threshold=0.25
        ),
        whisper=WhisperSettings(
            model_size='base',
            compute_type='float16'
        ),
        device=device
    )
    
    # Process video
    logging.info("Processing video with integrated processor")
    with IntegratedProcessor(settings) as processor:
        results = processor.process_video(frames, audio_path, fps)
    
    # Save results
    import json
    output_file = output_path / 'results.json'
    with open(output_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        processed_results = process_results_for_json(results)
        json.dump(processed_results, f, indent=2)
    
    logging.info(f"Results saved to: {output_file}")
    
    # Clean up audio file
    Path(audio_path).unlink()

def process_results_for_json(results: dict) -> dict:
    """Process results to make them JSON serializable."""
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return obj
    
    return convert_numpy(results)

def main():
    """Main function."""
    setup_logging()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Process video with integrated AI models')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output-dir', default='output',
                      help='Directory to save results (default: output)')
    parser.add_argument('--device', default='cuda',
                      help='Device to use (cuda or cpu, default: cuda)')
    args = parser.parse_args()
    
    try:
        process_video(args.video_path, args.output_dir, args.device)
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 
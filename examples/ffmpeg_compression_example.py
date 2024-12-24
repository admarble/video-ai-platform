from pathlib import Path
import sys
from typing import Optional
from src.processors.adaptive_compression import (
    create_compression_tuner,
    CompressionMetric,
    OptimizationStrategy,
    OptimizationConfig
)
from src.processors.ffmpeg_processor import FFmpegProcessor, VideoMetadata
from src.utils.logging_config import setup_logging, get_logger

def print_progress(current_time: float, total_time: float):
    """Print compression progress"""
    progress = min(100, (current_time / total_time) * 100)
    sys.stdout.write(f"\rProgress: {progress:.1f}%")
    sys.stdout.flush()

def compress_video(
    input_path: Path,
    output_path: Path,
    target_quality: float = 0.8,
    target_speed: float = 0.6,
    target_size_reduction: float = 0.7,
    log_file: Optional[Path] = None
):
    """Compress a video with adaptive optimization"""
    # Set up logging
    setup_logging(
        log_level="DEBUG" if "--debug" in sys.argv else "INFO",
        log_file=log_file
    )
    logger = get_logger(__name__)
    
    logger.info(f"Starting video compression process")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(
        f"Target metrics - Quality: {target_quality}, "
        f"Speed: {target_speed}, "
        f"Size Reduction: {target_size_reduction}"
    )

    # Initialize FFmpeg processor
    try:
        ffmpeg = FFmpegProcessor()
    except Exception as e:
        logger.error(f"Error initializing FFmpeg: {str(e)}")
        return

    # Get video information
    try:
        video_info = ffmpeg.get_video_info(input_path)
        logger.info("\nInput Video Information:")
        logger.info(f"Resolution: {video_info.width}x{video_info.height}")
        logger.info(f"Duration: {video_info.duration:.2f} seconds")
        logger.info(f"Bitrate: {video_info.bitrate}")
        logger.info(f"FPS: {video_info.fps}")
        logger.info(f"Format: {video_info.format}")
        logger.info(f"Size: {video_info.size / (1024*1024):.2f} MB")
    except Exception as e:
        logger.error(f"Error getting video info: {str(e)}")
        return

    # Create tuner with adaptive optimization
    config = OptimizationConfig(
        strategy=OptimizationStrategy.ADAPTIVE,
        learning_rate=0.1,
        history_weight_decay=0.9,
        gradient_momentum=0.9,
        min_samples=5,
        max_adjustment_per_step=0.2
    )
    logger.debug(f"Creating compression tuner with config: {config}")
    tuner = create_compression_tuner(optimization_config=config)

    # Set up compression parameters
    video_info_dict = {
        'resolution': (video_info.width, video_info.height),
        'has_motion': True,  # Could be detected from video analysis
        'duration': video_info.duration
    }

    target_metrics = {
        CompressionMetric.QUALITY: target_quality,
        CompressionMetric.SPEED: target_speed,
        CompressionMetric.SIZE_REDUCTION: target_size_reduction
    }

    # Select initial profile
    logger.info("Selecting compression profile...")
    profile = tuner.select_profile(video_info_dict, target_metrics)
    logger.info(f"Selected Profile: {profile.name}")
    logger.info(f"Video Codec: {profile.video_codec.name}")
    logger.info(f"Audio Codec: {profile.audio_codec.name}")
    logger.info(f"Container: {profile.container_format.value}")
    logger.info(f"Quality Value: {profile.quality_value}")
    logger.info(f"Preset: {profile.preset}")

    # Compress video with progress monitoring
    try:
        def progress_callback(current_time: float):
            print_progress(current_time, video_info.duration)

        logger.info("Starting compression...")
        metrics = ffmpeg.compress_video(
            input_path,
            output_path,
            profile,
            progress_callback
        )

        logger.info("\nCompression completed!")
        logger.info("\nAchieved Metrics:")
        logger.info(f"Quality Score: {metrics[CompressionMetric.QUALITY]:.2f}")
        logger.info(f"Speed Score: {metrics[CompressionMetric.SPEED]:.2f}")
        logger.info(f"Size Reduction: {metrics[CompressionMetric.SIZE_REDUCTION]:.2f}")

        # Get compressed video information
        output_info = ffmpeg.get_video_info(output_path)
        size_reduction = (1 - (output_info.size / video_info.size)) * 100

        logger.info("\nOutput Video Information:")
        logger.info(f"Size: {output_info.size / (1024*1024):.2f} MB")
        logger.info(f"Size Reduction: {size_reduction:.1f}%")
        logger.info(f"Bitrate: {output_info.bitrate}")

        # Update profile performance
        logger.debug("Updating profile performance metrics")
        tuner.update_profile_performance(profile.name, metrics)

        # Optimize profile for future use
        logger.info("Optimizing profile for future compressions...")
        optimized_profile = tuner.optimize_profile(profile.name, target_metrics)
        logger.info("Optimized profile settings:")
        logger.info(f"Quality Value: {optimized_profile.quality_value}")
        logger.info(f"Preset: {optimized_profile.preset}")
        logger.info(f"Video Bitrate: {optimized_profile.video_bitrate}")

    except Exception as e:
        logger.error(f"Error during compression: {str(e)}", exc_info=True)

def main():
    if len(sys.argv) < 3:
        print("Usage: python ffmpeg_compression_example.py input_video output_video [options]")
        print("Options:")
        print("  --quality FLOAT     Target quality (0-1), default: 0.8")
        print("  --speed FLOAT       Target speed (0-1), default: 0.6")
        print("  --reduction FLOAT   Target size reduction (0-1), default: 0.7")
        print("  --debug            Enable debug logging")
        print("  --log-file PATH    Path to log file")
        return

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    # Parse optional arguments
    quality = 0.8
    speed = 0.6
    reduction = 0.7
    log_file = None

    args = sys.argv[3:]
    i = 0
    while i < len(args):
        if args[i] == "--quality" and i + 1 < len(args):
            quality = float(args[i + 1])
            i += 2
        elif args[i] == "--speed" and i + 1 < len(args):
            speed = float(args[i + 1])
            i += 2
        elif args[i] == "--reduction" and i + 1 < len(args):
            reduction = float(args[i + 1])
            i += 2
        elif args[i] == "--log-file" and i + 1 < len(args):
            log_file = Path(args[i + 1])
            i += 2
        else:
            i += 1

    compress_video(
        input_path,
        output_path,
        target_quality=quality,
        target_speed=speed,
        target_size_reduction=reduction,
        log_file=log_file
    )

if __name__ == "__main__":
    main() 
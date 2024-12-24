#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
from typing import Optional, Dict, Any

from src.processors.adaptive_compression import (
    create_compression_tuner,
    CompressionMetric,
    OptimizationStrategy,
    OptimizationConfig,
    VideoCodec,
    AudioCodec,
    ContainerFormat,
    CompressionProfile
)
from src.processors.ffmpeg_processor import FFmpegProcessor
from src.utils.logging_config import setup_logging, get_logger

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with all options"""
    parser = argparse.ArgumentParser(
        description="Adaptive Video Compression CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output
    parser.add_argument(
        "input",
        type=Path,
        help="Input video file path"
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output video file path"
    )
    
    # Compression targets
    target_group = parser.add_argument_group("Compression Targets")
    target_group.add_argument(
        "--quality",
        type=float,
        default=0.8,
        help="Target quality (0-1)"
    )
    target_group.add_argument(
        "--speed",
        type=float,
        default=0.6,
        help="Target speed (0-1)"
    )
    target_group.add_argument(
        "--size-reduction",
        type=float,
        default=0.7,
        help="Target size reduction (0-1)"
    )
    
    # Codec options
    codec_group = parser.add_argument_group("Codec Options")
    codec_group.add_argument(
        "--video-codec",
        type=str,
        choices=[c.value for c in VideoCodec],
        default=VideoCodec.H264.value,
        help="Video codec to use"
    )
    codec_group.add_argument(
        "--audio-codec",
        type=str,
        choices=[c.value for c in AudioCodec],
        default=AudioCodec.AAC.value,
        help="Audio codec to use"
    )
    codec_group.add_argument(
        "--container",
        type=str,
        choices=[f.value for f in ContainerFormat],
        default=ContainerFormat.MP4.value,
        help="Container format"
    )
    codec_group.add_argument(
        "--preset",
        type=str,
        default="medium",
        help="Encoder preset (e.g., ultrafast, fast, medium, slow)"
    )
    codec_group.add_argument(
        "--video-bitrate",
        type=str,
        help="Video bitrate (e.g., 2M, 5M)"
    )
    codec_group.add_argument(
        "--audio-bitrate",
        type=str,
        default="128k",
        help="Audio bitrate"
    )
    codec_group.add_argument(
        "--multipass",
        action="store_true",
        help="Enable multipass encoding"
    )
    
    # Optimization options
    opt_group = parser.add_argument_group("Optimization Options")
    opt_group.add_argument(
        "--optimization-strategy",
        type=str,
        choices=[s.name for s in OptimizationStrategy],
        default=OptimizationStrategy.ADAPTIVE.name,
        help="Optimization strategy"
    )
    opt_group.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for optimization"
    )
    opt_group.add_argument(
        "--history-weight",
        type=float,
        default=0.9,
        help="Weight decay for historical performance"
    )
    
    # Logging options
    log_group = parser.add_argument_group("Logging Options")
    log_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    log_group.add_argument(
        "--log-file",
        type=Path,
        help="Log file path"
    )
    log_group.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    return parser

def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments"""
    # Check if input file exists
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Check if output directory exists
    output_dir = args.output.parent
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    # Validate metric ranges
    for metric, value in [
        ("quality", args.quality),
        ("speed", args.speed),
        ("size reduction", args.size_reduction)
    ]:
        if not 0 <= value <= 1:
            raise ValueError(
                f"Invalid {metric} value: {value}. Must be between 0 and 1"
            )
    
    # Validate learning rate and history weight
    if not 0 < args.learning_rate <= 1:
        raise ValueError(
            f"Invalid learning rate: {args.learning_rate}. Must be between 0 and 1"
        )
    if not 0 <= args.history_weight <= 1:
        raise ValueError(
            f"Invalid history weight: {args.history_weight}. Must be between 0 and 1"
        )

def print_progress(
    current_time: float,
    total_time: float,
    quiet: bool = False
) -> None:
    """Print compression progress"""
    if quiet:
        return
    progress = min(100, (current_time / total_time) * 100)
    sys.stdout.write(f"\rProgress: {progress:.1f}%")
    sys.stdout.flush()

def create_optimization_config(args: argparse.Namespace) -> OptimizationConfig:
    """Create optimization configuration from arguments"""
    return OptimizationConfig(
        strategy=OptimizationStrategy[args.optimization_strategy],
        learning_rate=args.learning_rate,
        history_weight_decay=args.history_weight,
        gradient_momentum=0.9,
        min_samples=5,
        max_adjustment_per_step=0.2
    )

def create_compression_profile(args: argparse.Namespace) -> CompressionProfile:
    """Create compression profile from arguments"""
    return CompressionProfile(
        name="cli_profile",
        video_codec=VideoCodec(args.video_codec),
        audio_codec=AudioCodec(args.audio_codec),
        container_format=ContainerFormat(args.container),
        video_bitrate=args.video_bitrate or "2M",
        audio_bitrate=args.audio_bitrate,
        preset=args.preset,
        quality_value=23,  # Default value, will be optimized
        multipass=args.multipass
    )

def main() -> None:
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        validate_args(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Set up logging
    setup_logging(
        log_level="DEBUG" if args.debug else "INFO",
        log_file=args.log_file
    )
    logger = get_logger(__name__)
    
    logger.info("Starting video compression")
    logger.debug(f"Arguments: {args}")
    
    try:
        # Initialize FFmpeg processor
        ffmpeg = FFmpegProcessor()
        
        # Get video information
        video_info = ffmpeg.get_video_info(args.input)
        logger.info("Input Video Information:")
        logger.info(f"Resolution: {video_info.width}x{video_info.height}")
        logger.info(f"Duration: {video_info.duration:.2f} seconds")
        logger.info(f"Size: {video_info.size / (1024*1024):.2f} MB")
        
        # Create optimization config and tuner
        opt_config = create_optimization_config(args)
        tuner = create_compression_tuner(optimization_config=opt_config)
        
        # Set up compression parameters
        video_info_dict = {
            'resolution': (video_info.width, video_info.height),
            'has_motion': True,
            'duration': video_info.duration
        }
        
        target_metrics = {
            CompressionMetric.QUALITY: args.quality,
            CompressionMetric.SPEED: args.speed,
            CompressionMetric.SIZE_REDUCTION: args.size_reduction
        }
        
        # Get initial profile
        if args.video_codec:
            # Use user-specified profile
            profile = create_compression_profile(args)
            logger.info("Using user-specified compression profile")
        else:
            # Let tuner select profile
            profile = tuner.select_profile(video_info_dict, target_metrics)
            logger.info("Using automatically selected compression profile")
        
        logger.info(f"Compression Profile:")
        logger.info(f"Video Codec: {profile.video_codec.name}")
        logger.info(f"Audio Codec: {profile.audio_codec.name}")
        logger.info(f"Container: {profile.container_format.value}")
        logger.info(f"Preset: {profile.preset}")
        
        # Compress video
        def progress_callback(current_time: float):
            print_progress(current_time, video_info.duration, args.quiet)
            
        metrics = ffmpeg.compress_video(
            args.input,
            args.output,
            profile,
            progress_callback
        )
        
        # Print results
        if not args.quiet:
            print("\n")  # New line after progress
        logger.info("Compression Results:")
        logger.info(f"Quality Score: {metrics[CompressionMetric.QUALITY]:.2f}")
        logger.info(f"Speed Score: {metrics[CompressionMetric.SPEED]:.2f}")
        logger.info(f"Size Reduction: {metrics[CompressionMetric.SIZE_REDUCTION]:.2f}")
        
        # Get output information
        output_info = ffmpeg.get_video_info(args.output)
        size_reduction = (1 - (output_info.size / video_info.size)) * 100
        
        logger.info("\nOutput Video Information:")
        logger.info(f"Size: {output_info.size / (1024*1024):.2f} MB")
        logger.info(f"Size Reduction: {size_reduction:.1f}%")
        logger.info(f"Bitrate: {output_info.bitrate}")
        
    except Exception as e:
        logger.error(f"Error during compression: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 
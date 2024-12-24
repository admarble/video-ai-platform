#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import json
from typing import Optional, Dict, Any, List

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
from src.processors.ffmpeg_processor import FFmpegProcessor, VideoMetadata
from src.utils.logging_config import setup_logging, get_logger

def setup_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to parser"""
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Log file path"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

def compress_command(args: argparse.Namespace) -> None:
    """Handle compress subcommand"""
    logger = get_logger(__name__)
    
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
        opt_config = OptimizationConfig(
            strategy=OptimizationStrategy[args.optimization_strategy],
            learning_rate=args.learning_rate,
            history_weight_decay=args.history_weight,
            gradient_momentum=0.9,
            min_samples=5,
            max_adjustment_per_step=0.2
        )
        tuner = create_compression_tuner(optimization_config=opt_config)
        
        # Load profile if specified
        if args.profile:
            with open(args.profile) as f:
                profile_data = json.load(f)
                profile = CompressionProfile(**profile_data)
                logger.info(f"Loaded profile from {args.profile}")
        else:
            # Create profile from arguments or use auto-selection
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
            
            profile = tuner.select_profile(video_info_dict, target_metrics)
            
        logger.info("Compression Profile:")
        logger.info(f"Video Codec: {profile.video_codec.name}")
        logger.info(f"Audio Codec: {profile.audio_codec.name}")
        logger.info(f"Container: {profile.container_format.value}")
        logger.info(f"Preset: {profile.preset}")
        
        # Compress video
        def progress_callback(current_time: float):
            if not args.quiet:
                progress = min(100, (current_time / video_info.duration) * 100)
                sys.stdout.write(f"\rProgress: {progress:.1f}%")
                sys.stdout.flush()
                
        metrics = ffmpeg.compress_video(
            args.input,
            args.output,
            profile,
            progress_callback
        )
        
        if not args.quiet:
            print("\n")
            
        logger.info("Compression Results:")
        logger.info(f"Quality Score: {metrics[CompressionMetric.QUALITY]:.2f}")
        logger.info(f"Speed Score: {metrics[CompressionMetric.SPEED]:.2f}")
        logger.info(f"Size Reduction: {metrics[CompressionMetric.SIZE_REDUCTION]:.2f}")
        
        # Save metrics if requested
        if args.save_metrics:
            metrics_data = {
                "input": str(args.input),
                "output": str(args.output),
                "profile": profile.__dict__,
                "metrics": {k.name: v for k, v in metrics.items()},
                "video_info": {
                    "input": {
                        "width": video_info.width,
                        "height": video_info.height,
                        "duration": video_info.duration,
                        "size": video_info.size
                    }
                }
            }
            
            # Add output video info
            output_info = ffmpeg.get_video_info(args.output)
            metrics_data["video_info"]["output"] = {
                "width": output_info.width,
                "height": output_info.height,
                "duration": output_info.duration,
                "size": output_info.size
            }
            
            with open(args.save_metrics, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            logger.info(f"Saved metrics to {args.save_metrics}")
            
    except Exception as e:
        logger.error(f"Error during compression: {str(e)}", exc_info=True)
        sys.exit(1)

def analyze_command(args: argparse.Namespace) -> None:
    """Handle analyze subcommand"""
    logger = get_logger(__name__)
    
    try:
        ffmpeg = FFmpegProcessor()
        video_info = ffmpeg.get_video_info(args.input)
        
        analysis = {
            "path": str(args.input),
            "video": {
                "resolution": f"{video_info.width}x{video_info.height}",
                "duration": f"{video_info.duration:.2f}s",
                "fps": video_info.fps,
                "bitrate": video_info.bitrate,
                "size_mb": video_info.size / (1024*1024),
                "format": video_info.format,
                "has_audio": video_info.has_audio
            }
        }
        
        if args.json:
            print(json.dumps(analysis, indent=2))
        else:
            print("\nVideo Analysis:")
            print(f"Path: {analysis['path']}")
            print("\nVideo Properties:")
            for key, value in analysis["video"].items():
                print(f"{key.replace('_', ' ').title()}: {value}")
                
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}", exc_info=True)
        sys.exit(1)

def profile_command(args: argparse.Namespace) -> None:
    """Handle profile subcommand"""
    logger = get_logger(__name__)
    
    if args.action == "create":
        try:
            profile = CompressionProfile(
                name=args.name,
                video_codec=VideoCodec(args.video_codec),
                audio_codec=AudioCodec(args.audio_codec),
                container_format=ContainerFormat(args.container),
                video_bitrate=args.video_bitrate,
                audio_bitrate=args.audio_bitrate,
                preset=args.preset,
                quality_value=args.quality_value,
                multipass=args.multipass
            )
            
            profile_data = {
                k: v.value if hasattr(v, 'value') else v
                for k, v in profile.__dict__.items()
            }
            
            with open(args.output, 'w') as f:
                json.dump(profile_data, f, indent=2)
            logger.info(f"Created profile: {args.output}")
            
        except Exception as e:
            logger.error(f"Error creating profile: {str(e)}", exc_info=True)
            sys.exit(1)
            
    elif args.action == "show":
        try:
            with open(args.profile) as f:
                profile_data = json.load(f)
                
            if args.json:
                print(json.dumps(profile_data, indent=2))
            else:
                print("\nCompression Profile:")
                for key, value in profile_data.items():
                    print(f"{key.replace('_', ' ').title()}: {value}")
                    
        except Exception as e:
            logger.error(f"Error reading profile: {str(e)}", exc_info=True)
            sys.exit(1)

def create_parser() -> argparse.ArgumentParser:
    """Create main argument parser"""
    parser = argparse.ArgumentParser(
        description="Cuthrough - Adaptive Video Compression Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    setup_common_args(parser)
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Compress command
    compress = subparsers.add_parser(
        "compress",
        help="Compress a video file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    compress.add_argument(
        "input",
        type=Path,
        help="Input video file"
    )
    compress.add_argument(
        "output",
        type=Path,
        help="Output video file"
    )
    compress.add_argument(
        "--profile",
        type=Path,
        help="Path to compression profile JSON"
    )
    compress.add_argument(
        "--quality",
        type=float,
        default=0.8,
        help="Target quality (0-1)"
    )
    compress.add_argument(
        "--speed",
        type=float,
        default=0.6,
        help="Target speed (0-1)"
    )
    compress.add_argument(
        "--size-reduction",
        type=float,
        default=0.7,
        help="Target size reduction (0-1)"
    )
    compress.add_argument(
        "--optimization-strategy",
        type=str,
        choices=[s.name for s in OptimizationStrategy],
        default=OptimizationStrategy.ADAPTIVE.name,
        help="Optimization strategy"
    )
    compress.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for optimization"
    )
    compress.add_argument(
        "--history-weight",
        type=float,
        default=0.9,
        help="Weight decay for historical performance"
    )
    compress.add_argument(
        "--save-metrics",
        type=Path,
        help="Save compression metrics to JSON file"
    )
    setup_common_args(compress)
    
    # Analyze command
    analyze = subparsers.add_parser(
        "analyze",
        help="Analyze a video file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    analyze.add_argument(
        "input",
        type=Path,
        help="Video file to analyze"
    )
    analyze.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    setup_common_args(analyze)
    
    # Profile command
    profile = subparsers.add_parser(
        "profile",
        help="Manage compression profiles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    profile_subparsers = profile.add_subparsers(
        dest="action",
        help="Profile action"
    )
    
    # Create profile
    create_profile = profile_subparsers.add_parser(
        "create",
        help="Create a new compression profile"
    )
    create_profile.add_argument(
        "output",
        type=Path,
        help="Output profile JSON file"
    )
    create_profile.add_argument(
        "--name",
        required=True,
        help="Profile name"
    )
    create_profile.add_argument(
        "--video-codec",
        type=str,
        choices=[c.value for c in VideoCodec],
        default=VideoCodec.H264.value,
        help="Video codec"
    )
    create_profile.add_argument(
        "--audio-codec",
        type=str,
        choices=[c.value for c in AudioCodec],
        default=AudioCodec.AAC.value,
        help="Audio codec"
    )
    create_profile.add_argument(
        "--container",
        type=str,
        choices=[f.value for f in ContainerFormat],
        default=ContainerFormat.MP4.value,
        help="Container format"
    )
    create_profile.add_argument(
        "--preset",
        default="medium",
        help="Encoder preset"
    )
    create_profile.add_argument(
        "--video-bitrate",
        default="2M",
        help="Video bitrate"
    )
    create_profile.add_argument(
        "--audio-bitrate",
        default="128k",
        help="Audio bitrate"
    )
    create_profile.add_argument(
        "--quality-value",
        type=int,
        default=23,
        help="Quality value (e.g., CRF)"
    )
    create_profile.add_argument(
        "--multipass",
        action="store_true",
        help="Enable multipass encoding"
    )
    setup_common_args(create_profile)
    
    # Show profile
    show_profile = profile_subparsers.add_parser(
        "show",
        help="Show compression profile"
    )
    show_profile.add_argument(
        "profile",
        type=Path,
        help="Profile JSON file"
    )
    show_profile.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    setup_common_args(show_profile)
    
    return parser

def main() -> None:
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    # Set up logging
    setup_logging(
        log_level="DEBUG" if args.debug else "INFO",
        log_file=args.log_file
    )
    
    # Execute command
    if args.command == "compress":
        compress_command(args)
    elif args.command == "analyze":
        analyze_command(args)
    elif args.command == "profile":
        profile_command(args)

if __name__ == "__main__":
    main() 
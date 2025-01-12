import subprocess
import json
import shlex
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import tempfile
import os
import time

from ..utils.logging_config import get_logger
from .adaptive_compression import (
    CompressionProfile,
    CompressionMetric,
    VideoCodec,
    AudioCodec,
    ContainerFormat
)

logger = get_logger(__name__)

class FFmpegError(Exception):
    """Base exception for FFmpeg-related errors"""
    pass

class FFmpegNotFoundError(FFmpegError):
    """Raised when FFmpeg is not installed or not found in PATH"""
    pass

@dataclass
class VideoMetadata:
    """Video file metadata"""
    width: int
    height: int
    duration: float
    bitrate: str
    fps: float
    has_audio: bool
    format: str
    size: int

class FFmpegProcessor:
    """Handles video compression using FFmpeg"""
    
    def __init__(self, ffmpeg_path: Optional[str] = None):
        self.logger = logger.getChild("processor")
        self.ffmpeg_path = ffmpeg_path or self._find_ffmpeg()
        if not self.ffmpeg_path:
            self.logger.error("FFmpeg not found in PATH")
            raise FFmpegNotFoundError("FFmpeg not found in PATH")
        self.logger.info(f"Initialized FFmpeg processor with path: {self.ffmpeg_path}")
            
    def _find_ffmpeg(self) -> Optional[str]:
        """Find FFmpeg executable in PATH"""
        try:
            result = subprocess.run(
                ["which", "ffmpeg"],
                capture_output=True,
                text=True,
                check=True
            )
            ffmpeg_path = result.stdout.strip()
            self.logger.debug(f"Found FFmpeg at: {ffmpeg_path}")
            return ffmpeg_path
        except subprocess.CalledProcessError:
            self.logger.warning("FFmpeg not found in PATH")
            return None
            
    def get_video_info(self, input_path: Path) -> VideoMetadata:
        """Get video file metadata using FFprobe"""
        self.logger.info(f"Getting video info for: {input_path}")
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(input_path)
            ]
            
            self.logger.debug(f"Running FFprobe command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = next(
                stream for stream in data["streams"]
                if stream["codec_type"] == "video"
            )
            
            # Check for audio stream
            has_audio = any(
                stream["codec_type"] == "audio"
                for stream in data["streams"]
            )
            
            metadata = VideoMetadata(
                width=int(video_stream["width"]),
                height=int(video_stream["height"]),
                duration=float(data["format"]["duration"]),
                bitrate=data["format"]["bit_rate"],
                fps=eval(video_stream["r_frame_rate"]),
                has_audio=has_audio,
                format=data["format"]["format_name"],
                size=int(data["format"]["size"])
            )
            
            self.logger.debug(f"Video metadata: {metadata}")
            return metadata
            
        except (subprocess.CalledProcessError, KeyError, json.JSONDecodeError) as e:
            error_msg = f"Error getting video info: {str(e)}"
            self.logger.error(error_msg)
            raise FFmpegError(error_msg)
            
    def _build_ffmpeg_command(
        self,
        input_path: Path,
        output_path: Path,
        profile: CompressionProfile,
        pass_number: Optional[int] = None
    ) -> List[str]:
        """Build FFmpeg command based on compression profile"""
        self.logger.debug(
            f"Building FFmpeg command for profile: {profile.name}, "
            f"pass: {pass_number if pass_number else 'single'}"
        )
        
        cmd = [
            self.ffmpeg_path,
            "-y",  # Overwrite output file
            "-i", str(input_path)
        ]
        
        # Video codec settings
        cmd.extend(["-c:v", profile.video_codec.value])
        
        # Quality settings based on codec
        if profile.video_codec in [VideoCodec.H264, VideoCodec.H265]:
            cmd.extend([
                f"-preset", profile.preset,
                f"-crf", str(profile.quality_value)
            ])
        elif profile.video_codec == VideoCodec.VP9:
            cmd.extend([
                f"-deadline", profile.preset,
                f"-cq-level", str(profile.quality_value),
                "-b:v", profile.video_bitrate
            ])
        elif profile.video_codec == VideoCodec.AV1:
            cmd.extend([
                f"-cpu-used", profile.preset,
                f"-crf", str(profile.quality_value)
            ])
            
        # Multipass encoding
        if profile.multipass and pass_number is not None:
            if profile.video_codec in [VideoCodec.VP9, VideoCodec.AV1]:
                pass_log = str(tempfile.mktemp())
                cmd.extend([
                    "-pass", str(pass_number),
                    "-passlogfile", pass_log
                ])
            else:
                cmd.extend([
                    "-pass", str(pass_number),
                    "-passlogfile", str(output_path.with_suffix('.log'))
                ])
                
        # Audio codec settings
        if profile.audio_codec != AudioCodec.COPY:
            cmd.extend([
                "-c:a", profile.audio_codec.value,
                "-b:a", profile.audio_bitrate
            ])
            
        # Output format
        cmd.extend([
            "-f", profile.container_format.value,
            str(output_path)
        ])
        
        self.logger.debug(f"Generated FFmpeg command: {' '.join(cmd)}")
        return cmd
        
    def compress_video(
        self,
        input_path: Path,
        output_path: Path,
        profile: CompressionProfile,
        progress_callback: Optional[callable] = None
    ) -> Dict[CompressionMetric, float]:
        """Compress video using the specified profile"""
        self.logger.info(
            f"Starting video compression: {input_path} -> {output_path} "
            f"using profile: {profile.name}"
        )
        
        start_time = time.time()
        input_info = self.get_video_info(input_path)
        
        try:
            if profile.multipass:
                self.logger.info("Starting multipass encoding")
                # First pass
                cmd_pass1 = self._build_ffmpeg_command(
                    input_path, output_path, profile, pass_number=1
                )
                self._run_ffmpeg(cmd_pass1, progress_callback)
                
                # Second pass
                cmd_pass2 = self._build_ffmpeg_command(
                    input_path, output_path, profile, pass_number=2
                )
                self._run_ffmpeg(cmd_pass2, progress_callback)
            else:
                self.logger.info("Starting single-pass encoding")
                # Single pass
                cmd = self._build_ffmpeg_command(
                    input_path, output_path, profile
                )
                self._run_ffmpeg(cmd, progress_callback)
                
            # Calculate metrics
            end_time = time.time()
            output_info = self.get_video_info(output_path)
            
            processing_time = end_time - start_time
            target_duration = float(input_info.duration)
            
            # Calculate metrics
            metrics = {
                CompressionMetric.QUALITY: self._calculate_quality_score(
                    input_path, output_path
                ),
                CompressionMetric.SPEED: min(
                    1.0,
                    target_duration / max(processing_time, 1)
                ),
                CompressionMetric.SIZE_REDUCTION: 1.0 - (
                    output_info.size / input_info.size
                )
            }
            
            self.logger.info(
                f"Compression completed. Metrics: "
                f"Quality={metrics[CompressionMetric.QUALITY]:.2f}, "
                f"Speed={metrics[CompressionMetric.SPEED]:.2f}, "
                f"Size Reduction={metrics[CompressionMetric.SIZE_REDUCTION]:.2f}"
            )
            
            return metrics
            
        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg compression failed: {e.stderr}"
            self.logger.error(error_msg)
            raise FFmpegError(error_msg)
            
    def _run_ffmpeg(
        self,
        cmd: List[str],
        progress_callback: Optional[callable] = None
    ) -> None:
        """Run FFmpeg command with progress monitoring"""
        self.logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        while True:
            if process.stderr is None:
                break
                
            line = process.stderr.readline()
            if not line:
                break
                
            # Log FFmpeg output at debug level
            self.logger.debug(f"FFmpeg: {line.strip()}")
                
            # Parse progress information
            if "time=" in line and progress_callback:
                try:
                    time_str = line.split("time=")[1].split()[0]
                    hours, minutes, seconds = map(
                        float, time_str.split(':')
                    )
                    current_time = hours * 3600 + minutes * 60 + seconds
                    progress_callback(current_time)
                except (IndexError, ValueError):
                    self.logger.warning(
                        f"Failed to parse progress from line: {line.strip()}"
                    )
                    
        process.wait()
        if process.returncode != 0:
            error_msg = f"FFmpeg process failed with return code: {process.returncode}"
            self.logger.error(error_msg)
            raise subprocess.CalledProcessError(
                process.returncode, cmd
            )
            
    def _calculate_quality_score(
        self,
        original_path: Path,
        compressed_path: Path
    ) -> float:
        """Calculate video quality score using VMAF or SSIM"""
        self.logger.info("Calculating video quality score")
        
        try:
            # Use VMAF if available
            self.logger.debug("Attempting VMAF measurement")
            cmd = [
                "ffmpeg",
                "-i", str(original_path),
                "-i", str(compressed_path),
                "-filter_complex", "[0:v][1:v]libvmaf",
                "-f", "null",
                "-"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse VMAF score
            for line in result.stderr.split('\n'):
                if "VMAF score:" in line:
                    score = float(line.split(':')[1].strip())
                    normalized_score = score / 100.0
                    self.logger.info(f"VMAF score: {normalized_score:.3f}")
                    return normalized_score
                    
        except subprocess.CalledProcessError:
            self.logger.warning("VMAF measurement failed, falling back to SSIM")
            # Fallback to SSIM if VMAF fails
            cmd = [
                "ffmpeg",
                "-i", str(original_path),
                "-i", str(compressed_path),
                "-filter_complex", "[0:v][1:v]ssim",
                "-f", "null",
                "-"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse SSIM score
            for line in result.stderr.split('\n'):
                if "SSIM All:" in line:
                    score = float(line.split(':')[1].strip().split()[0])
                    self.logger.info(f"SSIM score: {score:.3f}")
                    return score
                    
        self.logger.warning("Quality measurement failed, using default score")
        return 0.8  # Default fallback if quality measurement fails 
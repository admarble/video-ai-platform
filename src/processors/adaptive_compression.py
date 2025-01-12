from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, TypedDict, Set
import logging
import time
from enum import Enum
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta

class CompressionError(Exception):
    """Base exception for compression-related errors"""
    pass

class InvalidMetricError(CompressionError):
    """Raised when an invalid metric is provided"""
    pass

class InvalidVideoInfoError(CompressionError):
    """Raised when invalid video info is provided"""
    pass

class VideoInfo(TypedDict):
    """Type definition for video information"""
    resolution: Tuple[int, int]
    has_motion: bool
    duration: float

class CompressionMetric(Enum):
    """Metrics for evaluating compression performance"""
    QUALITY = "quality"          # Output quality score
    SPEED = "speed"             # Processing speed
    SIZE_REDUCTION = "size"     # Size reduction ratio
    MEMORY_USAGE = "memory"     # Memory usage during processing

class OptimizationStrategy(Enum):
    """Available optimization strategies"""
    SIMPLE = "simple"           # Basic threshold-based adjustments
    GRADIENT = "gradient"       # Gradient-based parameter updates
    ADAPTIVE = "adaptive"       # Adaptive learning with dynamic targets
    WEIGHTED = "weighted"       # Weighted history-based optimization

class VideoCodec(Enum):
    """Supported video codecs"""
    H264 = "libx264"      # H.264/AVC
    H265 = "libx265"      # H.265/HEVC
    VP9 = "libvpx-vp9"    # VP9
    AV1 = "libaom-av1"    # AV1
    MPEG4 = "mpeg4"       # MPEG-4
    VP8 = "libvpx"        # VP8
    THEORA = "libtheora"  # Theora

class AudioCodec(Enum):
    """Supported audio codecs"""
    AAC = "aac"           # AAC
    OPUS = "libopus"      # Opus
    VORBIS = "libvorbis"  # Vorbis
    MP3 = "libmp3lame"    # MP3
    AC3 = "ac3"           # AC3
    FLAC = "flac"         # FLAC

class ContainerFormat(Enum):
    """Supported container formats"""
    MP4 = "mp4"
    MKV = "mkv"
    WEBM = "webm"
    MOV = "mov"
    AVI = "avi"
    OGG = "ogg"

@dataclass
class CodecParameters:
    """Codec-specific parameters and constraints"""
    min_bitrate: str
    max_bitrate: str
    supported_presets: List[str]
    quality_param: str  # CRF for x264/x265, CQ for VP9, etc.
    quality_range: Tuple[int, int]  # Min and max quality values
    supported_pixel_formats: List[str]
    supported_containers: List[ContainerFormat]
    default_preset: str
    default_quality: int
    supports_multipass: bool
    recommended_bitrates: Dict[str, str]  # Resolution -> bitrate mapping

class CodecSupport:
    """Codec support and parameter configurations"""
    
    CODEC_PARAMS = {
        VideoCodec.H264: CodecParameters(
            min_bitrate="500k",
            max_bitrate="50M",
            supported_presets=[
                "veryslow", "slower", "slow", "medium",
                "fast", "faster", "veryfast", "superfast", "ultrafast"
            ],
            quality_param="crf",
            quality_range=(0, 51),
            supported_pixel_formats=["yuv420p", "yuv422p", "yuv444p"],
            supported_containers=[
                ContainerFormat.MP4, ContainerFormat.MKV,
                ContainerFormat.MOV, ContainerFormat.AVI
            ],
            default_preset="medium",
            default_quality=23,
            supports_multipass=True,
            recommended_bitrates={
                "1080p": "5M",
                "2160p": "15M",
                "4320p": "40M"
            }
        ),
        VideoCodec.H265: CodecParameters(
            min_bitrate="400k",
            max_bitrate="40M",
            supported_presets=[
                "veryslow", "slower", "slow", "medium",
                "fast", "faster", "veryfast", "superfast", "ultrafast"
            ],
            quality_param="crf",
            quality_range=(0, 51),
            supported_pixel_formats=["yuv420p", "yuv422p", "yuv444p"],
            supported_containers=[
                ContainerFormat.MP4, ContainerFormat.MKV,
                ContainerFormat.MOV
            ],
            default_preset="medium",
            default_quality=28,
            supports_multipass=True,
            recommended_bitrates={
                "1080p": "3M",
                "2160p": "10M",
                "4320p": "30M"
            }
        ),
        VideoCodec.VP9: CodecParameters(
            min_bitrate="300k",
            max_bitrate="30M",
            supported_presets=[
                "0", "1", "2", "3", "4", "5", "6"
            ],
            quality_param="cq",
            quality_range=(0, 63),
            supported_pixel_formats=["yuv420p", "yuv422p", "yuv444p"],
            supported_containers=[
                ContainerFormat.WEBM, ContainerFormat.MKV
            ],
            default_preset="1",
            default_quality=31,
            supports_multipass=True,
            recommended_bitrates={
                "1080p": "2.5M",
                "2160p": "8M",
                "4320p": "25M"
            }
        ),
        VideoCodec.AV1: CodecParameters(
            min_bitrate="200k",
            max_bitrate="25M",
            supported_presets=[
                "0", "1", "2", "3", "4", "5", "6", "7", "8"
            ],
            quality_param="cq",
            quality_range=(0, 63),
            supported_pixel_formats=["yuv420p", "yuv422p", "yuv444p"],
            supported_containers=[
                ContainerFormat.MP4, ContainerFormat.WEBM,
                ContainerFormat.MKV
            ],
            default_preset="4",
            default_quality=34,
            supports_multipass=True,
            recommended_bitrates={
                "1080p": "2M",
                "2160p": "6M",
                "4320p": "20M"
            }
        )
    }

    @staticmethod
    def get_codec_params(codec: VideoCodec) -> CodecParameters:
        """Get parameters for a specific codec"""
        return CodecSupport.CODEC_PARAMS[codec]

    @staticmethod
    def get_compatible_audio_codecs(
        video_codec: VideoCodec,
        container: ContainerFormat
    ) -> Set[AudioCodec]:
        """Get compatible audio codecs for a video codec and container combination"""
        if container == ContainerFormat.WEBM:
            return {AudioCodec.OPUS, AudioCodec.VORBIS}
        elif container == ContainerFormat.MP4:
            return {AudioCodec.AAC, AudioCodec.AC3, AudioCodec.MP3}
        elif container == ContainerFormat.MKV:
            return {
                AudioCodec.AAC, AudioCodec.AC3, AudioCodec.MP3,
                AudioCodec.OPUS, AudioCodec.VORBIS, AudioCodec.FLAC
            }
        elif container == ContainerFormat.OGG:
            return {AudioCodec.VORBIS, AudioCodec.OPUS}
        else:
            return {AudioCodec.AAC, AudioCodec.MP3}

    @staticmethod
    def get_recommended_settings(
        codec: VideoCodec,
        resolution: Tuple[int, int],
        target_quality: float
    ) -> Dict[str, Any]:
        """Get recommended codec settings based on resolution and quality target"""
        params = CodecSupport.get_codec_params(codec)
        height = resolution[1]
        
        # Determine resolution category
        if height <= 1080:
            res_key = "1080p"
        elif height <= 2160:
            res_key = "2160p"
        else:
            res_key = "4320p"
            
        # Adjust quality parameter based on target quality
        quality_range = params.quality_range[1] - params.quality_range[0]
        quality_value = int(
            params.quality_range[1] -
            (quality_range * target_quality)
        )
        
        # Select preset based on quality target
        preset_idx = int(
            (1 - target_quality) *
            (len(params.supported_presets) - 1)
        )
        preset = params.supported_presets[preset_idx]
        
        return {
            "bitrate": params.recommended_bitrates[res_key],
            "quality_value": quality_value,
            "preset": preset,
            "pixel_format": "yuv420p"  # Most compatible
        }

@dataclass
class OptimizationConfig:
    """Configuration for optimization strategies"""
    strategy: OptimizationStrategy = OptimizationStrategy.GRADIENT
    learning_rate: float = 0.1
    gradient_momentum: float = 0.9
    history_weight_decay: float = 0.95
    min_samples: int = 10
    max_adjustment_per_step: float = 0.2

class CompressionProfile:
    """Represents a compression configuration profile"""
    name: str
    video_codec: VideoCodec
    audio_codec: AudioCodec
    container_format: ContainerFormat
    video_bitrate: str
    audio_bitrate: str
    preset: str
    quality_value: int
    pixel_format: str = "yuv420p"
    performance_score: float = 0.0
    usage_count: int = 0
    multipass: bool = False

class AdaptiveCompressionTuner:
    """Auto-tunes compression parameters based on performance metrics"""
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        history_size: int = 100,
        learning_rate: float = 0.1,
        optimization_config: Optional[OptimizationConfig] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.history_size = history_size
        self.learning_rate = learning_rate
        self.optimization_config = optimization_config or OptimizationConfig()
        
        # Initialize profiles and history
        self.profiles: Dict[str, CompressionProfile] = {}
        self.history: List[Dict[str, Any]] = []
        
        # Initialize optimization state
        self.parameter_gradients: Dict[str, Dict[str, float]] = {}
        self.parameter_momentum: Dict[str, Dict[str, float]] = {}
        
        # Load or create default profiles
        if config_path and config_path.exists():
            self._load_config(config_path)
        else:
            self._create_default_profiles()
            
    def _create_default_profiles(self):
        """Create default compression profiles"""
        profiles = [
            CompressionProfile(
                name="high_quality",
                video_codec=VideoCodec.H265,
                audio_codec=AudioCodec.AAC,
                container_format=ContainerFormat.MP4,
                video_bitrate="6M",
                audio_bitrate="192k",
                preset="slower",
                quality_value=18,
                multipass=True
            ),
            CompressionProfile(
                name="balanced",
                video_codec=VideoCodec.H264,
                audio_codec=AudioCodec.AAC,
                container_format=ContainerFormat.MP4,
                video_bitrate="4M",
                audio_bitrate="128k",
                preset="medium",
                quality_value=23
            ),
            CompressionProfile(
                name="web_optimized",
                video_codec=VideoCodec.VP9,
                audio_codec=AudioCodec.OPUS,
                container_format=ContainerFormat.WEBM,
                video_bitrate="2M",
                audio_bitrate="96k",
                preset="1",
                quality_value=31
            ),
            CompressionProfile(
                name="next_gen",
                video_codec=VideoCodec.AV1,
                audio_codec=AudioCodec.OPUS,
                container_format=ContainerFormat.MP4,
                video_bitrate="1.5M",
                audio_bitrate="128k",
                preset="4",
                quality_value=34
            )
        ]
        
        for profile in profiles:
            self.profiles[profile.name] = profile
            
    def _load_config(self, config_path: Path):
        """Load compression profiles from config file"""
        try:
            with open(config_path) as f:
                data = json.load(f)
                
            self.profiles = {
                name: CompressionProfile(**profile_data)
                for name, profile_data in data['profiles'].items()
            }
            
        except Exception as e:
            self.logger.error(f"Error loading compression config: {str(e)}")
            self._create_default_profiles()
            
    def _validate_metrics(
        self,
        metrics: Dict[CompressionMetric, float]
    ) -> None:
        """Validate compression metrics"""
        for metric, value in metrics.items():
            if not isinstance(metric, CompressionMetric):
                raise InvalidMetricError(f"Invalid metric type: {metric}")
            if not isinstance(value, (int, float)):
                raise InvalidMetricError(
                    f"Metric value must be numeric, got {type(value)}"
                )
            if not 0 <= value <= 1:
                raise InvalidMetricError(
                    f"Metric value must be between 0 and 1, got {value}"
                )

    def _validate_video_info(self, video_info: Dict[str, Any]) -> None:
        """Validate video information"""
        required_fields = {'resolution', 'has_motion', 'duration'}
        missing_fields = required_fields - set(video_info.keys())
        if missing_fields:
            raise InvalidVideoInfoError(
                f"Missing required video info fields: {missing_fields}"
            )

        # Validate resolution
        resolution = video_info['resolution']
        if not (
            isinstance(resolution, (tuple, list)) and
            len(resolution) == 2 and
            all(isinstance(x, int) and x > 0 for x in resolution)
        ):
            raise InvalidVideoInfoError(
                "Resolution must be a tuple of two positive integers"
            )

        # Validate has_motion
        if not isinstance(video_info['has_motion'], bool):
            raise InvalidVideoInfoError("has_motion must be a boolean")

        # Validate duration
        if not (
            isinstance(video_info['duration'], (int, float)) and
            video_info['duration'] > 0
        ):
            raise InvalidVideoInfoError("duration must be a positive number")

    def select_profile(
        self,
        video_info: Dict[str, Any],
        target_metrics: Dict[CompressionMetric, float]
    ) -> CompressionProfile:
        """Select best compression profile based on video and target metrics.
        
        Args:
            video_info: Dictionary containing video information with keys:
                - resolution: Tuple[int, int] - Video dimensions (width, height)
                - has_motion: bool - Whether video has significant motion
                - duration: float - Video duration in seconds
            target_metrics: Dictionary mapping CompressionMetric to target values (0-1)
        
        Returns:
            CompressionProfile: The selected compression profile
            
        Raises:
            InvalidVideoInfoError: If video info is invalid
            InvalidMetricError: If metrics are invalid
        """
        self._validate_video_info(video_info)
        self._validate_metrics(target_metrics)
        
        # Calculate scores for each profile
        scores = {}
        for name, profile in self.profiles.items():
            score = self._calculate_profile_score(
                profile, video_info, target_metrics
            )
            scores[name] = score
            
        # Select best profile
        best_profile_name = max(scores.items(), key=lambda x: x[1])[0]
        selected_profile = self.profiles[best_profile_name]
        
        # Update usage count
        selected_profile.usage_count += 1
        
        return selected_profile
        
    def _calculate_profile_score(
        self,
        profile: CompressionProfile,
        video_info: Dict[str, Any],
        target_metrics: Dict[CompressionMetric, float]
    ) -> float:
        """Calculate score for profile based on requirements"""
        score = profile.performance_score
        
        # Get codec parameters
        codec_params = CodecSupport.get_codec_params(profile.video_codec)
        
        # Adjust for video characteristics
        if video_info.get('resolution', (0, 0))[0] > 1920:
            # For 4K content, favor modern codecs
            if profile.video_codec in [VideoCodec.H265, VideoCodec.VP9, VideoCodec.AV1]:
                score += 0.2
            # Favor higher bitrate
            recommended = CodecSupport.get_recommended_settings(
                profile.video_codec,
                video_info['resolution'],
                target_metrics.get(CompressionMetric.QUALITY, 0.8)
            )
            if int(profile.video_bitrate[:-1]) >= int(recommended['bitrate'][:-1]):
                score += 0.15
                
        if video_info.get('has_motion', False):
            # For high motion, favor certain codecs and presets
            if profile.video_codec in [VideoCodec.H264, VideoCodec.H265]:
                if profile.preset in ['slower', 'slow']:
                    score += 0.15
            elif profile.video_codec == VideoCodec.VP9:
                if profile.preset in ['0', '1']:
                    score += 0.15
                    
        # Adjust for target metrics
        if CompressionMetric.QUALITY in target_metrics:
            quality_range = codec_params.quality_range[1] - codec_params.quality_range[0]
            quality_score = 1.0 - (
                (profile.quality_value - codec_params.quality_range[0]) /
                quality_range
            )
            score += quality_score * target_metrics[CompressionMetric.QUALITY]
            
        if CompressionMetric.SPEED in target_metrics:
            # Normalize preset to 0-1 range for all codecs
            preset_idx = codec_params.supported_presets.index(profile.preset)
            speed_score = preset_idx / (len(codec_params.supported_presets) - 1)
            score += speed_score * target_metrics[CompressionMetric.SPEED]
            
        return score
        
    def update_profile_performance(
        self,
        profile_name: str,
        metrics: Dict[CompressionMetric, float]
    ):
        """Update profile performance based on compression results.
        
        Args:
            profile_name: Name of the profile to update
            metrics: Dictionary mapping CompressionMetric to achieved values (0-1)
            
        Raises:
            InvalidMetricError: If metrics are invalid
            KeyError: If profile_name doesn't exist
        """
        if profile_name not in self.profiles:
            raise KeyError(f"Profile not found: {profile_name}")
            
        self._validate_metrics(metrics)
        
        profile = self.profiles[profile_name]
        
        # Calculate new performance score
        quality_weight = 0.4
        speed_weight = 0.3
        size_weight = 0.3
        
        new_score = (
            metrics.get(CompressionMetric.QUALITY, 0.0) * quality_weight +
            metrics.get(CompressionMetric.SPEED, 0.0) * speed_weight +
            metrics.get(CompressionMetric.SIZE_REDUCTION, 0.0) * size_weight
        )
        
        # Update using exponential moving average
        profile.performance_score = (
            (1 - self.learning_rate) * profile.performance_score +
            self.learning_rate * new_score
        )
        
        # Store in history
        self.history.append({
            'profile': profile_name,
            'metrics': metrics,
            'timestamp': time.time()
        })
        
        # Trim history if needed
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]
            
    def _calculate_parameter_gradients(
        self,
        profile_name: str,
        target_metrics: Dict[CompressionMetric, float]
    ) -> Dict[str, float]:
        """Calculate gradients for profile parameters based on history"""
        profile_history = [
            entry for entry in self.history
            if entry['profile'] == profile_name
        ]
        
        if len(profile_history) < self.optimization_config.min_samples:
            return {}
            
        # Calculate weighted metrics
        weighted_metrics = self._calculate_weighted_metrics(profile_history)
        
        gradients = {}
        
        # CRF gradient (quality-focused)
        quality_diff = target_metrics.get(CompressionMetric.QUALITY, 0.8) - \
                      weighted_metrics.get(CompressionMetric.QUALITY, 0.8)
        gradients['crf'] = -quality_diff * 10  # Scale factor for CRF
        
        # Preset gradient (speed-focused)
        speed_diff = target_metrics.get(CompressionMetric.SPEED, 0.6) - \
                    weighted_metrics.get(CompressionMetric.SPEED, 0.6)
        gradients['preset_index'] = speed_diff * 2  # Scale factor for preset
        
        # Bitrate gradient (quality/size trade-off)
        size_weight = 0.4
        quality_weight = 0.6
        bitrate_gradient = (
            quality_diff * quality_weight -
            (weighted_metrics.get(CompressionMetric.SIZE_REDUCTION, 0.7) -
             target_metrics.get(CompressionMetric.SIZE_REDUCTION, 0.7)) * size_weight
        )
        gradients['bitrate'] = bitrate_gradient * 1.5  # Scale factor for bitrate
        
        return gradients

    def _calculate_weighted_metrics(
        self,
        history: List[Dict[str, Any]]
    ) -> Dict[CompressionMetric, float]:
        """Calculate weighted average of metrics with time decay"""
        if not history:
            return {}
            
        weights = np.array([
            self.optimization_config.history_weight_decay ** i
            for i in range(len(history))
        ])
        weights = weights / weights.sum()
        
        weighted_metrics = {}
        for metric in CompressionMetric:
            values = [entry['metrics'].get(metric, 0.0) for entry in history]
            weighted_metrics[metric] = float(np.average(values, weights=weights))
            
        return weighted_metrics

    def _apply_gradient_update(
        self,
        profile: CompressionProfile,
        gradients: Dict[str, float]
    ) -> None:
        """Apply gradient updates to profile parameters"""
        if not gradients:
            return
            
        codec_params = CodecSupport.get_codec_params(profile.video_codec)
            
        # Update quality value with momentum
        if 'quality' in gradients:
            momentum = self.parameter_momentum.setdefault(profile.name, {}).setdefault('quality', 0.0)
            momentum = self.optimization_config.gradient_momentum * momentum + \
                      self.optimization_config.learning_rate * gradients['quality']
            
            quality_range = codec_params.quality_range[1] - codec_params.quality_range[0]
            max_change = self.optimization_config.max_adjustment_per_step * quality_range
            quality_change = np.clip(momentum, -max_change, max_change)
            profile.quality_value = int(np.clip(
                profile.quality_value + quality_change,
                codec_params.quality_range[0],
                codec_params.quality_range[1]
            ))
            
            self.parameter_momentum[profile.name]['quality'] = momentum
            
        # Update preset
        if 'preset_index' in gradients:
            presets = codec_params.supported_presets
            current_idx = presets.index(profile.preset)
            
            momentum = self.parameter_momentum.setdefault(profile.name, {}).setdefault('preset', 0.0)
            momentum = self.optimization_config.gradient_momentum * momentum + \
                      self.optimization_config.learning_rate * gradients['preset_index']
            
            max_change = self.optimization_config.max_adjustment_per_step * len(presets)
            preset_change = int(np.clip(momentum, -max_change, max_change))
            new_idx = np.clip(current_idx + preset_change, 0, len(presets) - 1)
            profile.preset = presets[int(new_idx)]
            
            self.parameter_momentum[profile.name]['preset'] = momentum
            
        # Update bitrate
        if 'bitrate' in gradients:
            current_bitrate = int(profile.video_bitrate[:-1])
            min_bitrate = int(codec_params.min_bitrate[:-1])
            max_bitrate = int(codec_params.max_bitrate[:-1])
            
            momentum = self.parameter_momentum.setdefault(profile.name, {}).setdefault('bitrate', 0.0)
            momentum = self.optimization_config.gradient_momentum * momentum + \
                      self.optimization_config.learning_rate * gradients['bitrate']
            
            max_change = self.optimization_config.max_adjustment_per_step * current_bitrate
            bitrate_change = np.clip(momentum, -max_change, max_change)
            new_bitrate = int(np.clip(
                current_bitrate + bitrate_change,
                min_bitrate,
                max_bitrate
            ))
            profile.video_bitrate = f"{new_bitrate}M"
            
            self.parameter_momentum[profile.name]['bitrate'] = momentum

    def optimize_profile(
        self,
        profile_name: str,
        target_metrics: Dict[CompressionMetric, float]
    ) -> Optional[CompressionProfile]:
        """Optimize profile parameters using the selected strategy.
        
        Args:
            profile_name: Name of the profile to optimize
            target_metrics: Target metrics to optimize for
            
        Returns:
            Optional[CompressionProfile]: Optimized profile or None if profile not found
        """
        if profile_name not in self.profiles:
            return None
            
        profile = self.profiles[profile_name]
        
        if self.optimization_config.strategy == OptimizationStrategy.SIMPLE:
            return self._optimize_profile_simple(profile, target_metrics)
        elif self.optimization_config.strategy == OptimizationStrategy.GRADIENT:
            return self._optimize_profile_gradient(profile, target_metrics)
        elif self.optimization_config.strategy == OptimizationStrategy.ADAPTIVE:
            return self._optimize_profile_adaptive(profile, target_metrics)
        else:  # WEIGHTED
            return self._optimize_profile_weighted(profile, target_metrics)

    def _optimize_profile_simple(
        self,
        profile: CompressionProfile,
        target_metrics: Dict[CompressionMetric, float]
    ) -> CompressionProfile:
        """Simple threshold-based optimization strategy"""
        # This is the existing optimization logic
        profile_history = [
            entry for entry in self.history
            if entry['profile'] == profile.name
        ]
        
        if not profile_history:
            return profile
            
        quality_scores = [
            entry['metrics'].get(CompressionMetric.QUALITY, 0.0)
            for entry in profile_history
        ]
        speed_scores = [
            entry['metrics'].get(CompressionMetric.SPEED, 0.0)
            for entry in profile_history
        ]
        
        quality_avg = np.mean(quality_scores)
        speed_avg = np.mean(speed_scores)
        
        target_quality = target_metrics.get(CompressionMetric.QUALITY, 0.8)
        if quality_avg < target_quality - 0.1:
            profile.crf = max(profile.crf - 2, 0)
        elif quality_avg > target_quality + 0.1:
            profile.crf = min(profile.crf + 2, 51)
            
        target_speed = target_metrics.get(CompressionMetric.SPEED, 0.5)
        presets = ['veryslow', 'slower', 'slow', 'medium', 'fast', 'veryfast']
        current_idx = presets.index(profile.preset)
        
        if speed_avg < target_speed - 0.2:
            if current_idx < len(presets) - 1:
                profile.preset = presets[current_idx + 1]
        elif speed_avg > target_speed + 0.2:
            if current_idx > 0:
                profile.preset = presets[current_idx - 1]
                
        return profile

    def _optimize_profile_gradient(
        self,
        profile: CompressionProfile,
        target_metrics: Dict[CompressionMetric, float]
    ) -> CompressionProfile:
        """Gradient-based optimization strategy"""
        gradients = self._calculate_parameter_gradients(profile.name, target_metrics)
        self._apply_gradient_update(profile, gradients)
        return profile

    def _optimize_profile_adaptive(
        self,
        profile: CompressionProfile,
        target_metrics: Dict[CompressionMetric, float]
    ) -> CompressionProfile:
        """Adaptive optimization with dynamic targets"""
        profile_history = [
            entry for entry in self.history
            if entry['profile'] == profile.name
        ]
        
        if len(profile_history) < self.optimization_config.min_samples:
            return self._optimize_profile_simple(profile, target_metrics)
            
        # Calculate trend
        recent_metrics = self._calculate_weighted_metrics(profile_history[-5:])
        older_metrics = self._calculate_weighted_metrics(profile_history[:-5])
        
        # Adjust learning rate based on trend
        metric_trends = {}
        for metric in CompressionMetric:
            if metric in recent_metrics and metric in older_metrics:
                trend = abs(recent_metrics[metric] - older_metrics[metric])
                metric_trends[metric] = trend
                
        # If metrics are unstable, reduce learning rate
        max_trend = max(metric_trends.values(), default=0)
        if max_trend > 0.1:
            effective_learning_rate = self.optimization_config.learning_rate * 0.5
        else:
            effective_learning_rate = self.optimization_config.learning_rate
            
        # Apply gradient update with adjusted learning rate
        original_learning_rate = self.optimization_config.learning_rate
        self.optimization_config.learning_rate = effective_learning_rate
        
        gradients = self._calculate_parameter_gradients(profile.name, target_metrics)
        self._apply_gradient_update(profile, gradients)
        
        self.optimization_config.learning_rate = original_learning_rate
        
        return profile

    def _optimize_profile_weighted(
        self,
        profile: CompressionProfile,
        target_metrics: Dict[CompressionMetric, float]
    ) -> CompressionProfile:
        """Weighted history-based optimization strategy"""
        profile_history = [
            entry for entry in self.history
            if entry['profile'] == profile.name
        ]
        
        if len(profile_history) < self.optimization_config.min_samples:
            return self._optimize_profile_simple(profile, target_metrics)
            
        # Calculate time-based weights
        now = datetime.now()
        weights = []
        for entry in profile_history:
            age = now - datetime.fromtimestamp(entry['timestamp'])
            weight = np.exp(-age.total_seconds() / (7 * 24 * 3600))  # 1-week half-life
            weights.append(weight)
            
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Calculate weighted metrics
        weighted_metrics = {}
        for metric in CompressionMetric:
            values = [entry['metrics'].get(metric, 0.0) for entry in profile_history]
            weighted_metrics[metric] = float(np.average(values, weights=weights))
            
        # Create adjusted targets based on weighted history
        adjusted_targets = {}
        for metric, target in target_metrics.items():
            if metric in weighted_metrics:
                # Move target gradually based on historical performance
                adjusted_targets[metric] = (
                    0.7 * target +
                    0.3 * weighted_metrics[metric]
                )
            else:
                adjusted_targets[metric] = target
                
        # Apply gradient update with adjusted targets
        gradients = self._calculate_parameter_gradients(profile.name, adjusted_targets)
        self._apply_gradient_update(profile, gradients)
        
        return profile

    def get_profile_stats(self) -> Dict[str, Any]:
        """Get statistics about profile performance"""
        stats = {}
        
        for name, profile in self.profiles.items():
            profile_history = [
                entry for entry in self.history
                if entry['profile'] == name
            ]
            
            if not profile_history:
                continue
                
            metrics = {
                metric: np.mean([
                    entry['metrics'].get(metric, 0.0)
                    for entry in profile_history
                ])
                for metric in CompressionMetric
            }
            
            stats[name] = {
                'performance_score': profile.performance_score,
                'usage_count': profile.usage_count,
                'average_metrics': metrics
            }
            
        return stats

def create_compression_tuner(
    config_path: Optional[Path] = None
) -> AdaptiveCompressionTuner:
    """Create compression tuner instance"""
    return AdaptiveCompressionTuner(config_path=config_path) 
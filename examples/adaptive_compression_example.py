from pathlib import Path
from src.processors.adaptive_compression import (
    create_compression_tuner,
    CompressionMetric,
    OptimizationStrategy,
    OptimizationConfig,
    VideoCodec,
    AudioCodec,
    ContainerFormat,
    CodecSupport
)

def demonstrate_codec_selection(video_info: dict, target_metrics: dict):
    """Demonstrate codec selection and optimization for different scenarios"""
    print("\nDemonstrating codec selection and optimization:")
    print("-" * 50)

    # Create tuner with adaptive optimization
    config = OptimizationConfig(
        strategy=OptimizationStrategy.ADAPTIVE,
        learning_rate=0.1,
        history_weight_decay=0.9,
        gradient_momentum=0.9,
        min_samples=5,
        max_adjustment_per_step=0.2
    )
    tuner = create_compression_tuner(optimization_config=config)

    # Test different scenarios
    scenarios = [
        {
            "name": "4K HDR Content",
            "info": {
                "resolution": (3840, 2160),
                "has_motion": True,
                "duration": 300
            },
            "metrics": {
                CompressionMetric.QUALITY: 0.9,
                CompressionMetric.SPEED: 0.4,
                CompressionMetric.SIZE_REDUCTION: 0.6
            }
        },
        {
            "name": "Web Streaming",
            "info": {
                "resolution": (1920, 1080),
                "has_motion": True,
                "duration": 600
            },
            "metrics": {
                CompressionMetric.QUALITY: 0.7,
                CompressionMetric.SPEED: 0.8,
                CompressionMetric.SIZE_REDUCTION: 0.8
            }
        },
        {
            "name": "Archive Quality",
            "info": {
                "resolution": (4096, 2160),
                "has_motion": False,
                "duration": 1800
            },
            "metrics": {
                CompressionMetric.QUALITY: 1.0,
                CompressionMetric.SPEED: 0.2,
                CompressionMetric.SIZE_REDUCTION: 0.4
            }
        }
    ]

    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print("-" * 30)

        # Get recommended settings for each codec
        for codec in [VideoCodec.H264, VideoCodec.H265, VideoCodec.VP9, VideoCodec.AV1]:
            recommended = CodecSupport.get_recommended_settings(
                codec,
                scenario['info']['resolution'],
                scenario['metrics'][CompressionMetric.QUALITY]
            )
            print(f"\n{codec.name} Recommended Settings:")
            print(f"Bitrate: {recommended['bitrate']}")
            print(f"Quality Value: {recommended['quality_value']}")
            print(f"Preset: {recommended['preset']}")

            # Get compatible audio codecs for different containers
            for container in CodecSupport.get_codec_params(codec).supported_containers:
                audio_codecs = CodecSupport.get_compatible_audio_codecs(codec, container)
                print(f"Compatible audio codecs for {container.value}: {[ac.name for ac in audio_codecs]}")

        # Select and optimize profile
        profile = tuner.select_profile(scenario['info'], scenario['metrics'])
        print(f"\nSelected Profile: {profile.name}")
        print(f"Video Codec: {profile.video_codec.name}")
        print(f"Audio Codec: {profile.audio_codec.name}")
        print(f"Container: {profile.container_format.value}")
        print(f"Video Bitrate: {profile.video_bitrate}")
        print(f"Quality Value: {profile.quality_value}")
        print(f"Preset: {profile.preset}")
        print(f"Multipass: {profile.multipass}")

        # Simulate compression and optimization
        print("\nSimulating compression and optimization...")
        for i in range(3):
            # Simulate achieved metrics
            achieved_metrics = {
                CompressionMetric.QUALITY: min(0.95, scenario['metrics'][CompressionMetric.QUALITY] + (-0.1 + i * 0.05)),
                CompressionMetric.SPEED: min(0.95, scenario['metrics'][CompressionMetric.SPEED] + (-0.1 + i * 0.05)),
                CompressionMetric.SIZE_REDUCTION: min(0.95, scenario['metrics'][CompressionMetric.SIZE_REDUCTION] + (-0.1 + i * 0.05))
            }
            
            # Update and optimize
            tuner.update_profile_performance(profile.name, achieved_metrics)
            profile = tuner.optimize_profile(profile.name, scenario['metrics'])
            
            print(f"\nIteration {i + 1}:")
            print(f"Quality Value: {profile.quality_value}")
            print(f"Preset: {profile.preset}")
            print(f"Video Bitrate: {profile.video_bitrate}")

def main():
    # Define base video info and target metrics
    video_info = {
        'resolution': (3840, 2160),
        'has_motion': True,
        'duration': 300
    }

    target_metrics = {
        CompressionMetric.QUALITY: 0.8,
        CompressionMetric.SPEED: 0.6,
        CompressionMetric.SIZE_REDUCTION: 0.7
    }

    # Demonstrate codec selection and optimization
    demonstrate_codec_selection(video_info, target_metrics)

    print("\nCodec Comparison:")
    print("\n1. H.264/AVC (libx264):")
    print("   - Excellent compatibility")
    print("   - Good quality/compression ratio")
    print("   - Fast encoding speeds available")
    print("   - Best for 1080p content")
    
    print("\n2. H.265/HEVC (libx265):")
    print("   - Better compression than H.264")
    print("   - Good for 4K content")
    print("   - Slower encoding")
    print("   - Hardware support improving")
    
    print("\n3. VP9 (libvpx-vp9):")
    print("   - Excellent for web delivery")
    print("   - Good quality at lower bitrates")
    print("   - Slower encoding than H.264")
    print("   - Free and open source")
    
    print("\n4. AV1 (libaom-av1):")
    print("   - Best compression efficiency")
    print("   - Excellent quality at low bitrates")
    print("   - Very slow encoding")
    print("   - Future-proof choice")

if __name__ == "__main__":
    main() 
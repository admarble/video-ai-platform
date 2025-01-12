import unittest
from pathlib import Path
import tempfile
import json
from src.processors.adaptive_compression import (
    AdaptiveCompressionTuner,
    CompressionMetric,
    CompressionProfile,
    InvalidMetricError,
    InvalidVideoInfoError
)

class TestAdaptiveCompressionTuner(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.tuner = AdaptiveCompressionTuner()
        self.valid_video_info = {
            'resolution': (1920, 1080),
            'has_motion': True,
            'duration': 120.0
        }
        self.valid_metrics = {
            CompressionMetric.QUALITY: 0.8,
            CompressionMetric.SPEED: 0.6,
            CompressionMetric.SIZE_REDUCTION: 0.7
        }

    def test_create_default_profiles(self):
        """Test that default profiles are created correctly."""
        self.assertEqual(len(self.tuner.profiles), 3)
        self.assertIn('high_quality', self.tuner.profiles)
        self.assertIn('balanced', self.tuner.profiles)
        self.assertIn('fast', self.tuner.profiles)

        # Test high quality profile settings
        high_quality = self.tuner.profiles['high_quality']
        self.assertEqual(high_quality.video_codec, 'libx264')
        self.assertEqual(high_quality.video_bitrate, '6M')
        self.assertEqual(high_quality.preset, 'slower')
        self.assertEqual(high_quality.crf, 18)

    def test_load_config(self):
        """Test loading profiles from config file."""
        config = {
            'profiles': {
                'test_profile': {
                    'name': 'test_profile',
                    'video_codec': 'libx264',
                    'audio_codec': 'aac',
                    'video_bitrate': '4M',
                    'audio_bitrate': '128k',
                    'preset': 'medium',
                    'crf': 23,
                    'compression_level': 5
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            json.dump(config, f)
            f.flush()
            tuner = AdaptiveCompressionTuner(config_path=Path(f.name))
            
        self.assertIn('test_profile', tuner.profiles)
        profile = tuner.profiles['test_profile']
        self.assertEqual(profile.video_bitrate, '4M')
        self.assertEqual(profile.preset, 'medium')

    def test_validate_metrics(self):
        """Test metric validation."""
        # Test valid metrics
        self.tuner._validate_metrics(self.valid_metrics)

        # Test invalid metric type
        invalid_metrics = {'invalid': 0.5}
        with self.assertRaises(InvalidMetricError):
            self.tuner._validate_metrics(invalid_metrics)

        # Test invalid metric value type
        invalid_metrics = {CompressionMetric.QUALITY: 'invalid'}
        with self.assertRaises(InvalidMetricError):
            self.tuner._validate_metrics(invalid_metrics)

        # Test out of range metric value
        invalid_metrics = {CompressionMetric.QUALITY: 1.5}
        with self.assertRaises(InvalidMetricError):
            self.tuner._validate_metrics(invalid_metrics)

    def test_validate_video_info(self):
        """Test video info validation."""
        # Test valid video info
        self.tuner._validate_video_info(self.valid_video_info)

        # Test missing fields
        invalid_info = {'resolution': (1920, 1080)}
        with self.assertRaises(InvalidVideoInfoError):
            self.tuner._validate_video_info(invalid_info)

        # Test invalid resolution
        invalid_info = {
            'resolution': 'invalid',
            'has_motion': True,
            'duration': 120.0
        }
        with self.assertRaises(InvalidVideoInfoError):
            self.tuner._validate_video_info(invalid_info)

        # Test negative duration
        invalid_info = {
            'resolution': (1920, 1080),
            'has_motion': True,
            'duration': -1
        }
        with self.assertRaises(InvalidVideoInfoError):
            self.tuner._validate_video_info(invalid_info)

    def test_select_profile(self):
        """Test profile selection logic."""
        # Test 4K video selection
        video_info = {
            'resolution': (3840, 2160),
            'has_motion': True,
            'duration': 120.0
        }
        profile = self.tuner.select_profile(video_info, self.valid_metrics)
        self.assertEqual(profile.name, 'high_quality')

        # Test HD video selection
        video_info['resolution'] = (1920, 1080)
        profile = self.tuner.select_profile(video_info, self.valid_metrics)
        self.assertEqual(profile.name, 'balanced')

        # Test speed priority
        speed_metrics = {
            CompressionMetric.QUALITY: 0.3,
            CompressionMetric.SPEED: 0.9,
            CompressionMetric.SIZE_REDUCTION: 0.5
        }
        profile = self.tuner.select_profile(video_info, speed_metrics)
        self.assertEqual(profile.name, 'fast')

    def test_update_profile_performance(self):
        """Test profile performance updates."""
        profile_name = 'balanced'
        initial_score = self.tuner.profiles[profile_name].performance_score

        # Update with good metrics
        metrics = {
            CompressionMetric.QUALITY: 0.9,
            CompressionMetric.SPEED: 0.8,
            CompressionMetric.SIZE_REDUCTION: 0.7
        }
        self.tuner.update_profile_performance(profile_name, metrics)

        # Check that score improved
        new_score = self.tuner.profiles[profile_name].performance_score
        self.assertGreater(new_score, initial_score)

        # Check history update
        self.assertEqual(len(self.tuner.history), 1)
        self.assertEqual(self.tuner.history[0]['profile'], profile_name)

        # Test history size limit
        for _ in range(self.tuner.history_size + 10):
            self.tuner.update_profile_performance(profile_name, metrics)
        self.assertEqual(len(self.tuner.history), self.tuner.history_size)

    def test_optimize_profile(self):
        """Test profile optimization logic."""
        profile_name = 'balanced'
        initial_crf = self.tuner.profiles[profile_name].crf
        initial_preset = self.tuner.profiles[profile_name].preset

        # Update with metrics indicating need for better quality
        metrics = {
            CompressionMetric.QUALITY: 0.5,  # Lower than target
            CompressionMetric.SPEED: 0.8,
            CompressionMetric.SIZE_REDUCTION: 0.7
        }
        self.tuner.update_profile_performance(profile_name, metrics)

        # Optimize profile
        optimized = self.tuner.optimize_profile(
            profile_name,
            {CompressionMetric.QUALITY: 0.8}  # Target higher quality
        )

        # Check that CRF was lowered for better quality
        self.assertLess(optimized.crf, initial_crf)

        # Test preset adjustment for speed
        metrics = {
            CompressionMetric.QUALITY: 0.8,
            CompressionMetric.SPEED: 0.3,  # Much slower than target
            CompressionMetric.SIZE_REDUCTION: 0.7
        }
        self.tuner.update_profile_performance(profile_name, metrics)

        optimized = self.tuner.optimize_profile(
            profile_name,
            {CompressionMetric.SPEED: 0.7}  # Target higher speed
        )

        # Check that preset was changed for better speed
        preset_speeds = ['veryslow', 'slower', 'slow', 'medium', 'fast', 'veryfast']
        initial_preset_idx = preset_speeds.index(initial_preset)
        new_preset_idx = preset_speeds.index(optimized.preset)
        self.assertGreater(new_preset_idx, initial_preset_idx)

    def test_get_profile_stats(self):
        """Test profile statistics generation."""
        # Update some profiles with metrics
        metrics = {
            CompressionMetric.QUALITY: 0.8,
            CompressionMetric.SPEED: 0.7,
            CompressionMetric.SIZE_REDUCTION: 0.6
        }
        self.tuner.update_profile_performance('balanced', metrics)
        self.tuner.update_profile_performance('fast', metrics)

        stats = self.tuner.get_profile_stats()

        # Check stats structure
        self.assertIn('balanced', stats)
        self.assertIn('fast', stats)
        
        # Check stats content
        balanced_stats = stats['balanced']
        self.assertIn('performance_score', balanced_stats)
        self.assertIn('usage_count', balanced_stats)
        self.assertIn('average_metrics', balanced_stats)
        
        # Check metric averages
        avg_metrics = balanced_stats['average_metrics']
        self.assertEqual(avg_metrics[CompressionMetric.QUALITY], 0.8)
        self.assertEqual(avg_metrics[CompressionMetric.SPEED], 0.7)
        self.assertEqual(avg_metrics[CompressionMetric.SIZE_REDUCTION], 0.6)

if __name__ == '__main__':
    unittest.main() 
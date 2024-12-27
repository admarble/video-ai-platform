import unittest
import subprocess
from unittest.mock import patch, MagicMock, call
import tempfile
import json
import os
from pathlib import Path
import pytest
from src.processors.ffmpeg_processor import (
    FFmpegProcessor,
    FFmpegError,
    FFmpegNotFoundError,
    VideoMetadata
)
from src.processors.adaptive_compression import (
    CompressionProfile,
    VideoCodec,
    AudioCodec,
    ContainerFormat,
    CompressionMetric
)

class TestFFmpegProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.mock_ffmpeg_path = "/usr/bin/ffmpeg"
        self.test_input = Path("test_input.mp4")
        self.test_output = Path("test_output.mp4")
        
        # Create a test profile
        self.test_profile = CompressionProfile(
            name="test_profile",
            video_codec=VideoCodec.H264,
            audio_codec=AudioCodec.AAC,
            container_format=ContainerFormat.MP4,
            video_bitrate="2M",
            audio_bitrate="128k",
            preset="medium",
            quality_value=23,
            multipass=False
        )
        
        # Sample FFprobe output
        self.sample_probe_data = {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/1"
                },
                {
                    "codec_type": "audio"
                }
            ],
            "format": {
                "duration": "60.0",
                "bit_rate": "5000000",
                "format_name": "mp4",
                "size": "37500000"
            }
        }

    @patch('subprocess.run')
    def test_find_ffmpeg(self, mock_run):
        """Test FFmpeg executable detection"""
        # Test successful detection
        mock_run.return_value.stdout = self.mock_ffmpeg_path
        processor = FFmpegProcessor()
        self.assertEqual(processor.ffmpeg_path, self.mock_ffmpeg_path)
        
        # Test FFmpeg not found
        mock_run.side_effect = subprocess.CalledProcessError(1, [])
        with self.assertRaises(FFmpegNotFoundError):
            FFmpegProcessor()

    @patch('subprocess.run')
    def test_get_video_info(self, mock_run):
        """Test video metadata extraction"""
        # Mock FFprobe output
        mock_run.return_value.stdout = json.dumps(self.sample_probe_data)
        
        processor = FFmpegProcessor(self.mock_ffmpeg_path)
        metadata = processor.get_video_info(self.test_input)
        
        self.assertEqual(metadata.width, 1920)
        self.assertEqual(metadata.height, 1080)
        self.assertEqual(metadata.duration, 60.0)
        self.assertEqual(metadata.fps, 30.0)
        self.assertTrue(metadata.has_audio)
        
        # Test error handling
        mock_run.side_effect = subprocess.CalledProcessError(1, [])
        with self.assertRaises(FFmpegError):
            processor.get_video_info(self.test_input)

    def test_build_ffmpeg_command(self):
        """Test FFmpeg command generation for different codecs"""
        processor = FFmpegProcessor(self.mock_ffmpeg_path)
        
        # Test H.264 command
        h264_profile = self.test_profile
        cmd = processor._build_ffmpeg_command(
            self.test_input,
            self.test_output,
            h264_profile
        )
        
        expected_cmd = [
            self.mock_ffmpeg_path,
            "-y",
            "-i", str(self.test_input),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-f", "mp4",
            str(self.test_output)
        ]
        
        self.assertEqual(cmd, expected_cmd)
        
        # Test VP9 command
        vp9_profile = CompressionProfile(
            name="vp9_test",
            video_codec=VideoCodec.VP9,
            audio_codec=AudioCodec.OPUS,
            container_format=ContainerFormat.WEBM,
            video_bitrate="2M",
            audio_bitrate="128k",
            preset="1",
            quality_value=31,
            multipass=True
        )
        
        cmd = processor._build_ffmpeg_command(
            self.test_input,
            self.test_output,
            vp9_profile,
            pass_number=1
        )
        
        self.assertIn("-c:v", cmd)
        self.assertIn("libvpx-vp9", cmd)
        self.assertIn("-deadline", cmd)
        self.assertIn("-pass", cmd)
        self.assertIn("1", cmd)

    @patch('subprocess.Popen')
    def test_run_ffmpeg(self, mock_popen):
        """Test FFmpeg process execution and progress monitoring"""
        # Mock process with progress output
        mock_process = MagicMock()
        mock_process.stderr.readline.side_effect = [
            "frame=  100 fps=25 q=28.0 size=    384kB time=00:00:10.00",
            "frame=  200 fps=25 q=28.0 size=    768kB time=00:00:20.00",
            ""
        ]
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        processor = FFmpegProcessor(self.mock_ffmpeg_path)
        progress_calls = []
        
        def progress_callback(time):
            progress_calls.append(time)
            
        processor._run_ffmpeg(["dummy", "command"], progress_callback)
        
        self.assertEqual(len(progress_calls), 2)
        mock_popen.assert_called_once()
        
        # Test error handling
        mock_process.returncode = 1
        with self.assertRaises(subprocess.CalledProcessError):
            processor._run_ffmpeg(["dummy", "command"])

    @patch('subprocess.run')
    def test_calculate_quality_score(self, mock_run):
        """Test video quality measurement"""
        processor = FFmpegProcessor(self.mock_ffmpeg_path)
        
        # Test VMAF measurement
        mock_run.return_value.stderr = "VMAF score: 95.43"
        score = processor._calculate_quality_score(
            self.test_input,
            self.test_output
        )
        self.assertAlmostEqual(score, 0.9543)
        
        # Test SSIM fallback
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, []),  # VMAF fails
            MagicMock(stderr="SSIM All:0.95")     # SSIM succeeds
        ]
        score = processor._calculate_quality_score(
            self.test_input,
            self.test_output
        )
        self.assertAlmostEqual(score, 0.95)

@pytest.mark.integration
class TestFFmpegIntegration:
    """Integration tests for FFmpeg processor with actual video files"""
    
    @pytest.fixture
    def sample_video(self):
        """Create a test video file"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            # Generate a test video using FFmpeg
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", "testsrc=duration=5:size=1280x720:rate=30",
                "-c:v", "libx264",
                "-crf", "23",
                f.name
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                yield Path(f.name)
            finally:
                os.unlink(f.name)
                
    def test_video_compression(self, sample_video):
        """Test actual video compression"""
        processor = FFmpegProcessor()
        output_path = Path(tempfile.mktemp(suffix='.mp4'))
        
        try:
            # Test with H.264 profile
            profile = CompressionProfile(
                name="test_compression",
                video_codec=VideoCodec.H264,
                audio_codec=AudioCodec.AAC,
                container_format=ContainerFormat.MP4,
                video_bitrate="1M",
                audio_bitrate="128k",
                preset="veryfast",
                quality_value=28,
                multipass=False
            )
            
            metrics = processor.compress_video(
                sample_video,
                output_path,
                profile
            )
            
            # Verify compression results
            assert output_path.exists()
            assert metrics[CompressionMetric.QUALITY] > 0.7
            assert metrics[CompressionMetric.SIZE_REDUCTION] > 0
            
            # Verify output video metadata
            output_info = processor.get_video_info(output_path)
            assert output_info.width == 1280
            assert output_info.height == 720
            assert output_info.duration > 4.9  # Allow small duration difference
            
        finally:
            if output_path.exists():
                os.unlink(output_path)
                
    def test_multipass_encoding(self, sample_video):
        """Test multipass encoding"""
        processor = FFmpegProcessor()
        output_path = Path(tempfile.mktemp(suffix='.webm'))
        
        try:
            # Test with VP9 multipass profile
            profile = CompressionProfile(
                name="test_multipass",
                video_codec=VideoCodec.VP9,
                audio_codec=AudioCodec.OPUS,
                container_format=ContainerFormat.WEBM,
                video_bitrate="1M",
                audio_bitrate="128k",
                preset="1",
                quality_value=31,
                multipass=True
            )
            
            metrics = processor.compress_video(
                sample_video,
                output_path,
                profile
            )
            
            # Verify compression results
            assert output_path.exists()
            assert metrics[CompressionMetric.QUALITY] > 0.7
            
            # Verify output format
            output_info = processor.get_video_info(output_path)
            assert "webm" in output_info.format.lower()
            
        finally:
            if output_path.exists():
                os.unlink(output_path)
                
    def test_quality_measurement(self, sample_video):
        """Test quality measurement with actual videos"""
        processor = FFmpegProcessor()
        output_path = Path(tempfile.mktemp(suffix='.mp4'))
        
        try:
            # Compress with different quality settings
            for quality in [18, 28, 38]:  # High, Medium, Low quality
                profile = CompressionProfile(
                    name=f"quality_test_{quality}",
                    video_codec=VideoCodec.H264,
                    audio_codec=AudioCodec.AAC,
                    container_format=ContainerFormat.MP4,
                    video_bitrate="2M",
                    audio_bitrate="128k",
                    preset="medium",
                    quality_value=quality,
                    multipass=False
                )
                
                processor.compress_video(
                    sample_video,
                    output_path,
                    profile
                )
                
                # Measure quality
                score = processor._calculate_quality_score(
                    sample_video,
                    output_path
                )
                
                # Quality should decrease as CRF increases
                expected_min_quality = 1.0 - (quality / 51.0) * 0.5
                assert score > expected_min_quality
                
        finally:
            if output_path.exists():
                os.unlink(output_path)

if __name__ == '__main__':
    unittest.main() 
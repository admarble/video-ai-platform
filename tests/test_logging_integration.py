"""Integration tests for the logging system with actual components."""

import pytest
import tempfile
import shutil
from pathlib import Path
import json
import time
from datetime import datetime

from src.core.logging import LoggingManager, LogLevel
from src.core.video_processor import VideoProcessor, VideoMetadata
from src.core.service_manager import ServiceManager
from src.api.endpoints.video import router as video_router
from fastapi.testclient import TestClient
from fastapi import FastAPI

class TestLoggingIntegration:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
        
    @pytest.fixture
    def video_processor(self, temp_dir):
        """Create a video processor instance with logging."""
        upload_dir = temp_dir / "uploads"
        processing_dir = temp_dir / "processing"
        return VideoProcessor(
            upload_dir=str(upload_dir),
            processing_dir=str(processing_dir)
        )
        
    @pytest.fixture
    def service_manager(self, temp_dir):
        """Create a service manager instance with logging."""
        return ServiceManager(cache_dir=str(temp_dir / "cache"))
        
    @pytest.fixture
    def test_app(self, temp_dir):
        """Create a FastAPI test app with logging."""
        app = FastAPI()
        app.include_router(video_router)
        return TestClient(app)
        
    def test_video_processor_logging(self, video_processor, temp_dir):
        """Test logging during video processing operations."""
        # Create test video metadata
        metadata = VideoMetadata(
            video_id="test_123",
            filename="test.mp4",
            file_path=str(temp_dir / "test.mp4"),
            file_size=1024,
            duration=10.0,
            width=1920,
            height=1080,
            fps=30.0,
            created_at=datetime.now(),
            format="mp4"
        )
        
        # Create dummy video file
        with open(metadata.file_path, 'wb') as f:
            f.write(b'dummy video content')
            
        # Process video and check logs
        try:
            video_processor.process_video(metadata)
        except Exception:
            pass  # Expected to fail with dummy video
            
        # Check log file contents
        log_path = temp_dir / "logs" / "video_processing.log"
        log_content = log_path.read_text()
        
        assert "Starting video processing" in log_content
        assert metadata.video_id in log_content
        assert "ERROR" in log_content  # Should contain error for dummy video
        
    def test_service_manager_logging(self, service_manager, temp_dir):
        """Test logging during service manager operations."""
        # Initialize services and check logs
        service_manager.initialize_services(['scene_processor', 'object_detector'])
        
        # Check log file contents
        log_path = temp_dir / "logs" / "system.log"
        log_content = log_path.read_text()
        
        assert "Initialized service" in log_content
        assert "scene_processor" in log_content
        assert "object_detector" in log_content
        
        # Test cleanup logging
        service_manager.cleanup()
        log_content = log_path.read_text()
        assert "cleanup" in log_content.lower()
        
    def test_api_endpoint_logging(self, test_app, temp_dir):
        """Test logging in API endpoints."""
        # Make test API calls
        response = test_app.post(
            "/upload",
            files={"file": ("test.mp4", b"dummy content", "video/mp4")}
        )
        
        # Check access log
        log_path = temp_dir / "logs" / "access.log"
        log_content = log_path.read_text()
        
        assert "POST /upload" in log_content
        assert "test.mp4" in log_content
        
        if response.status_code != 200:
            assert "ERROR" in log_content
        else:
            assert "SUCCESS" in log_content
            
    def test_error_propagation(self, video_processor, temp_dir):
        """Test error logging propagation through components."""
        # Create invalid metadata to trigger error
        metadata = VideoMetadata(
            video_id="error_test",
            filename="nonexistent.mp4",
            file_path="/nonexistent/path",
            file_size=0,
            duration=0,
            width=0,
            height=0,
            fps=0,
            created_at=datetime.now(),
            format="mp4"
        )
        
        # Attempt processing and check error logs
        with pytest.raises(Exception):
            video_processor.process_video(metadata)
            
        # Check logs in all components
        video_log = (temp_dir / "logs" / "video_processing.log").read_text()
        system_log = (temp_dir / "logs" / "system.log").read_text()
        
        assert "ERROR" in video_log
        assert "error_test" in video_log
        assert "Failed to process video" in video_log
        assert "nonexistent.mp4" in system_log
        
    def test_concurrent_logging(self, video_processor, temp_dir):
        """Test logging behavior with concurrent operations."""
        import threading
        
        def process_dummy_video(video_id: str):
            metadata = VideoMetadata(
                video_id=video_id,
                filename=f"{video_id}.mp4",
                file_path=str(temp_dir / f"{video_id}.mp4"),
                file_size=1024,
                duration=10.0,
                width=1920,
                height=1080,
                fps=30.0,
                created_at=datetime.now(),
                format="mp4"
            )
            
            # Create dummy file
            with open(metadata.file_path, 'wb') as f:
                f.write(b'dummy content')
                
            try:
                video_processor.process_video(metadata)
            except Exception:
                pass
                
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(
                target=process_dummy_video,
                args=(f"concurrent_test_{i}",)
            )
            threads.append(thread)
            thread.start()
            
        # Wait for all threads
        for thread in threads:
            thread.join()
            
        # Check logs
        log_content = (temp_dir / "logs" / "video_processing.log").read_text()
        
        # Verify all video IDs are in logs
        for i in range(3):
            assert f"concurrent_test_{i}" in log_content
            
    def test_log_rotation_during_processing(self, video_processor, temp_dir):
        """Test log rotation during continuous processing."""
        # Create a large number of log entries to trigger rotation
        metadata = VideoMetadata(
            video_id="rotation_test",
            filename="test.mp4",
            file_path=str(temp_dir / "test.mp4"),
            file_size=1024,
            duration=10.0,
            width=1920,
            height=1080,
            fps=30.0,
            created_at=datetime.now(),
            format="mp4"
        )
        
        # Create dummy file
        with open(metadata.file_path, 'wb') as f:
            f.write(b'dummy content')
            
        # Process multiple times to generate logs
        for _ in range(10):
            try:
                video_processor.process_video(metadata)
            except Exception:
                pass
            time.sleep(0.1)  # Small delay to ensure different timestamps
            
        # Check for rotated log files
        log_files = list((temp_dir / "logs").glob("video_processing.log*"))
        assert len(log_files) > 1  # Should have main log and at least one backup
        
    def test_logging_with_custom_config(self, temp_dir):
        """Test logging with custom configuration during processing."""
        # Create custom logging config
        config = {
            "video_processing": {
                "level": "DEBUG",
                "format": "CUSTOM: %(asctime)s - %(levelname)s - %(message)s - %(video_id)s",
                "max_bytes": 1024,
                "backup_count": 2
            }
        }
        
        # Initialize processor with custom config
        upload_dir = temp_dir / "uploads"
        processing_dir = temp_dir / "processing"
        processor = VideoProcessor(
            upload_dir=str(upload_dir),
            processing_dir=str(processing_dir)
        )
        processor.logging_manager = LoggingManager(base_dir=temp_dir, config=config)
        
        # Create test metadata
        metadata = VideoMetadata(
            video_id="custom_config_test",
            filename="test.mp4",
            file_path=str(temp_dir / "test.mp4"),
            file_size=1024,
            duration=10.0,
            width=1920,
            height=1080,
            fps=30.0,
            created_at=datetime.now(),
            format="mp4"
        )
        
        # Create dummy file
        with open(metadata.file_path, 'wb') as f:
            f.write(b'dummy content')
            
        # Process and check logs
        try:
            processor.process_video(metadata)
        except Exception:
            pass
            
        log_content = (temp_dir / "logs" / "video_processing.log").read_text()
        assert "CUSTOM:" in log_content
        assert "custom_config_test" in log_content 
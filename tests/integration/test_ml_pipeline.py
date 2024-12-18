import unittest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import os

from src.exceptions import VideoProcessingError, AudioProcessingError, ServiceInitializationError
from src.models.scene_processor import SceneProcessor, SceneSegment
from src.models.object_detector import ObjectDetector, DetectedObject
from src.models.audio_processor import AudioProcessor, AudioSegment
from src.models.text_video_aligner import TextVideoAligner, SearchResult

class TestFrameExtraction(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.video_path = os.path.join(self.temp_dir, "test_video.mp4")
        
    def tearDown(self):
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            
    def test_extract_frames_invalid_path(self):
        with self.assertRaises(VideoProcessingError):
            _extract_frames("nonexistent_video.mp4")
            
    def test_extract_frames_valid_params(self):
        with patch('decord.VideoReader') as mock_reader:
            mock_reader.return_value.get_batch.return_value = np.zeros((10, 100, 100, 3))
            mock_reader.return_value.get_avg_fps.return_value = 30
            
            frames, fps = _extract_frames(
                self.video_path,
                sampling_rate=2,
                max_frames=5
            )
            
            self.assertEqual(fps, 30)
            self.assertEqual(frames.shape[0], 5)
            self.assertEqual(frames.shape[3], 3)

class TestSceneProcessor(unittest.TestCase):
    def setUp(self):
        self.scene_processor = SceneProcessor()
        self.test_frames = torch.rand(10, 3, 224, 224)  # Mock video frames
        
    def test_scene_detection(self):
        scenes = self.scene_processor.detect_scenes(self.test_frames)
        self.assertIsInstance(scenes[0], SceneSegment)
        self.assertTrue(len(scenes) > 0)
        
    def test_empty_input(self):
        with self.assertRaises(ValueError):
            self.scene_processor.detect_scenes(torch.empty(0))

class TestObjectDetector(unittest.TestCase):
    def setUp(self):
        self.object_detector = ObjectDetector()
        self.test_frame = torch.rand(3, 224, 224)
        
    def test_object_detection(self):
        detections = self.object_detector.detect_objects(self.test_frame)
        self.assertIsInstance(detections[0], DetectedObject)
        self.assertTrue(all(0 <= obj.confidence <= 1 for obj in detections))
        
    def test_batch_detection(self):
        batch_frames = torch.rand(5, 3, 224, 224)
        batch_detections = self.object_detector.detect_objects_batch(batch_frames)
        self.assertEqual(len(batch_detections), 5)

class TestAudioProcessor(unittest.TestCase):
    def setUp(self):
        self.audio_processor = AudioProcessor()
        self.temp_dir = tempfile.mkdtemp()
        self.audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        
    def tearDown(self):
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            
    def test_audio_segmentation(self):
        with patch('librosa.load') as mock_load:
            mock_load.return_value = (np.random.rand(16000), 16000)
            segments = self.audio_processor.segment_audio(self.audio_path)
            self.assertIsInstance(segments[0], AudioSegment)
            
    def test_invalid_audio_file(self):
        with self.assertRaises(AudioProcessingError):
            self.audio_processor.segment_audio("nonexistent_audio.wav")

class TestTextVideoAligner(unittest.TestCase):
    def setUp(self):
        self.aligner = TextVideoAligner()
        self.test_text = "Test query text"
        self.test_video_features = torch.rand(10, 512)  # Mock video features
        
    def test_text_video_alignment(self):
        results = self.aligner.find_matches(
            query_text=self.test_text,
            video_features=self.test_video_features
        )
        self.assertIsInstance(results[0], SearchResult)
        self.assertTrue(all(0 <= result.similarity_score <= 1 for result in results))
        
    def test_empty_video_features(self):
        with self.assertRaises(ValueError):
            self.aligner.find_matches(
                query_text=self.test_text,
                video_features=torch.empty(0)
            )

# ... rest of the test classes as provided ...
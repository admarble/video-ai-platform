from dataclasses import dataclass
from typing import Optional, List, Tuple
import asyncio
import logging
import torch
import numpy as np
import decord
import gc
import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
import librosa
from pathlib import Path
import subprocess

from src.core.model_loader import ModelManager
from src.services.ml import SceneAnalyzer, ObjectDetector
from src.models.domain.video import ProcessingResult

@dataclass
class ProcessingConfig:
    gpu_enabled: bool = True
    batch_size: int = 32
    max_memory_gb: float = 4.0
    num_workers: int = 4
    cache_dir: Optional[str] = None
    logging_level: str = "INFO"

@dataclass
class AudioSegment:
    """Represents a processed audio segment with transcription"""
    start_time: float
    end_time: float
    text: str
    confidence: float

class AudioProcessingError(Exception):
    """Custom exception for audio processing errors"""
    pass

class VideoProcessor:
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.logger = logging.getLogger(__name__)
        self.model_manager = ModelManager()
        self.scene_analyzer = SceneAnalyzer(self.model_manager)
        self.object_detector = ObjectDetector(self.model_manager)
        self._setup_logging()

    def __enter__(self):
        self._initialize_services()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_resources()

    def _initialize_services(self):
        """Initialize processing services and GPU if available"""
        if self.config.gpu_enabled and torch.cuda.is_available():
            self.logger.info("GPU acceleration enabled")
            self.device = torch.device("cuda")
        else:
            self.logger.info("Running on CPU")
            self.device = torch.device("cpu")

    def _cleanup_resources(self):
        """Clean up GPU memory and other resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.logger.info("Resources cleaned up")

    def _setup_logging(self):
        """Configure logging based on config"""
        logging.basicConfig(
            level=self.config.logging_level,
            format='%(asctime)s %(levelname)s [%(name)s] %(message)s'
        )

    async def process_video(self, video_path: str) -> ProcessingResult:
        """Process video through ML pipeline with improved memory management"""
        self.logger.info(f"Starting video processing: {video_path}")
        
        try:
            # Initialize video reader with GPU support if available
            ctx = decord.gpu(0) if (self.config.gpu_enabled and torch.cuda.is_available()) else decord.cpu(0)
            video_reader = decord.VideoReader(video_path, ctx=ctx)
            total_frames = len(video_reader)
            
            self.logger.info(f"Processing {total_frames} frames")
            
            # Process in optimized batches
            batch_results = []
            for batch_frames, frame_indices in self._get_frame_batches(video_reader, total_frames):
                self.logger.debug(f"Processing batch of {len(frame_indices)} frames")
                
                # Process batch
                scene_results = await self.scene_analyzer.analyze(batch_frames)
                object_results = await self._process_batch_with_memory_limit(
                    batch_frames,
                    max_memory_gb=self.config.max_memory_gb
                )
                
                batch_results.append((scene_results, object_results))
                
            result = self._combine_results(batch_results)
            self.logger.info("Video processing completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}", exc_info=True)
            raise

    def _get_frame_batches(self, video_reader: decord.VideoReader, total_frames: int) -> Tuple[np.ndarray, List[int]]:
        """Generator for memory-efficient frame batch extraction
        
        Args:
            video_reader: Decord video reader instance
            total_frames: Total number of frames to process
            
        Yields:
            Tuple containing:
                - np.ndarray: Batch of frames
                - List[int]: Frame indices for the batch
        """
        for batch_idx in range(0, total_frames, self.config.batch_size):
            frame_indices = list(range(
                batch_idx,
                min(batch_idx + self.config.batch_size, total_frames)
            ))
            batch_frames = video_reader.get_batch(frame_indices).asnumpy()
            yield batch_frames, frame_indices

    async def _process_batch_with_memory_limit(
        self,
        frames: np.ndarray,
        max_memory_gb: float = 4.0
    ) -> List[List[dict]]:
        """Process frames with memory limit"""
        # Calculate memory usage per frame (approximate)
        frame_memory = frames[0].nbytes * frames.shape[0]
        available_memory = max_memory_gb * 1024 * 1024 * 1024  # Convert to bytes
        
        if frame_memory > available_memory:
            # Adjust batch size to fit memory constraints
            ratio = available_memory / frame_memory
            adjusted_batch_size = max(1, int(self.config.batch_size * ratio))
            self.logger.warning(f"Adjusting batch size to {adjusted_batch_size} due to memory constraints")
            
            results = []
            for i in range(0, len(frames), adjusted_batch_size):
                batch = frames[i:i + adjusted_batch_size]
                result = self.object_detector.process_frames(
                    batch,
                    batch_size=adjusted_batch_size,
                    enable_tracking=True
                )
                results.extend(result)
            return results
        
        return self.object_detector.process_frames(
            frames,
            batch_size=self.config.batch_size,
            enable_tracking=True
        )

    def _combine_results(self, batch_results: List[Tuple[List, List]]) -> ProcessingResult:
        """Combine batch results into final output"""
        scenes = []
        objects = []
        
        for scene_batch, object_batch in batch_results:
            scenes.extend(scene_batch)
            objects.extend(object_batch)
        
        return ProcessingResult(
            scenes=scenes,
            objects=objects,
            metadata=self._extract_metadata()
        )

    def _extract_metadata(self) -> dict:
        """Extract processing metadata"""
        return {
            "gpu_enabled": self.config.gpu_enabled and torch.cuda.is_available(),
            "batch_size": self.config.batch_size,
            "device": str(self.device)
        }

class AudioProcessor:
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        device: Optional[str] = None,
        sample_rate: int = 16000
    ):
        """Initialize audio processor with Wav2Vec2 model."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = sample_rate
        
        # Initialize Wav2Vec2 model and processor
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.to(self.device)
        
        logging.info(f"Initialized AudioProcessor with {model_name} on {self.device}")

    def _extract_audio(self, video_path: str) -> Tuple[np.ndarray, int]:
        """Extract audio from video file using FFmpeg."""
        try:
            if not Path(video_path).exists():
                raise AudioProcessingError(f"Video file not found: {video_path}")
            
            temp_audio = Path(video_path).with_suffix('.tmp.wav')
            
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', str(self.sample_rate),
                '-ac', '1', '-y', str(temp_audio)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            audio_array, sr = librosa.load(
                temp_audio,
                sr=self.sample_rate,
                mono=True
            )
            
            temp_audio.unlink()
            return audio_array, sr
            
        except subprocess.CalledProcessError as e:
            raise AudioProcessingError(f"FFmpeg error: {e.stderr.decode()}")
        except Exception as e:
            raise AudioProcessingError(f"Error extracting audio: {str(e)}")

    def _segment_audio(
        self,
        audio: np.ndarray,
        min_silence_len: int = 500,
        silence_thresh: float = -40
    ) -> list[Tuple[int, int]]:
        """Segment audio based on silence detection."""
        min_silence_samples = int(min_silence_len * self.sample_rate / 1000)
        db = librosa.amplitude_to_db(np.abs(audio), ref=np.max)
        silent = db < silence_thresh
        
        boundaries = []
        is_silent = True
        current_start = 0
        
        for i, s in enumerate(silent):
            if is_silent and not s:
                current_start = i
                is_silent = False
            elif not is_silent and s:
                if i - current_start >= min_silence_samples:
                    boundaries.append((current_start, i))
                is_silent = True
        
        if not is_silent:
            boundaries.append((current_start, len(audio)))
            
        return boundaries

    def _transcribe_segment(
        self,
        audio_segment: np.ndarray
    ) -> Tuple[str, float]:
        """Transcribe an audio segment using Wav2Vec2."""
        inputs = self.processor(
            audio_segment,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        predicted_ids = torch.argmax(logits, dim=-1)
        confidence = torch.mean(torch.max(logits.softmax(dim=-1), dim=-1)[0])
        
        transcription = self.processor.decode(predicted_ids[0])
        
        return transcription.text, confidence.item()

    def process_audio(
        self,
        video_path: str,
        segment_audio: bool = True
    ) -> list[AudioSegment]:
        """Process audio from video file for transcription."""
        logging.info(f"Processing audio from {video_path}")
        
        audio_array, sr = self._extract_audio(video_path)
        
        if segment_audio:
            segments = self._segment_audio(audio_array)
            results = []
            
            for start_idx, end_idx in segments:
                start_time = start_idx / sr
                end_time = end_idx / sr
                segment = audio_array[start_idx:end_idx]
                text, confidence = self._transcribe_segment(segment)
                
                if text.strip():
                    results.append(
                        AudioSegment(
                            start_time=start_time,
                            end_time=end_time,
                            text=text,
                            confidence=confidence
                        )
                    )
        else:
            text, confidence = self._transcribe_segment(audio_array)
            results = [
                AudioSegment(
                    start_time=0,
                    end_time=len(audio_array) / sr,
                    text=text,
                    confidence=confidence
                )
            ]
            
        logging.info(f"Completed audio processing: {len(results)} segments transcribed")
        return results
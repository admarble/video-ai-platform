import asyncio
import cv2
import numpy as np
import torch
from typing import List, Dict, Any
from src.core.model_loader import ModelManager
from src.models.domain.video import (
    ProcessingResult, 
    VideoMetadata, 
    ProcessingStatus,
    SceneInfo,
    ObjectDetection
)
from src.core.config import settings
from datetime import datetime
import uuid
import logging
from PIL import Image

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        self.model_manager = ModelManager()
        self._initialize_services()
        self.device = torch.device("cuda" if torch.cuda.is_available() and settings.USE_CUDA else "cpu")

    def _initialize_services(self):
        """Initialize all required processing services"""
        self.scene_model = self.model_manager.get_model("scene")
        self.object_model = self.model_manager.get_model("object")
        self.audio_model = self.model_manager.get_model("audio")
        self.clip_model = self.model_manager.get_model("clip")
        
        # Get processors
        self.scene_processor = self.model_manager.get_processor("scene")
        self.object_processor = self.model_manager.get_processor("object")
        self.audio_processor = self.model_manager.get_processor("audio")
        self.clip_processor = self.model_manager.get_processor("clip")

    async def init_video(self, video_upload) -> VideoMetadata:
        """Initialize video metadata"""
        cap = cv2.VideoCapture(str(video_upload.url))
        
        metadata = VideoMetadata(
            id=str(uuid.uuid4()),
            url=video_upload.url,
            status=ProcessingStatus.PENDING,
            duration=float(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            resolution=(
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        cap.release()
        return metadata

    async def process_video(self, video_path: str) -> ProcessingResult:
        """Process video through all models"""
        try:
            frames = await self._extract_frames(video_path)
            
            # Process in parallel
            scene_task = self._process_scenes(frames)
            object_task = self._process_objects(frames)
            audio_task = self._process_audio(video_path)
            
            scenes, objects, audio = await asyncio.gather(
                scene_task,
                object_task,
                audio_task
            )
            
            # Generate CLIP embeddings
            embeddings = await self._generate_embeddings(frames)
            
            return ProcessingResult(
                video_id=str(uuid.uuid4()),
                scenes=scenes,
                objects=objects,
                transcript=audio,
                embeddings=embeddings
            )
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise

    async def _extract_frames(self, video_path: str, sample_rate: int = 1) -> List[np.ndarray]:
        """
        Extract frames from video with sampling
        Args:
            video_path: Path to video file
            sample_rate: Extract every nth frame
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_rate == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            frame_count += 1
            
        cap.release()
        return frames

    async def _process_scenes(self, frames: List[np.ndarray]) -> List[SceneInfo]:
        """Process scenes using VideoMAE"""
        scenes = []
        batch_size = settings.BATCH_SIZE
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            
            # Prepare inputs
            inputs = self.scene_processor(
                [Image.fromarray(frame) for frame in batch_frames],
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.scene_model(**inputs)
                
            # Process outputs
            for j, logits in enumerate(outputs.logits):
                timestamp = (i + j) / len(frames)
                top_pred = torch.topk(logits, k=1)
                
                scene_info = SceneInfo(
                    start_time=timestamp,
                    end_time=timestamp + 1/len(frames),
                    description=self.scene_processor.config.id2label[top_pred.indices[0].item()],
                    confidence=float(top_pred.values[0]),
                    objects=[]  # Will be populated by object detection
                )
                scenes.append(scene_info)
        
        return scenes

    async def _process_objects(self, frames: List[np.ndarray]) -> List[ObjectDetection]:
        """Detect objects using DETR"""
        objects = []
        batch_size = settings.BATCH_SIZE
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            
            # Prepare inputs
            inputs = self.object_processor(
                [Image.fromarray(frame) for frame in batch_frames],
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.object_model(**inputs)
            
            # Process detections
            for j, (logits, boxes) in enumerate(zip(outputs.logits, outputs.pred_boxes)):
                timestamp = (i + j) / len(frames)
                
                # Get predictions above threshold
                scores = torch.sigmoid(logits)
                predictions = torch.where(scores > 0.5)
                
                for pred_idx in range(len(predictions[0])):
                    label_id = predictions[1][pred_idx].item()
                    score = scores[predictions[0][pred_idx], label_id].item()
                    box = boxes[predictions[0][pred_idx]].tolist()
                    
                    detection = ObjectDetection(
                        label=self.object_processor.config.id2label[label_id],
                        confidence=score,
                        bbox=box,
                        timestamp=timestamp
                    )
                    objects.append(detection)
        
        return objects

    async def _process_audio(self, video_path: str) -> Dict[float, str]:
        """Process audio using Wav2Vec2"""
        import moviepy.editor as mp
        import soundfile as sf
        import io
        
        # Extract audio from video
        video = mp.VideoFileClip(video_path)
        audio = video.audio
        
        # Save audio to buffer
        audio_buffer = io.BytesIO()
        audio.write_audiofile(audio_buffer, codec='pcm_s16le')
        audio_buffer.seek(0)
        
        # Load audio data
        audio_data, sample_rate = sf.read(audio_buffer)
        
        # Process in chunks
        chunk_duration = 30  # seconds
        chunk_size = chunk_duration * sample_rate
        transcription = {}
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            
            # Prepare inputs
            inputs = self.audio_processor(
                chunk,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.audio_model(**inputs)
            
            # Decode predictions
            predicted_ids = torch.argmax(outputs.logits, dim=-1)
            transcription[i/sample_rate] = self.audio_processor.decode(predicted_ids[0])
        
        return transcription

    async def _generate_embeddings(self, frames: List[np.ndarray]) -> Dict[str, List[float]]:
        """Generate CLIP embeddings for frames"""
        embeddings = []
        batch_size = settings.BATCH_SIZE
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            
            # Prepare inputs
            inputs = self.clip_processor(
                [Image.fromarray(frame) for frame in batch_frames],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.clip_model.get_image_features(**inputs)
                
            # Normalize embeddings
            embeddings.extend(
                torch.nn.functional.normalize(outputs, dim=-1).cpu().numpy().tolist()
            )
        
        return {
            "frame_embeddings": embeddings,
            "frame_count": len(frames)
        }

    async def process_batch(self, video_paths: List[str]) -> List[ProcessingResult]:
        """Process multiple videos in parallel"""
        tasks = [self.process_video(path) for path in video_paths]
        return await asyncio.gather(*tasks)

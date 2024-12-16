import asyncio
import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple
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
from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        self.model_manager = ModelManager()
        self._initialize_services()
        self.device = torch.device("cuda" if torch.cuda.is_available() and settings.USE_CUDA else "cpu")
        self.batch_size = settings.BATCH_SIZE

    def _initialize_services(self):
        """Initialize all required processing services"""
        self.scene_model = self.model_manager.get_model("scene")
        self.scene_processor = self.model_manager.get_processor("scene")
        
        self.object_model = self.model_manager.get_model("object")
        self.object_processor = self.model_manager.get_processor("object")
        
        self.audio_model = self.model_manager.get_model("audio")
        self.audio_processor = self.model_manager.get_processor("audio")
        
        self.clip_model = self.model_manager.get_model("clip")
        self.clip_processor = self.model_manager.get_processor("clip")

    async def init_video(self, video_upload) -> VideoMetadata:
        """Initialize video metadata"""
        cap = cv2.VideoCapture(str(video_upload.url))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps else None
        cap.release()

        return VideoMetadata(
            id=str(uuid.uuid4()),
            url=video_upload.url,
            status=ProcessingStatus.PENDING,
            duration=duration,
            frame_count=frame_count,
            fps=fps,
            resolution=(width, height),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

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
        """Extract frames from video with sampling"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_rate == 0:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                
            frame_count += 1
            
        cap.release()
        return frames

    async def _process_scenes(self, frames: List[np.ndarray]) -> List[SceneInfo]:
        """Process scenes using VideoMAE"""
        scenes = []
        
        try:
            # Process frames in batches
            for i in range(0, len(frames), self.batch_size):
                batch_frames = frames[i:i + self.batch_size]
                
                # Prepare inputs
                inputs = self.scene_processor(
                    batch_frames,
                    return_tensors="pt",
                    sampling_rate=8  # Sample every 8th frame
                ).to(self.device)
                
                # Run inference with automatic mixed precision
                with autocast(enabled=settings.FP16):
                    with torch.no_grad():
                        outputs = self.scene_model(**inputs)
                
                # Process outputs
                for j, logits in enumerate(outputs.logits):
                    scene_probs = torch.nn.functional.softmax(logits, dim=-1)
                    top_scenes = torch.topk(scene_probs, k=1)
                    
                    scene_info = SceneInfo(
                        start_time=float(i + j) / len(frames),
                        end_time=float(i + j + 1) / len(frames),
                        description=str(top_scenes.indices[0].item()),
                        confidence=float(top_scenes.values[0].item()),
                        objects=[]  # Will be filled by object detection
                    )
                    scenes.append(scene_info)
                    
        except Exception as e:
            logger.error(f"Error in scene processing: {str(e)}")
            raise
            
        return scenes

    async def _process_objects(self, frames: List[np.ndarray]) -> List[ObjectDetection]:
        """Detect objects using DETR"""
        objects = []
        
        try:
            # Process frames in batches
            for i in range(0, len(frames), self.batch_size):
                batch_frames = frames[i:i + self.batch_size]
                
                # Prepare inputs
                inputs = self.object_processor(
                    batch_frames,
                    return_tensors="pt"
                ).to(self.device)
                
                # Run inference
                with autocast(enabled=settings.FP16):
                    with torch.no_grad():
                        outputs = self.object_model(**inputs)
                
                # Process detections
                for j, (img_boxes, img_labels, img_scores) in enumerate(zip(
                    outputs.pred_boxes, outputs.pred_labels, outputs.scores
                )):
                    # Filter by confidence
                    mask = img_scores > 0.5
                    boxes = img_boxes[mask].cpu().numpy()
                    labels = img_labels[mask].cpu().numpy()
                    scores = img_scores[mask].cpu().numpy()
                    
                    timestamp = float(i + j) / len(frames)
                    
                    for box, label, score in zip(boxes, labels, scores):
                        obj_detection = ObjectDetection(
                            label=str(label),
                            confidence=float(score),
                            bbox=box.tolist(),
                            timestamp=timestamp
                        )
                        objects.append(obj_detection)
                        
        except Exception as e:
            logger.error(f"Error in object detection: {str(e)}")
            raise
            
        return objects

    async def _process_audio(self, video_path: str) -> Dict[float, str]:
        """Process audio using Wav2Vec2"""
        try:
            # Extract audio from video
            import moviepy.editor as mp
            video = mp.VideoFileClip(video_path)
            audio = video.audio
            
            # Process in chunks
            chunk_duration = 30  # seconds
            transcript = {}
            
            for i in range(0, int(video.duration), chunk_duration):
                # Extract audio chunk
                chunk = audio.subclip(i, min(i + chunk_duration, video.duration))
                
                # Convert to array
                array = chunk.to_soundarray()
                inputs = self.audio_processor(
                    array,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).to(self.device)
                
                # Run inference
                with autocast(enabled=settings.FP16):
                    with torch.no_grad():
                        outputs = self.audio_model(**inputs)
                
                # Process transcription
                tokens = outputs.logits.argmax(dim=-1)
                transcription = self.audio_processor.batch_decode(tokens)[0]
                
                # Add to timeline
                transcript[float(i)] = transcription
                
            video.close()
            return transcript
            
        except Exception as e:
            logger.error(f"Error in audio processing: {str(e)}")
            raise

    async def _generate_embeddings(self, frames: List[np.ndarray]) -> Dict[str, List[float]]:
        """Generate CLIP embeddings for frames"""
        try:
            embeddings = {}
            
            # Process frames in batches
            for i in range(0, len(frames), self.batch_size):
                batch_frames = frames[i:i + self.batch_size]
                
                # Prepare inputs
                inputs = self.clip_processor(
                    images=batch_frames,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                # Generate embeddings
                with autocast(enabled=settings.FP16):
                    with torch.no_grad():
                        outputs = self.clip_model.get_image_features(**inputs)
                
                # Store embeddings
                for j, embedding in enumerate(outputs):
                    timestamp = float(i + j) / len(frames)
                    embeddings[str(timestamp)] = embedding.cpu().numpy().tolist()
                    
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def _batch_frames(self, frames: List[np.ndarray]) -> List[List[np.ndarray]]:
        """Utility function to batch frames"""
        return [
            frames[i:i + self.batch_size] 
            for i in range(0, len(frames), self.batch_size)
        ]
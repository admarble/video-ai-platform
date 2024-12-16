import asyncio
import cv2
import numpy as np
import torch
from typing import List, Dict, Any
from src.core.model_loader import ModelManager
from src.models.domain.video import ProcessingResult, VideoMetadata, ProcessingStatus
from src.models.domain.video import SceneInfo, ObjectDetection
from datetime import datetime
import uuid
import logging
from decord import VideoReader
from decord import cpu, gpu
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        self.model_manager = ModelManager()
        self._initialize_services()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 32

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
        video = cv2.VideoCapture(str(video_upload.url))
        
        metadata = VideoMetadata(
            id=str(uuid.uuid4()),
            url=video_upload.url,
            status=ProcessingStatus.PENDING,
            duration=video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS),
            frame_count=int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=video.get(cv2.CAP_PROP_FPS),
            resolution=(
                int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        video.release()
        return metadata

    async def process_video(self, video_path: str) -> ProcessingResult:
        """Process video through all models"""
        try:
            # Extract frames efficiently using decord
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

    async def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from video using decord"""
        try:
            # Use GPU if available
            ctx = gpu if torch.cuda.is_available() else cpu
            vr = VideoReader(video_path, ctx=ctx)
            
            # Calculate frame indices for extraction
            total_frames = len(vr)
            # Extract 1 frame per second
            frame_indices = list(range(0, total_frames, int(vr.get_avg_fps())))
            
            # Extract frames in batches
            frames = []
            for i in range(0, len(frame_indices), self.batch_size):
                batch_indices = frame_indices[i:i + self.batch_size]
                batch_frames = vr.get_batch(batch_indices).asnumpy()
                frames.extend(batch_frames)
            
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            raise

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
                    padding=True
                ).to(self.device)
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.scene_model(**inputs)
                    logits = outputs.logits
                    
                    # Get scene predictions
                    probs = F.softmax(logits, dim=-1)
                    predictions = torch.argmax(probs, dim=-1)
                    
                    # Create scene info objects
                    for j, pred in enumerate(predictions):
                        frame_idx = i + j
                        scenes.append(SceneInfo(
                            start_time=frame_idx / self.batch_size,
                            end_time=(frame_idx + 1) / self.batch_size,
                            description=str(pred.item()),  # Convert to actual scene description
                            confidence=float(probs[j, pred].item()),
                            objects=[]  # Will be filled by object detection
                        ))
            
            return scenes
            
        except Exception as e:
            logger.error(f"Error processing scenes: {str(e)}")
            raise

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
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.object_model(**inputs)
                    
                    # Process each frame's predictions
                    for j, frame_outputs in enumerate(outputs.pred_boxes):
                        frame_idx = i + j
                        timestamp = frame_idx / self.batch_size
                        
                        # Get boxes and labels
                        boxes = frame_outputs.detach().cpu().numpy()
                        scores = outputs.pred_logits[j].softmax(-1)[:, :-1].max(-1)
                        labels = outputs.pred_logits[j].softmax(-1)[:, :-1].argmax(-1)
                        
                        # Filter high-confidence detections
                        for box, score, label in zip(boxes, scores, labels):
                            if score > 0.7:  # Confidence threshold
                                objects.append(ObjectDetection(
                                    label=str(label.item()),  # Convert to actual label
                                    confidence=float(score.item()),
                                    bbox=box.tolist(),
                                    timestamp=timestamp
                                ))
            
            return objects
            
        except Exception as e:
            logger.error(f"Error detecting objects: {str(e)}")
            raise

    async def _process_audio(self, video_path: str) -> Dict[float, str]:
        """Process audio using Wav2Vec2"""
        try:
            # Extract audio from video
            audio = await self._extract_audio(video_path)
            
            # Process audio in chunks
            transcript = {}
            chunk_size = 30 * 16000  # 30 seconds at 16kHz
            
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                
                # Prepare inputs
                inputs = self.audio_processor(
                    chunk,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.audio_model(**inputs)
                    chunks = self.audio_processor.decode(outputs.logits.argmax(dim=-1)[0])
                    
                    # Add to transcript with timestamp
                    timestamp = i / 16000  # Convert samples to seconds
                    transcript[timestamp] = chunks
            
            return transcript
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            raise

    async def _extract_audio(self, video_path: str) -> np.ndarray:
        """Extract audio from video file"""
        try:
            import moviepy.editor as mp
            video = mp.VideoFileClip(video_path)
            audio = video.audio
            samples = audio.to_soundarray(fps=16000)
            return samples.mean(axis=1) if len(samples.shape) > 1 else samples
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            raise

    async def _generate_embeddings(self, frames: List[np.ndarray]) -> Dict[str, List[float]]:
        """Generate CLIP embeddings for frames"""
        try:
            embeddings = []
            
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
                with torch.no_grad():
                    outputs = self.clip_model.get_image_features(**inputs)
                    batch_embeddings = outputs.cpu().numpy()
                    embeddings.extend(batch_embeddings)
            
            return {
                "frame_embeddings": [emb.tolist() for emb in embeddings]
            }
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
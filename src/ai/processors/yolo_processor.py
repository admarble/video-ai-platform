from typing import Dict, List, Optional, Any, Tuple
import torch
import numpy as np
from ultralytics import YOLO
import logging
from dataclasses import dataclass

@dataclass
class YOLOSettings:
    model_size: str = 'x'  # n, s, m, l, x
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_det: int = 300
    batch_size: int = 16

class YOLOProcessor:
    def __init__(
        self,
        settings: YOLOSettings,
        device: Optional[str] = None
    ):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            self.model = YOLO(f'yolov8{settings.model_size}.pt')
            self.model.to(self.device)
        except Exception as e:
            self.logger.error(f"Failed to initialize YOLO model: {str(e)}")
            raise

    def detect_objects(
        self,
        frames: np.ndarray,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Run object detection with tracking on video frames."""
        conf = conf_threshold or self.settings.conf_threshold
        iou = iou_threshold or self.settings.iou_threshold
        
        try:
            # Run detection with tracking
            results = self.model.track(
                source=frames,
                conf=conf,
                iou=iou,
                persist=True,
                verbose=False,
                max_det=self.settings.max_det,
                device=self.device
            )
            
            return self._process_detections(results)
            
        except Exception as e:
            self.logger.error(f"Error during object detection: {str(e)}")
            return self.handle_detection_error(e, {'frames': frames})

    def _process_detections(
        self,
        results: List[Any]
    ) -> List[Dict[str, Any]]:
        """Process YOLO detection results into structured format."""
        detections = []
        
        for idx, result in enumerate(results):
            frame_detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    detection = {
                        'track_id': int(box.id.item()) if box.id is not None else None,
                        'class_id': int(box.cls.item()),
                        'label': result.names[int(box.cls.item())],
                        'confidence': float(box.conf.item()),
                        'bbox': box.xyxy.cpu().numpy()[0].tolist(),
                        'frame_idx': idx
                    }
                    frame_detections.append(detection)
            detections.append(frame_detections)
            
        return detections

    def analyze_spatial_relationships(
        self,
        detections: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze spatial relationships between detected objects."""
        relationships = {}
        
        for frame_dets in detections:
            frame_rels = []
            for i, det1 in enumerate(frame_dets):
                for det2 in frame_dets[i+1:]:
                    rel = self._compute_spatial_relationship(det1, det2)
                    if rel:
                        frame_rels.append(rel)
            
            if frame_rels:
                relationships[frame_dets[0]['frame_idx']] = frame_rels
                
        return relationships

    def _compute_spatial_relationship(
        self,
        det1: Dict[str, Any],
        det2: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Compute spatial relationship between two detections."""
        bbox1 = det1['bbox']
        bbox2 = det2['bbox']
        
        # Calculate centers
        center1 = [(bbox1[0] + bbox1[2])/2, (bbox1[1] + bbox1[3])/2]
        center2 = [(bbox2[0] + bbox2[2])/2, (bbox2[1] + bbox2[3])/2]
        
        # Calculate distance
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # Calculate relative position
        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]
        
        # Determine relationship type
        if distance < min(
            (bbox1[2] - bbox1[0] + bbox2[2] - bbox2[0])/4,
            (bbox1[3] - bbox1[1] + bbox2[3] - bbox2[1])/4
        ):
            rel_type = 'near'
        else:
            rel_type = 'far'
        
        return {
            'object1': {
                'track_id': det1['track_id'],
                'label': det1['label']
            },
            'object2': {
                'track_id': det2['track_id'],
                'label': det2['label']
            },
            'relationship': rel_type,
            'distance': float(distance),
            'relative_position': {
                'dx': float(dx),
                'dy': float(dy)
            }
        }

    def handle_detection_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Handle detection errors and return partial results if possible."""
        self.logger.error(f"Detection error: {str(error)}")
        
        if isinstance(error, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            
            # Try processing with smaller batch size
            if context.get('frames') is not None:
                try:
                    # Reduce batch size and max detections
                    self.settings.batch_size //= 2
                    self.settings.max_det //= 2
                    return self.detect_objects(context['frames'])
                except Exception as e:
                    self.logger.error(f"Recovery attempt failed: {str(e)}")
        
        return [] 
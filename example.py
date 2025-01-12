import torch
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification
import numpy as np
import cv2
from typing import List

def load_model(model_path: str = None):
    """Load the VideoMAE model."""
    if model_path:
        model = VideoMAEForVideoClassification.from_pretrained(model_path)
    else:
        model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics')
    
    feature_extractor = VideoMAEFeatureExtractor.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics')
    return model, feature_extractor

def process_video(video_path: str, model: VideoMAEForVideoClassification, feature_extractor: VideoMAEFeatureExtractor):
    """Process a video file and return predictions."""
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Extract frames
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    # Prepare inputs
    inputs = feature_extractor(frames, return_tensors="pt")
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get predicted class
    predicted_class_idx = logits.argmax(-1).item()
    
    return {
        'class_idx': predicted_class_idx,
        'class_name': model.config.id2label[predicted_class_idx],
        'confidence': torch.softmax(logits, dim=1)[0][predicted_class_idx].item()
    }

def main():
    # Load model and feature extractor
    print("Loading model...")
    model, feature_extractor = load_model()
    model.eval()
    
    print("Model loaded successfully!")
    print(f"Number of classes: {model.config.num_labels}")
    
    # Example usage
    # results = process_video('path_to_your_video.mp4', model, feature_extractor)
    # print(f"Predicted class: {results['class_name']}")
    # print(f"Confidence: {results['confidence']:.2%}")

if __name__ == '__main__':
    main() 
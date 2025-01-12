# VideoMAE Implementation

This project implements video understanding using the VideoMAE (Video Masked Autoencoder) model through Hugging Face's transformers library.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main script `example.py` provides functionality to:
- Load the VideoMAE model
- Process video files
- Get action recognition predictions

Example usage:
```python
from example import load_model, process_video

# Load model
model, feature_extractor = load_model()

# Process video
results = process_video('path_to_video.mp4', model, feature_extractor)
print(f"Predicted action: {results['class_name']}")
print(f"Confidence: {results['confidence']:.2%}")
```

## Model Details

This implementation uses the VideoMAE model fine-tuned on the Kinetics-400 dataset, which can recognize 400 different human actions in videos.

## License

MIT License 
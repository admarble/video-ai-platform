import numpy as np
import pytest
from src.services.video.text_video_aligner import TextVideoAligner, SearchResult

@pytest.fixture
def aligner():
    return TextVideoAligner(device="cpu", batch_size=2)

@pytest.fixture
def sample_frames():
    # Create dummy frames (3 frames of 224x224 RGB)
    return np.random.rand(3, 224, 224, 3).astype(np.float32)

def test_search_frames(aligner, sample_frames):
    # Generate embeddings
    frame_embeddings = aligner._generate_embeddings(sample_frames)
    
    # Test search
    results = aligner.search_frames(
        query="a test image",
        frame_embeddings=frame_embeddings,
        fps=30.0,
        top_k=2
    )
    
    assert isinstance(results, list)
    assert len(results) <= 2
    assert all(isinstance(r, SearchResult) for r in results)

def test_batch_search_frames(aligner, sample_frames):
    frame_embeddings = aligner._generate_embeddings(sample_frames)
    
    queries = ["first query", "second query"]
    results = aligner.batch_search_frames(
        queries=queries,
        frame_embeddings=frame_embeddings,
        fps=30.0
    )
    
    assert isinstance(results, dict)
    assert set(results.keys()) == set(queries) 
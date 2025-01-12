import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from tqdm import tqdm

@dataclass
class SearchResult:
    """Represents a frame matching a text query"""
    frame_idx: int
    timestamp: float
    similarity_score: float

class TextVideoAligner:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Initialize CLIP model for text-video alignment.
        
        Args:
            model_name: CLIP model to use for embeddings
            device: Device to run model on ('cuda' or 'cpu')
            batch_size: Batch size for processing frames
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        # Initialize CLIP model and processor
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.to(self.device)
        
        logging.info(f"Initialized TextVideoAligner with {model_name} on {self.device}")
        
    def _generate_embeddings(
        self,
        frames: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate CLIP embeddings for video frames.
        
        Args:
            frames: numpy array of frames (N, H, W, C)
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            numpy array of embeddings (N, embedding_dim)
        """
        embeddings = []
        
        # Process frames in batches
        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i:min(i + self.batch_size, len(frames))]
            
            # Prepare inputs
            inputs = self.processor(
                images=list(batch_frames),
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                
            # Normalize if requested
            if normalize:
                outputs = outputs / outputs.norm(dim=-1, keepdim=True)
                
            embeddings.append(outputs.cpu().numpy())
            
        # Concatenate all batches
        all_embeddings = np.concatenate(embeddings, axis=0)
        
        logging.info(f"Generated embeddings with shape {all_embeddings.shape}")
        return all_embeddings
        
    def _generate_text_embedding(
        self,
        text: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate CLIP embedding for text query.
        
        Args:
            text: Text query
            normalize: Whether to L2-normalize embedding
            
        Returns:
            numpy array of text embedding
        """
        # Prepare inputs
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            
        # Normalize if requested
        if normalize:
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)
            
        return outputs.cpu().numpy()
        
    def search_frames(
        self,
        query: str,
        frame_embeddings: np.ndarray,
        fps: float,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for frames matching a text query.
        
        Args:
            query: Text query
            frame_embeddings: Pre-computed frame embeddings
            fps: Frames per second of the video
            top_k: Number of results to return
            threshold: Minimum similarity score threshold
            
        Returns:
            List of SearchResult objects sorted by similarity
        """
        # Generate query embedding
        query_embedding = self._generate_text_embedding(query)
        
        # Calculate similarities
        similarities = frame_embeddings @ query_embedding.T
        similarities = similarities.squeeze()
        
        # Filter by threshold
        valid_indices = similarities >= threshold
        valid_similarities = similarities[valid_indices]
        valid_frame_indices = np.where(valid_indices)[0]
        
        # Sort by similarity
        sorted_indices = np.argsort(-valid_similarities)[:top_k]
        
        # Create results
        results = []
        for idx in sorted_indices:
            frame_idx = valid_frame_indices[idx]
            result = SearchResult(
                frame_idx=frame_idx,
                timestamp=frame_idx / fps,
                similarity_score=float(valid_similarities[idx])
            )
            results.append(result)
            
        return results
        
    def batch_search_frames(
        self,
        queries: List[str],
        frame_embeddings: np.ndarray,
        fps: float,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> Dict[str, List[SearchResult]]:
        """
        Search for frames matching multiple text queries.
        
        Args:
            queries: List of text queries
            frame_embeddings: Pre-computed frame embeddings
            fps: Frames per second of the video
            top_k: Number of results per query
            threshold: Minimum similarity score threshold
            
        Returns:
            Dictionary mapping queries to their search results
        """
        results = {}
        
        # Process queries in batches
        for query in tqdm(queries, desc="Processing queries"):
            query_results = self.search_frames(
                query,
                frame_embeddings,
                fps,
                top_k,
                threshold
            )
            results[query] = query_results
            
        return results
        
    def find_temporal_matches(
        self,
        query: str,
        frame_embeddings: np.ndarray,
        fps: float,
        window_size: int = 5,
        stride: int = 1,
        threshold: float = 0.0
    ) -> List[Tuple[int, int, float]]:
        """
        Find temporal segments matching a text query.
        
        Args:
            query: Text query
            frame_embeddings: Pre-computed frame embeddings
            fps: Frames per second of the video
            window_size: Size of temporal window in frames
            stride: Stride for sliding window
            threshold: Minimum similarity score threshold
            
        Returns:
            List of (start_frame, end_frame, score) tuples
        """
        query_embedding = self._generate_text_embedding(query)
        matches = []
        
        # Slide window over frames
        for start_idx in range(0, len(frame_embeddings) - window_size + 1, stride):
            end_idx = start_idx + window_size
            
            # Get window embeddings and calculate mean
            window_embedding = frame_embeddings[start_idx:end_idx].mean(axis=0)
            window_embedding = window_embedding / np.linalg.norm(window_embedding)
            
            # Calculate similarity
            similarity = float(window_embedding @ query_embedding.T)
            
            if similarity >= threshold:
                matches.append((start_idx, end_idx, similarity))
                
        # Merge overlapping matches
        merged = []
        if matches:
            current_start, current_end, current_score = matches[0]
            
            for start, end, score in matches[1:]:
                if start <= current_end:
                    # Merge overlapping segments
                    current_end = max(current_end, end)
                    current_score = max(current_score, score)
                else:
                    # Add current segment and start new one
                    merged.append((current_start, current_end, current_score))
                    current_start, current_end, current_score = start, end, score
                    
            merged.append((current_start, current_end, current_score))
            
        return merged 
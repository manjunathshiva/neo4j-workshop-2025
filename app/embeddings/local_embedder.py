"""
Local embedding generation using sentence-transformers.
Optimized for GitHub Codespace environment with memory-efficient processing.
"""

import logging
import os
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

try:
    from ..config import get_config
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import get_config

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    embeddings: np.ndarray
    texts: List[str]
    processing_time: float
    model_info: Dict[str, Any]
    batch_info: Dict[str, Any]

@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 32
    max_length: int = 512
    overlap_tokens: int = 50
    normalize_embeddings: bool = True
    show_progress: bool = True

class LocalEmbedder:
    """
    Local embedding generation using sentence-transformers.
    Optimized for Codespace environment with memory-efficient batch processing.
    """
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize local embedder with specified model.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to use ('cpu', 'cuda', 'auto')
        """
        self.config = get_config()
        self.model_name = model_name or self.config.llm.local_embedding_model
        self.device = self._determine_device(device)
        
        self._model: Optional[SentenceTransformer] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model_info: Dict[str, Any] = {}
        self._is_loaded = False
        
        # Codespace optimization settings
        self.batch_config = BatchConfig(
            batch_size=16,  # Smaller batches for Codespace memory limits
            max_length=512,
            overlap_tokens=50,
            normalize_embeddings=True,
            show_progress=True
        )
        
        logger.info(f"LocalEmbedder initialized with model: {self.model_name}, device: {self.device}")
    
    def _determine_device(self, device: Optional[str]) -> str:
        """Determine the best device for embedding generation."""
        if device == "auto" or device is None:
            if torch.cuda.is_available():
                # Check if we have enough GPU memory (Codespace may have limited GPU)
                try:
                    torch.cuda.empty_cache()
                    device = "cuda"
                    logger.info("Using CUDA device for embeddings")
                except:
                    device = "cpu"
                    logger.info("CUDA available but using CPU for stability in Codespace")
            else:
                device = "cpu"
                logger.info("Using CPU device for embeddings")
        
        return device
    
    def load_model(self, force_reload: bool = False) -> bool:
        """
        Load the sentence-transformers model with Codespace optimizations.
        
        Args:
            force_reload: Force reload even if model is already loaded
            
        Returns:
            True if model loaded successfully
        """
        if self._is_loaded and not force_reload:
            logger.info(f"Model {self.model_name} already loaded")
            return True
        
        try:
            logger.info(f"Loading sentence-transformers model: {self.model_name}")
            start_time = time.time()
            
            # Load model with Codespace-friendly settings
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=os.path.expanduser("~/.cache/sentence_transformers")
            )
            
            # Load tokenizer for text analysis
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Get model information
            self._model_info = {
                "model_name": self.model_name,
                "device": str(self._model.device),
                "max_seq_length": self._model.max_seq_length,
                "embedding_dimension": self._model.get_sentence_embedding_dimension(),
                "load_time": time.time() - start_time,
                "model_size_mb": self._estimate_model_size(),
                "tokenizer_vocab_size": len(self._tokenizer.vocab) if self._tokenizer else 0
            }
            
            self._is_loaded = True
            
            logger.info(f"Model loaded successfully in {self._model_info['load_time']:.2f}s")
            logger.info(f"Embedding dimension: {self._model_info['embedding_dimension']}")
            logger.info(f"Max sequence length: {self._model_info['max_seq_length']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            self._is_loaded = False
            return False
    
    def _estimate_model_size(self) -> float:
        """Estimate model size in MB."""
        try:
            if self._model is None:
                return 0.0
            
            total_params = sum(p.numel() for p in self._model.parameters())
            # Rough estimate: 4 bytes per parameter (float32)
            size_mb = (total_params * 4) / (1024 * 1024)
            return round(size_mb, 2)
        except:
            return 0.0
    
    def embed_texts(
        self, 
        texts: List[str], 
        batch_config: Optional[BatchConfig] = None,
        show_progress: bool = True
    ) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts with batch processing.
        
        Args:
            texts: List of texts to embed
            batch_config: Custom batch configuration
            show_progress: Show progress information
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        if not self._is_loaded:
            if not self.load_model():
                raise RuntimeError("Failed to load embedding model")
        
        if not texts:
            return EmbeddingResult(
                embeddings=np.array([]),
                texts=[],
                processing_time=0.0,
                model_info=self._model_info,
                batch_info={"total_batches": 0, "total_texts": 0}
            )
        
        config = batch_config or self.batch_config
        start_time = time.time()
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        if show_progress:
            logger.info(f"Batch size: {config.batch_size}, Max length: {config.max_length}")
        
        try:
            # Process texts in batches for memory efficiency
            all_embeddings = []
            processed_texts = []
            total_batches = (len(texts) + config.batch_size - 1) // config.batch_size
            
            for i in range(0, len(texts), config.batch_size):
                batch_texts = texts[i:i + config.batch_size]
                batch_num = (i // config.batch_size) + 1
                
                if show_progress and total_batches > 1:
                    logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
                
                # Preprocess texts for optimal embedding
                processed_batch = self._preprocess_texts(batch_texts, config.max_length)
                
                # Generate embeddings for batch
                batch_embeddings = self._model.encode(
                    processed_batch,
                    batch_size=len(processed_batch),  # Process entire batch at once
                    normalize_embeddings=config.normalize_embeddings,
                    show_progress_bar=False,  # We handle progress logging
                    convert_to_numpy=True
                )
                
                all_embeddings.append(batch_embeddings)
                processed_texts.extend(processed_batch)
                
                # Memory cleanup for large batches
                if batch_num % 5 == 0:  # Every 5 batches
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Combine all embeddings
            final_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
            processing_time = time.time() - start_time
            
            batch_info = {
                "total_batches": total_batches,
                "total_texts": len(texts),
                "processed_texts": len(processed_texts),
                "batch_size": config.batch_size,
                "embedding_shape": final_embeddings.shape,
                "memory_optimized": True
            }
            
            logger.info(f"Embedding generation completed in {processing_time:.2f}s")
            logger.info(f"Generated {final_embeddings.shape[0]} embeddings of dimension {final_embeddings.shape[1]}")
            
            return EmbeddingResult(
                embeddings=final_embeddings,
                texts=processed_texts,
                processing_time=processing_time,
                model_info=self._model_info,
                batch_info=batch_info
            )
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise RuntimeError(f"Embedding generation failed: {str(e)}")
    
    def _preprocess_texts(self, texts: List[str], max_length: int) -> List[str]:
        """
        Preprocess texts for optimal embedding generation.
        
        Args:
            texts: List of texts to preprocess
            max_length: Maximum token length
            
        Returns:
            List of preprocessed texts
        """
        processed = []
        
        for text in texts:
            # Clean and normalize text
            cleaned = text.strip()
            if not cleaned:
                cleaned = "[EMPTY]"  # Handle empty texts
            
            # Truncate if necessary (tokenizer-aware)
            if self._tokenizer and len(cleaned) > max_length * 4:  # Rough char estimate
                tokens = self._tokenizer.encode(cleaned, truncation=True, max_length=max_length)
                cleaned = self._tokenizer.decode(tokens, skip_special_tokens=True)
            
            processed.append(cleaned)
        
        return processed
    
    def embed_single_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text (convenience method).
        
        Args:
            text: Text to embed
            normalize: Whether to normalize the embedding
            
        Returns:
            Numpy array with embedding
        """
        result = self.embed_texts([text], BatchConfig(
            batch_size=1,
            normalize_embeddings=normalize,
            show_progress=False
        ), show_progress=False)
        
        return result.embeddings[0] if len(result.embeddings) > 0 else np.array([])
    
    def compute_similarity(
        self, 
        embeddings1: Union[np.ndarray, List[np.ndarray]], 
        embeddings2: Union[np.ndarray, List[np.ndarray]]
    ) -> Union[float, np.ndarray]:
        """
        Compute cosine similarity between embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            Similarity score(s)
        """
        # Convert to numpy arrays if needed
        if isinstance(embeddings1, list):
            embeddings1 = np.array(embeddings1)
        if isinstance(embeddings2, list):
            embeddings2 = np.array(embeddings2)
        
        # Ensure 2D arrays
        if embeddings1.ndim == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if embeddings2.ndim == 1:
            embeddings2 = embeddings2.reshape(1, -1)
        
        # Compute cosine similarity
        similarity = np.dot(embeddings1, embeddings2.T)
        
        # Return scalar if single comparison
        if similarity.shape == (1, 1):
            return float(similarity[0, 0])
        
        return similarity
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return self._model_info.copy()
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension of the loaded model."""
        if not self._is_loaded:
            if not self.load_model():
                return 0
        return self._model_info.get("embedding_dimension", 0)
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self._is_loaded
    
    def estimate_memory_usage(self, num_texts: int, avg_text_length: int = 100) -> Dict[str, float]:
        """
        Estimate memory usage for embedding generation.
        
        Args:
            num_texts: Number of texts to embed
            avg_text_length: Average text length in characters
            
        Returns:
            Dictionary with memory estimates in MB
        """
        if not self._is_loaded:
            return {"error": "Model not loaded"}
        
        # Rough estimates based on model and batch processing
        embedding_dim = self._model_info.get("embedding_dimension", 384)
        model_size = self._model_info.get("model_size_mb", 80)
        
        # Estimate embedding storage (float32 = 4 bytes)
        embeddings_mb = (num_texts * embedding_dim * 4) / (1024 * 1024)
        
        # Estimate text processing memory (rough)
        text_processing_mb = (num_texts * avg_text_length * 2) / (1024 * 1024)  # Unicode
        
        # Batch processing overhead
        batch_overhead_mb = (self.batch_config.batch_size * embedding_dim * 4) / (1024 * 1024)
        
        total_estimated = model_size + embeddings_mb + text_processing_mb + batch_overhead_mb
        
        return {
            "model_size_mb": model_size,
            "embeddings_storage_mb": round(embeddings_mb, 2),
            "text_processing_mb": round(text_processing_mb, 2),
            "batch_overhead_mb": round(batch_overhead_mb, 2),
            "total_estimated_mb": round(total_estimated, 2),
            "codespace_friendly": total_estimated < 1000  # Under 1GB
        }
    
    def cleanup(self):
        """Clean up model and free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._is_loaded = False
        self._model_info = {}
        
        logger.info("LocalEmbedder cleaned up and memory freed")

def create_local_embedder(model_name: Optional[str] = None) -> LocalEmbedder:
    """
    Factory function to create a LocalEmbedder instance.
    
    Args:
        model_name: Optional model name override
        
    Returns:
        Configured LocalEmbedder instance
    """
    embedder = LocalEmbedder(model_name=model_name)
    
    # Pre-load model for immediate use
    if not embedder.load_model():
        logger.warning("Failed to pre-load embedding model")
    
    return embedder
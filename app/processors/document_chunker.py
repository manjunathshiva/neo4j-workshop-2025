"""
Document chunking and preprocessing utilities.
Implements various chunking strategies for optimal processing.
"""

import logging
import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SENTENCE_BASED = "sentence_based"
    PARAGRAPH_BASED = "paragraph_based"
    SEMANTIC_BASED = "semantic_based"

@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""
    id: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    word_count: int
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Calculate word count after initialization."""
        if not self.word_count:
            self.word_count = len(self.text.split())

class ChunkingStrategyBase(ABC):
    """Base class for chunking strategies."""
    
    @abstractmethod
    def chunk_text(self, text: str, document_id: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Chunk text according to the strategy.
        
        Args:
            text: Text to chunk
            document_id: Unique identifier for the document
            metadata: Document metadata
            
        Returns:
            List of DocumentChunk objects
        """
        pass

class FixedSizeChunker(ChunkingStrategyBase):
    """Chunks text into fixed-size pieces with overlap."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Initialize fixed-size chunker.
        
        Args:
            chunk_size: Target size of each chunk in tokens/words
            overlap: Number of overlapping tokens between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, document_id: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Chunk text into fixed-size pieces."""
        if not text.strip():
            return []
        
        # Split text into words for more precise chunking
        words = text.split()
        chunks = []
        
        start_idx = 0
        chunk_index = 0
        
        while start_idx < len(words):
            # Calculate end index for this chunk
            end_idx = min(start_idx + self.chunk_size, len(words))
            
            # Extract chunk words
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)
            
            # Calculate character positions in original text
            start_char = len(" ".join(words[:start_idx]))
            if start_idx > 0:
                start_char += 1  # Account for space before first word
            
            end_char = start_char + len(chunk_text)
            
            # Create chunk
            chunk_id = f"{document_id}_chunk_{chunk_index:04d}"
            chunk = DocumentChunk(
                id=chunk_id,
                text=chunk_text,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                word_count=len(chunk_words),
                metadata={
                    **metadata,
                    "chunking_strategy": "fixed_size",
                    "chunk_size": self.chunk_size,
                    "overlap": self.overlap,
                    "total_words": len(words)
                }
            )
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            if end_idx >= len(words):
                break
            
            start_idx = max(start_idx + self.chunk_size - self.overlap, start_idx + 1)
            chunk_index += 1
        
        return chunks

class SentenceBasedChunker(ChunkingStrategyBase):
    """Chunks text based on sentence boundaries."""
    
    def __init__(self, max_chunk_size: int = 512, min_chunk_size: int = 50):
        """
        Initialize sentence-based chunker.
        
        Args:
            max_chunk_size: Maximum size of each chunk in words
            min_chunk_size: Minimum size of each chunk in words
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'[.!?]+\s+')
    
    def chunk_text(self, text: str, document_id: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Chunk text based on sentence boundaries."""
        if not text.strip():
            return []
        
        # Split text into sentences
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        
        chunks = []
        current_chunk_sentences = []
        current_word_count = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # Check if adding this sentence would exceed max chunk size
            if (current_word_count + sentence_words > self.max_chunk_size and 
                current_chunk_sentences and 
                current_word_count >= self.min_chunk_size):
                
                # Create chunk from current sentences
                chunk = self._create_chunk_from_sentences(
                    current_chunk_sentences, document_id, chunk_index, text, metadata
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk_sentences = [sentence]
                current_word_count = sentence_words
                chunk_index += 1
            else:
                # Add sentence to current chunk
                current_chunk_sentences.append(sentence)
                current_word_count += sentence_words
        
        # Create final chunk if there are remaining sentences
        if current_chunk_sentences:
            chunk = self._create_chunk_from_sentences(
                current_chunk_sentences, document_id, chunk_index, text, metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be enhanced with more sophisticated NLP
        sentences = self.sentence_endings.split(text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence.split()) >= 3:  # Minimum 3 words per sentence
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _create_chunk_from_sentences(
        self, 
        sentences: List[str], 
        document_id: str, 
        chunk_index: int, 
        original_text: str,
        metadata: Dict[str, Any]
    ) -> DocumentChunk:
        """Create a DocumentChunk from a list of sentences."""
        chunk_text = ". ".join(sentences)
        if not chunk_text.endswith('.'):
            chunk_text += "."
        
        # Find character positions in original text
        start_char = original_text.find(sentences[0])
        if start_char == -1:
            start_char = 0
        
        end_char = start_char + len(chunk_text)
        
        chunk_id = f"{document_id}_chunk_{chunk_index:04d}"
        
        return DocumentChunk(
            id=chunk_id,
            text=chunk_text,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            word_count=len(chunk_text.split()),
            metadata={
                **metadata,
                "chunking_strategy": "sentence_based",
                "sentence_count": len(sentences),
                "max_chunk_size": self.max_chunk_size,
                "min_chunk_size": self.min_chunk_size
            }
        )

class ParagraphBasedChunker(ChunkingStrategyBase):
    """Chunks text based on paragraph boundaries."""
    
    def __init__(self, max_chunk_size: int = 512, combine_small_paragraphs: bool = True):
        """
        Initialize paragraph-based chunker.
        
        Args:
            max_chunk_size: Maximum size of each chunk in words
            combine_small_paragraphs: Whether to combine small paragraphs
        """
        self.max_chunk_size = max_chunk_size
        self.combine_small_paragraphs = combine_small_paragraphs
    
    def chunk_text(self, text: str, document_id: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Chunk text based on paragraph boundaries."""
        if not text.strip():
            return []
        
        # Split text into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            return []
        
        chunks = []
        current_chunk_paragraphs = []
        current_word_count = 0
        chunk_index = 0
        
        for paragraph in paragraphs:
            paragraph_words = len(paragraph.split())
            
            # If single paragraph is too large, split it further
            if paragraph_words > self.max_chunk_size:
                # Create chunk from current paragraphs if any
                if current_chunk_paragraphs:
                    chunk = self._create_chunk_from_paragraphs(
                        current_chunk_paragraphs, document_id, chunk_index, text, metadata
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_chunk_paragraphs = []
                    current_word_count = 0
                
                # Split large paragraph using sentence-based chunker
                sentence_chunker = SentenceBasedChunker(self.max_chunk_size)
                large_para_chunks = sentence_chunker.chunk_text(paragraph, document_id, metadata)
                
                # Add these chunks with updated indices
                for large_chunk in large_para_chunks:
                    large_chunk.chunk_index = chunk_index
                    large_chunk.id = f"{document_id}_chunk_{chunk_index:04d}"
                    chunks.append(large_chunk)
                    chunk_index += 1
                
                continue
            
            # Check if adding this paragraph would exceed max chunk size
            if (current_word_count + paragraph_words > self.max_chunk_size and 
                current_chunk_paragraphs):
                
                # Create chunk from current paragraphs
                chunk = self._create_chunk_from_paragraphs(
                    current_chunk_paragraphs, document_id, chunk_index, text, metadata
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk_paragraphs = [paragraph]
                current_word_count = paragraph_words
                chunk_index += 1
            else:
                # Add paragraph to current chunk
                current_chunk_paragraphs.append(paragraph)
                current_word_count += paragraph_words
        
        # Create final chunk if there are remaining paragraphs
        if current_chunk_paragraphs:
            chunk = self._create_chunk_from_paragraphs(
                current_chunk_paragraphs, document_id, chunk_index, text, metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk_from_paragraphs(
        self, 
        paragraphs: List[str], 
        document_id: str, 
        chunk_index: int, 
        original_text: str,
        metadata: Dict[str, Any]
    ) -> DocumentChunk:
        """Create a DocumentChunk from a list of paragraphs."""
        chunk_text = "\n\n".join(paragraphs)
        
        # Find character positions in original text
        start_char = original_text.find(paragraphs[0])
        if start_char == -1:
            start_char = 0
        
        end_char = start_char + len(chunk_text)
        
        chunk_id = f"{document_id}_chunk_{chunk_index:04d}"
        
        return DocumentChunk(
            id=chunk_id,
            text=chunk_text,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            word_count=len(chunk_text.split()),
            metadata={
                **metadata,
                "chunking_strategy": "paragraph_based",
                "paragraph_count": len(paragraphs),
                "max_chunk_size": self.max_chunk_size
            }
        )

class DocumentChunker:
    """
    Main document chunker that supports multiple chunking strategies.
    Provides text preprocessing and chunking for optimal processing.
    """
    
    def __init__(self, strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE, **kwargs):
        """
        Initialize document chunker.
        
        Args:
            strategy: Chunking strategy to use
            **kwargs: Strategy-specific parameters
        """
        self.strategy = strategy
        self.chunker = self._create_chunker(strategy, **kwargs)
    
    def _create_chunker(self, strategy: ChunkingStrategy, **kwargs) -> ChunkingStrategyBase:
        """Create appropriate chunker based on strategy."""
        if strategy == ChunkingStrategy.FIXED_SIZE:
            return FixedSizeChunker(
                chunk_size=kwargs.get('chunk_size', 512),
                overlap=kwargs.get('overlap', 50)
            )
        elif strategy == ChunkingStrategy.SENTENCE_BASED:
            return SentenceBasedChunker(
                max_chunk_size=kwargs.get('max_chunk_size', 512),
                min_chunk_size=kwargs.get('min_chunk_size', 50)
            )
        elif strategy == ChunkingStrategy.PARAGRAPH_BASED:
            return ParagraphBasedChunker(
                max_chunk_size=kwargs.get('max_chunk_size', 512),
                combine_small_paragraphs=kwargs.get('combine_small_paragraphs', True)
            )
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")
    
    def chunk_document(
        self, 
        text: str, 
        document_id: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Chunk a document using the configured strategy.
        
        Args:
            text: Document text to chunk
            document_id: Unique identifier for the document
            metadata: Optional metadata to include in chunks
            
        Returns:
            List of DocumentChunk objects
        """
        if metadata is None:
            metadata = {}
        
        # Preprocess text
        preprocessed_text = self.preprocess_text(text)
        
        # Add preprocessing info to metadata
        preprocessing_metadata = {
            **metadata,
            "original_length": len(text),
            "preprocessed_length": len(preprocessed_text),
            "preprocessing_applied": True
        }
        
        # Chunk the preprocessed text
        chunks = self.chunker.chunk_text(preprocessed_text, document_id, preprocessing_metadata)
        
        logger.info(f"Chunked document {document_id} into {len(chunks)} chunks using {self.strategy.value} strategy")
        
        return chunks
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before chunking.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace while preserving paragraph structure
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n[ \t]*\n', '\n\n', text)  # Clean paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double newline
        
        # Remove leading/trailing whitespace from lines
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines]
        text = '\n'.join(cleaned_lines)
        
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalize unicode characters
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        return text.strip()
    
    def get_chunking_stats(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Get statistics about the chunking results.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Dictionary with chunking statistics
        """
        if not chunks:
            return {"chunk_count": 0}
        
        word_counts = [chunk.word_count for chunk in chunks]
        char_counts = [len(chunk.text) for chunk in chunks]
        
        return {
            "chunk_count": len(chunks),
            "total_words": sum(word_counts),
            "total_characters": sum(char_counts),
            "avg_words_per_chunk": sum(word_counts) / len(chunks),
            "avg_chars_per_chunk": sum(char_counts) / len(chunks),
            "min_words_per_chunk": min(word_counts),
            "max_words_per_chunk": max(word_counts),
            "chunking_strategy": self.strategy.value,
            "chunk_ids": [chunk.id for chunk in chunks]
        }
    
    def validate_chunks(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Validate chunking results for quality and consistency.
        
        Args:
            chunks: List of document chunks to validate
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "chunk_count": len(chunks)
        }
        
        if not chunks:
            validation["errors"].append("No chunks generated")
            validation["valid"] = False
            return validation
        
        # Check for empty chunks
        empty_chunks = [chunk for chunk in chunks if not chunk.text.strip()]
        if empty_chunks:
            validation["warnings"].append(f"Found {len(empty_chunks)} empty chunks")
        
        # Check for very small chunks (less than 10 words)
        small_chunks = [chunk for chunk in chunks if chunk.word_count < 10]
        if small_chunks:
            validation["warnings"].append(f"Found {len(small_chunks)} very small chunks (< 10 words)")
        
        # Check for duplicate chunk IDs
        chunk_ids = [chunk.id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            validation["errors"].append("Duplicate chunk IDs found")
            validation["valid"] = False
        
        # Check chunk index consistency
        expected_indices = list(range(len(chunks)))
        actual_indices = [chunk.chunk_index for chunk in chunks]
        if actual_indices != expected_indices:
            validation["warnings"].append("Chunk indices are not sequential")
        
        return validation
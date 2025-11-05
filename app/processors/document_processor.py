"""
Main document processor that orchestrates document upload, text extraction, and chunking.
Provides comprehensive document processing pipeline with metadata storage and status tracking.
"""

import logging
import uuid
import hashlib
import re
from datetime import datetime
from typing import Optional, Dict, Any, List, Union, BinaryIO
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .text_extractor import TextExtractor, ExtractionResult, DocumentType
from .document_chunker import DocumentChunker, DocumentChunk, ChunkingStrategy

# Import config with fallback for testing
try:
    from ..config import get_config
except ImportError:
    try:
        from config import get_config
    except ImportError:
        # Fallback config for testing
        class MockConfig:
            class App:
                max_document_size_mb = 10
                chunk_size = 512
                chunk_overlap = 50
            app = App()
        
        def get_config():
            return MockConfig()

logger = logging.getLogger(__name__)

class ProcessingStatus(Enum):
    """Document processing status."""
    PENDING = "pending"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ProcessedDocument:
    """Represents a fully processed document with all metadata."""
    id: str
    title: str
    filename: str
    file_type: str
    content: str
    chunks: List[DocumentChunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Processing statistics
    original_size_bytes: int = 0
    word_count: int = 0
    character_count: int = 0
    chunk_count: int = 0
    processing_time_seconds: float = 0.0
    
    def __post_init__(self):
        """Calculate statistics after initialization."""
        if self.content:
            self.word_count = len(self.content.split())
            self.character_count = len(self.content)
        self.chunk_count = len(self.chunks)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "filename": self.filename,
            "file_type": self.file_type,
            "content": self.content,
            "chunks": [
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "word_count": chunk.word_count,
                    "metadata": chunk.metadata
                }
                for chunk in self.chunks
            ],
            "metadata": self.metadata,
            "processing_status": self.processing_status.value,
            "created_at": self.created_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "error_message": self.error_message,
            "original_size_bytes": self.original_size_bytes,
            "word_count": self.word_count,
            "character_count": self.character_count,
            "chunk_count": self.chunk_count,
            "processing_time_seconds": self.processing_time_seconds
        }

class DocumentProcessor:
    """
    Main document processor that handles the complete pipeline:
    1. Document upload and validation
    2. Text extraction from various formats
    3. Document chunking and preprocessing
    4. Metadata storage and processing status tracking
    """
    
    def __init__(self):
        """Initialize document processor with configuration."""
        self.config = get_config()
        self.text_extractor = TextExtractor(
            max_file_size_mb=self.config.app.max_document_size_mb
        )
        self.document_chunker = DocumentChunker(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=self.config.app.chunk_size,
            overlap=self.config.app.chunk_overlap
        )
        
        # In-memory storage for workshop demonstration
        # In production, this would be replaced with database storage
        self._processed_documents: Dict[str, ProcessedDocument] = {}
        
        logger.info(f"DocumentProcessor initialized with max file size: {self.config.app.max_document_size_mb}MB")
    
    def process_file(self, file_path: Union[str, Path], title: Optional[str] = None) -> ProcessedDocument:
        """
        Process a document file from file path.
        
        Args:
            file_path: Path to the document file
            title: Optional custom title for the document
            
        Returns:
            ProcessedDocument with extraction and chunking results
        """
        start_time = datetime.now()
        file_path = Path(file_path)
        
        # Generate document ID and basic metadata
        document_id = self._generate_document_id(file_path.name)
        document_title = title or file_path.stem
        
        # Get file size
        try:
            file_size = file_path.stat().st_size
        except Exception as e:
            logger.error(f"Error getting file size for {file_path}: {str(e)}")
            return self._create_failed_document(
                document_id, document_title, file_path.name, f"File access error: {str(e)}"
            )
        
        # Create initial document record
        document = ProcessedDocument(
            id=document_id,
            title=document_title,
            filename=file_path.name,
            file_type=file_path.suffix.lower(),
            content="",
            original_size_bytes=file_size,
            processing_status=ProcessingStatus.PENDING
        )
        
        try:
            # Store document for tracking
            self._processed_documents[document_id] = document
            
            # Extract text
            document.processing_status = ProcessingStatus.EXTRACTING
            extraction_result = self.text_extractor.extract_from_file(file_path)
            
            if not extraction_result.success:
                document.processing_status = ProcessingStatus.FAILED
                document.error_message = extraction_result.error_message
                document.processed_at = datetime.now()
                document.processing_time_seconds = (datetime.now() - start_time).total_seconds()
                return document
            
            # Update document with extracted content
            document.content = extraction_result.text
            document.metadata.update(extraction_result.metadata)
            
            # Update word and character counts
            document.word_count = len(extraction_result.text.split())
            document.character_count = len(extraction_result.text)
            
            # Chunk document
            document.processing_status = ProcessingStatus.CHUNKING
            chunks = self.document_chunker.chunk_document(
                text=extraction_result.text,
                document_id=document_id,
                metadata={
                    "filename": file_path.name,
                    "title": document_title,
                    "file_type": file_path.suffix.lower(),
                    "original_size_bytes": file_size,
                    **extraction_result.metadata
                }
            )
            
            document.chunks = chunks
            document.chunk_count = len(chunks)
            document.processing_status = ProcessingStatus.COMPLETED
            document.processed_at = datetime.now()
            document.processing_time_seconds = (datetime.now() - start_time).total_seconds()
            
            # Add chunking statistics to metadata
            chunking_stats = self.document_chunker.get_chunking_stats(chunks)
            document.metadata["chunking_stats"] = chunking_stats
            
            # Validate chunks
            chunk_validation = self.document_chunker.validate_chunks(chunks)
            document.metadata["chunk_validation"] = chunk_validation
            
            logger.info(f"Successfully processed document {document_id}: {len(chunks)} chunks, {document.word_count} words")
            
            return document
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            document.processing_status = ProcessingStatus.FAILED
            document.error_message = f"Processing failed: {str(e)}"
            document.processed_at = datetime.now()
            document.processing_time_seconds = (datetime.now() - start_time).total_seconds()
            return document
    
    def process_uploaded_file(
        self, 
        file_content: bytes, 
        filename: str, 
        title: Optional[str] = None
    ) -> ProcessedDocument:
        """
        Process an uploaded file from bytes content.
        
        Args:
            file_content: Raw file content as bytes
            filename: Original filename
            title: Optional custom title for the document
            
        Returns:
            ProcessedDocument with extraction and chunking results
        """
        start_time = datetime.now()
        
        # Generate document ID and basic metadata
        document_id = self._generate_document_id(filename)
        document_title = title or Path(filename).stem
        file_type = Path(filename).suffix.lower()
        
        # Create initial document record
        document = ProcessedDocument(
            id=document_id,
            title=document_title,
            filename=filename,
            file_type=file_type,
            content="",
            original_size_bytes=len(file_content),
            processing_status=ProcessingStatus.PENDING
        )
        
        try:
            # Store document for tracking
            self._processed_documents[document_id] = document
            
            # Validate file type
            if not self.text_extractor.is_supported_file(filename):
                document.processing_status = ProcessingStatus.FAILED
                document.error_message = f"Unsupported file type: {file_type}"
                document.processed_at = datetime.now()
                document.processing_time_seconds = (datetime.now() - start_time).total_seconds()
                return document
            
            # Get document type
            doc_type = self.text_extractor.get_document_type(filename)
            if not doc_type:
                document.processing_status = ProcessingStatus.FAILED
                document.error_message = f"Could not determine document type for: {filename}"
                document.processed_at = datetime.now()
                document.processing_time_seconds = (datetime.now() - start_time).total_seconds()
                return document
            
            # Extract text
            document.processing_status = ProcessingStatus.EXTRACTING
            extraction_result = self.text_extractor.extract_from_bytes(
                file_content=file_content,
                document_type=doc_type,
                filename=filename
            )
            
            if not extraction_result.success:
                document.processing_status = ProcessingStatus.FAILED
                document.error_message = extraction_result.error_message
                document.processed_at = datetime.now()
                document.processing_time_seconds = (datetime.now() - start_time).total_seconds()
                return document
            
            # Update document with extracted content
            document.content = extraction_result.text
            document.metadata.update(extraction_result.metadata)
            
            # Update word and character counts
            document.word_count = len(extraction_result.text.split())
            document.character_count = len(extraction_result.text)
            
            # Chunk document
            document.processing_status = ProcessingStatus.CHUNKING
            chunks = self.document_chunker.chunk_document(
                text=extraction_result.text,
                document_id=document_id,
                metadata={
                    "filename": filename,
                    "title": document_title,
                    "file_type": file_type,
                    "original_size_bytes": len(file_content),
                    **extraction_result.metadata
                }
            )
            
            document.chunks = chunks
            document.chunk_count = len(chunks)
            document.processing_status = ProcessingStatus.COMPLETED
            document.processed_at = datetime.now()
            document.processing_time_seconds = (datetime.now() - start_time).total_seconds()
            
            # Add chunking statistics to metadata
            chunking_stats = self.document_chunker.get_chunking_stats(chunks)
            document.metadata["chunking_stats"] = chunking_stats
            
            # Validate chunks
            chunk_validation = self.document_chunker.validate_chunks(chunks)
            document.metadata["chunk_validation"] = chunk_validation
            
            logger.info(f"Successfully processed uploaded document {document_id}: {len(chunks)} chunks, {document.word_count} words")
            
            return document
            
        except Exception as e:
            logger.error(f"Error processing uploaded document {filename}: {str(e)}")
            document.processing_status = ProcessingStatus.FAILED
            document.error_message = f"Processing failed: {str(e)}"
            document.processed_at = datetime.now()
            document.processing_time_seconds = (datetime.now() - start_time).total_seconds()
            return document
    
    def get_document(self, document_id: str) -> Optional[ProcessedDocument]:
        """
        Get a processed document by ID.
        
        Args:
            document_id: Unique document identifier
            
        Returns:
            ProcessedDocument if found, None otherwise
        """
        return self._processed_documents.get(document_id)
    
    def list_documents(self) -> List[ProcessedDocument]:
        """
        Get list of all processed documents.
        
        Returns:
            List of ProcessedDocument objects
        """
        return list(self._processed_documents.values())
    
    def get_processing_status(self, document_id: str) -> Optional[ProcessingStatus]:
        """
        Get processing status for a document.
        
        Args:
            document_id: Unique document identifier
            
        Returns:
            ProcessingStatus if document exists, None otherwise
        """
        document = self._processed_documents.get(document_id)
        return document.processing_status if document else None
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get summary of all document processing.
        
        Returns:
            Dictionary with processing statistics
        """
        documents = list(self._processed_documents.values())
        
        if not documents:
            return {
                "total_documents": 0,
                "status_counts": {},
                "total_chunks": 0,
                "total_words": 0,
                "avg_processing_time": 0.0
            }
        
        status_counts = {}
        for status in ProcessingStatus:
            status_counts[status.value] = sum(1 for doc in documents if doc.processing_status == status)
        
        total_chunks = sum(doc.chunk_count for doc in documents)
        total_words = sum(doc.word_count for doc in documents)
        processing_times = [doc.processing_time_seconds for doc in documents if doc.processing_time_seconds > 0]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
        
        return {
            "total_documents": len(documents),
            "status_counts": status_counts,
            "total_chunks": total_chunks,
            "total_words": total_words,
            "avg_processing_time": avg_processing_time,
            "supported_file_types": self.text_extractor.get_supported_extensions(),
            "chunking_strategy": self.document_chunker.strategy.value
        }
    
    def validate_file_upload(self, filename: str, file_size: int) -> Dict[str, Any]:
        """
        Validate a file before processing.
        
        Args:
            filename: Name of the file to validate
            file_size: Size of the file in bytes
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "file_info": {
                "filename": filename,
                "file_size_bytes": file_size,
                "file_size_mb": file_size / (1024 * 1024),
                "file_type": Path(filename).suffix.lower()
            }
        }
        
        # Check file size
        max_size_bytes = self.config.app.max_document_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            validation["valid"] = False
            validation["errors"].append(
                f"File too large: {file_size / (1024*1024):.1f}MB (max: {self.config.app.max_document_size_mb}MB)"
            )
        
        # Check file type
        if not self.text_extractor.is_supported_file(filename):
            validation["valid"] = False
            validation["errors"].append(
                f"Unsupported file type: {Path(filename).suffix.lower()}. "
                f"Supported types: {self.text_extractor.get_supported_extensions()}"
            )
        
        # Check filename
        if not filename.strip():
            validation["valid"] = False
            validation["errors"].append("Filename cannot be empty")
        
        # Warnings for edge cases
        if file_size < 100:  # Very small files
            validation["warnings"].append("File is very small and may not contain meaningful content")
        
        if len(filename) > 255:
            validation["warnings"].append("Filename is very long and may cause issues")
        
        return validation
    
    def clear_processed_documents(self):
        """Clear all processed documents (for workshop demonstration)."""
        self._processed_documents.clear()
        logger.info("Cleared all processed documents")
    
    def _generate_document_id(self, filename: str) -> str:
        """
        Generate a unique document ID.
        
        Args:
            filename: Original filename
            
        Returns:
            Unique document identifier
        """
        # Create a hash of filename + timestamp for uniqueness
        timestamp = datetime.now().isoformat()
        content = f"{filename}_{timestamp}"
        hash_object = hashlib.md5(content.encode())
        hash_hex = hash_object.hexdigest()[:8]
        
        # Create readable ID
        clean_filename = re.sub(r'[^a-zA-Z0-9_-]', '_', Path(filename).stem)
        return f"doc_{clean_filename}_{hash_hex}"
    
    def _create_failed_document(
        self, 
        document_id: str, 
        title: str, 
        filename: str, 
        error_message: str
    ) -> ProcessedDocument:
        """Create a failed document record."""
        return ProcessedDocument(
            id=document_id,
            title=title,
            filename=filename,
            file_type=Path(filename).suffix.lower(),
            content="",
            processing_status=ProcessingStatus.FAILED,
            error_message=error_message,
            processed_at=datetime.now()
        )
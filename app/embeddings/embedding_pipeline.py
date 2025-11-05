"""
Embedding pipeline that combines local embedding generation with Qdrant cloud storage.
Provides end-to-end embedding workflow for documents and text chunks.
"""

import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

try:
    from .local_embedder import LocalEmbedder, EmbeddingResult, BatchConfig
    from .qdrant_manager import QdrantManager, VectorPoint, SearchResult, UploadResult, CollectionInfo
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from embeddings.local_embedder import LocalEmbedder, EmbeddingResult, BatchConfig
    from embeddings.qdrant_manager import QdrantManager, VectorPoint, SearchResult, UploadResult, CollectionInfo

logger = logging.getLogger(__name__)

class EmbeddingType(Enum):
    """Types of embeddings supported by the pipeline."""
    DOCUMENT = "document"
    CHUNK = "chunk"
    ENTITY = "entity"
    QUERY = "query"

@dataclass
class DocumentEmbedding:
    """Represents a document embedding with metadata."""
    document_id: str
    document_title: str
    content: str
    embedding_type: EmbeddingType
    chunk_index: Optional[int] = None
    entity_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PipelineResult:
    """Result of embedding pipeline processing."""
    success: bool
    processed_count: int
    failed_count: int
    collection_name: str
    embedding_time: float
    upload_time: float
    total_time: float
    errors: List[str]
    embedding_info: Dict[str, Any]
    upload_info: Dict[str, Any]

class EmbeddingPipeline:
    """
    Complete embedding pipeline that generates embeddings locally and stores them in Qdrant.
    Handles document processing, chunking, embedding generation, and vector storage.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedding pipeline.
        
        Args:
            model_name: Optional custom embedding model name
        """
        self.embedder = LocalEmbedder(model_name=model_name)
        self.qdrant_manager = QdrantManager()
        
        # Default collection configurations
        self.collection_configs = {
            EmbeddingType.DOCUMENT: {
                "name": "documents",
                "description": "Full document embeddings"
            },
            EmbeddingType.CHUNK: {
                "name": "chunks", 
                "description": "Document chunk embeddings"
            },
            EmbeddingType.ENTITY: {
                "name": "entities",
                "description": "Entity embeddings"
            }
        }
        
        logger.info("EmbeddingPipeline initialized")
    
    def initialize(self) -> bool:
        """
        Initialize the embedding pipeline by loading models and setting up collections.
        
        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing embedding pipeline...")
            
            # Load embedding model
            if not self.embedder.load_model():
                logger.error("Failed to load embedding model")
                return False
            
            # Get embedding dimension for collection setup
            embedding_dim = self.embedder.get_embedding_dimension()
            logger.info(f"Embedding dimension: {embedding_dim}")
            
            # Setup default collections
            for embedding_type, config in self.collection_configs.items():
                collection_name = config["name"]
                
                if not self.qdrant_manager.collection_exists(collection_name):
                    logger.info(f"Creating collection '{collection_name}' for {embedding_type.value}")
                    
                    success = self.qdrant_manager.create_collection(
                        collection_name=collection_name,
                        vector_size=embedding_dim,
                        distance="Cosine",
                        on_disk_payload=True
                    )
                    
                    if not success:
                        logger.error(f"Failed to create collection '{collection_name}'")
                        return False
                else:
                    logger.info(f"Collection '{collection_name}' already exists")
            
            logger.info("Embedding pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing embedding pipeline: {str(e)}")
            return False
    
    def process_documents(
        self,
        documents: List[DocumentEmbedding],
        embedding_type: EmbeddingType = EmbeddingType.DOCUMENT,
        batch_size: int = 16,
        upload_batch_size: int = 100
    ) -> PipelineResult:
        """
        Process documents through the complete embedding pipeline.
        
        Args:
            documents: List of DocumentEmbedding objects
            embedding_type: Type of embeddings to generate
            batch_size: Batch size for embedding generation
            upload_batch_size: Batch size for vector upload
            
        Returns:
            PipelineResult with processing statistics
        """
        start_time = time.time()
        processed_count = 0
        failed_count = 0
        errors = []
        
        collection_name = self.collection_configs[embedding_type]["name"]
        
        try:
            logger.info(f"Processing {len(documents)} documents for {embedding_type.value} embeddings")
            
            # Extract texts for embedding
            texts = [doc.content for doc in documents]
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embedding_start = time.time()
            
            embedding_result = self.embedder.embed_texts(
                texts=texts,
                batch_config=BatchConfig(
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress=True
                )
            )
            
            embedding_time = time.time() - embedding_start
            
            if len(embedding_result.embeddings) == 0:
                return PipelineResult(
                    success=False,
                    processed_count=0,
                    failed_count=len(documents),
                    collection_name=collection_name,
                    embedding_time=embedding_time,
                    upload_time=0,
                    total_time=time.time() - start_time,
                    errors=["No embeddings generated"],
                    embedding_info={},
                    upload_info={}
                )
            
            # Prepare vector points for upload
            logger.info("Preparing vectors for upload...")
            vector_points = []
            
            for i, (doc, embedding) in enumerate(zip(documents, embedding_result.embeddings)):
                # Create payload with document metadata
                payload = {
                    "document_id": doc.document_id,
                    "document_title": doc.document_title,
                    "content": doc.content[:1000],  # Truncate for storage efficiency
                    "embedding_type": embedding_type.value,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "content_length": len(doc.content)
                }
                
                # Add type-specific metadata
                if doc.chunk_index is not None:
                    payload["chunk_index"] = doc.chunk_index
                
                if doc.entity_name:
                    payload["entity_name"] = doc.entity_name
                
                if doc.metadata:
                    payload.update(doc.metadata)
                
                # Create vector point
                vector_point = VectorPoint(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload=payload
                )
                
                vector_points.append(vector_point)
            
            # Upload vectors to Qdrant
            logger.info(f"Uploading {len(vector_points)} vectors to collection '{collection_name}'")
            upload_start = time.time()
            
            upload_result = self.qdrant_manager.upload_vectors(
                collection_name=collection_name,
                vectors=vector_points,
                batch_size=upload_batch_size,
                wait=True
            )
            
            upload_time = time.time() - upload_start
            
            processed_count = upload_result.uploaded_count
            failed_count = upload_result.failed_count
            errors.extend(upload_result.errors)
            
            total_time = time.time() - start_time
            
            logger.info(f"Pipeline processing completed: {processed_count} successful, {failed_count} failed")
            logger.info(f"Total time: {total_time:.2f}s (embedding: {embedding_time:.2f}s, upload: {upload_time:.2f}s)")
            
            return PipelineResult(
                success=upload_result.success,
                processed_count=processed_count,
                failed_count=failed_count,
                collection_name=collection_name,
                embedding_time=embedding_time,
                upload_time=upload_time,
                total_time=total_time,
                errors=errors,
                embedding_info=embedding_result.batch_info,
                upload_info={
                    "operation_id": upload_result.operation_id,
                    "upload_batches": (len(vector_points) + upload_batch_size - 1) // upload_batch_size
                }
            )
            
        except Exception as e:
            error_msg = f"Pipeline processing failed: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            
            return PipelineResult(
                success=False,
                processed_count=processed_count,
                failed_count=len(documents) - processed_count,
                collection_name=collection_name,
                embedding_time=0,
                upload_time=0,
                total_time=time.time() - start_time,
                errors=errors,
                embedding_info={},
                upload_info={}
            )
    
    def create_document_embeddings_from_chunks(
        self,
        document_id: str,
        document_title: str,
        chunks: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentEmbedding]:
        """
        Create document embeddings from text chunks.
        
        Args:
            document_id: Unique document identifier
            document_title: Document title
            chunks: List of text chunks
            metadata: Optional metadata
            
        Returns:
            List of DocumentEmbedding objects
        """
        embeddings = []
        
        # Create document-level embedding (from all chunks combined)
        full_text = " ".join(chunks)
        doc_embedding = DocumentEmbedding(
            document_id=document_id,
            document_title=document_title,
            content=full_text,
            embedding_type=EmbeddingType.DOCUMENT,
            metadata=metadata or {}
        )
        embeddings.append(doc_embedding)
        
        # Create chunk-level embeddings
        for i, chunk_text in enumerate(chunks):
            chunk_embedding = DocumentEmbedding(
                document_id=document_id,
                document_title=document_title,
                content=chunk_text,
                embedding_type=EmbeddingType.CHUNK,
                chunk_index=i,
                metadata=metadata or {}
            )
            embeddings.append(chunk_embedding)
        
        return embeddings

    def search_similar_documents(
        self,
        query_text: str,
        embedding_type: EmbeddingType = EmbeddingType.DOCUMENT,
        limit: int = 10,
        score_threshold: float = 0.3,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents using text query.
        
        Args:
            query_text: Text query to search for
            embedding_type: Type of embeddings to search
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filters: Optional payload filters
            
        Returns:
            List of SearchResult objects
        """
        try:
            logger.info(f"Searching for similar documents: '{query_text[:50]}...'")
            
            # Generate query embedding
            query_embedding = self.embedder.embed_single_text(query_text, normalize=True)
            
            if len(query_embedding) == 0:
                logger.error("Failed to generate query embedding")
                return []
            
            # Search in appropriate collection
            collection_name = self.collection_configs[embedding_type]["name"]
            
            results = self.qdrant_manager.search_vectors(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit,
                score_threshold=None,  # Remove threshold to see all results
                payload_filter=filters,
                with_payload=True,
                with_vectors=False
            )
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {str(e)}")
            return []
    
    def search_with_vector(
        self,
        query_vector: Union[np.ndarray, List[float]],
        embedding_type: EmbeddingType = EmbeddingType.DOCUMENT,
        limit: int = 10,
        score_threshold: float = 0.3,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents using vector query.
        
        Args:
            query_vector: Query vector for search
            embedding_type: Type of embeddings to search
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filters: Optional payload filters
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Convert numpy array to list if needed
            if isinstance(query_vector, np.ndarray):
                query_vector = query_vector.tolist()
            
            collection_name = self.collection_configs[embedding_type]["name"]
            
            results = self.qdrant_manager.search_vectors(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=None,  # Remove threshold to see all results
                payload_filter=filters,
                with_payload=True,
                with_vectors=False
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching with vector: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all embedding collections.
        
        Returns:
            Dictionary with stats for each collection
        """
        stats = {}
        
        for embedding_type, config in self.collection_configs.items():
            collection_name = config["name"]
            collection_stats = self.qdrant_manager.get_collection_stats(collection_name)
            
            stats[embedding_type.value] = {
                "collection_name": collection_name,
                "description": config["description"],
                **collection_stats
            }
        
        return stats
    
    def create_document_embeddings_from_chunks(
        self,
        document_id: str,
        document_title: str,
        chunks: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentEmbedding]:
        """
        Create DocumentEmbedding objects from document chunks.
        
        Args:
            document_id: Unique document identifier
            document_title: Document title
            chunks: List of text chunks
            metadata: Optional metadata to include
            
        Returns:
            List of DocumentEmbedding objects
        """
        embeddings = []
        
        for i, chunk in enumerate(chunks):
            doc_embedding = DocumentEmbedding(
                document_id=document_id,
                document_title=document_title,
                content=chunk,
                embedding_type=EmbeddingType.CHUNK,
                chunk_index=i,
                metadata=metadata
            )
            embeddings.append(doc_embedding)
        
        return embeddings
    
    def create_entity_embeddings(
        self,
        document_id: str,
        document_title: str,
        entities: List[Dict[str, Any]]
    ) -> List[DocumentEmbedding]:
        """
        Create DocumentEmbedding objects for entities.
        
        Args:
            document_id: Source document identifier
            document_title: Source document title
            entities: List of entity dictionaries with name and description
            
        Returns:
            List of DocumentEmbedding objects for entities
        """
        embeddings = []
        
        for entity in entities:
            entity_name = entity.get("name", "")
            entity_description = entity.get("description", "")
            entity_type = entity.get("type", "UNKNOWN")
            
            # Create content for embedding (name + description)
            content = f"{entity_name}: {entity_description}" if entity_description else entity_name
            
            doc_embedding = DocumentEmbedding(
                document_id=document_id,
                document_title=document_title,
                content=content,
                embedding_type=EmbeddingType.ENTITY,
                entity_name=entity_name,
                metadata={
                    "entity_type": entity_type,
                    "entity_description": entity_description,
                    **{k: v for k, v in entity.items() if k not in ["name", "description", "type"]}
                }
            )
            embeddings.append(doc_embedding)
        
        return embeddings
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of the embedding pipeline.
        
        Returns:
            Dictionary with health check results
        """
        health_status = {
            "embedder_ready": False,
            "qdrant_connected": False,
            "collections_ready": {},
            "model_info": {},
            "errors": []
        }
        
        try:
            # Check embedder
            health_status["embedder_ready"] = self.embedder.is_model_loaded()
            if health_status["embedder_ready"]:
                health_status["model_info"] = self.embedder.get_model_info()
            else:
                health_status["errors"].append("Embedding model not loaded")
            
            # Check Qdrant connection
            qdrant_health = self.qdrant_manager.health_check()
            health_status["qdrant_connected"] = qdrant_health["connected"]
            
            if not health_status["qdrant_connected"]:
                health_status["errors"].extend(qdrant_health["errors"])
            
            # Check collections
            for embedding_type, config in self.collection_configs.items():
                collection_name = config["name"]
                collection_info = self.qdrant_manager.get_collection_info(collection_name)
                
                if collection_info:
                    health_status["collections_ready"][embedding_type.value] = {
                        "exists": True,
                        "vectors_count": collection_info.vectors_count,
                        "points_count": collection_info.points_count
                    }
                else:
                    health_status["collections_ready"][embedding_type.value] = {
                        "exists": False,
                        "vectors_count": 0,
                        "points_count": 0
                    }
                    health_status["errors"].append(f"Collection '{collection_name}' not found")
            
        except Exception as e:
            health_status["errors"].append(f"Health check failed: {str(e)}")
        
        return health_status
    
    def cleanup(self):
        """Clean up resources and free memory."""
        logger.info("Cleaning up embedding pipeline...")
        self.embedder.cleanup()
        logger.info("Embedding pipeline cleanup completed")

def create_embedding_pipeline(model_name: Optional[str] = None) -> EmbeddingPipeline:
    """
    Factory function to create and initialize an EmbeddingPipeline.
    
    Args:
        model_name: Optional custom embedding model name
        
    Returns:
        Initialized EmbeddingPipeline instance
    """
    pipeline = EmbeddingPipeline(model_name=model_name)
    
    # Initialize the pipeline
    if not pipeline.initialize():
        logger.warning("Failed to fully initialize embedding pipeline")
    
    return pipeline
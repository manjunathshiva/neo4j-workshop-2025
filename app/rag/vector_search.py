"""
Vector similarity search engine for semantic document retrieval.
Implements vector-based document retrieval using Qdrant cloud storage.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

try:
    from ..embeddings.embedding_pipeline import EmbeddingPipeline, EmbeddingType
    from ..embeddings.qdrant_manager import SearchResult
    from ..config import get_config
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from embeddings.embedding_pipeline import EmbeddingPipeline, EmbeddingType
    from embeddings.qdrant_manager import SearchResult
    from config import get_config

logger = logging.getLogger(__name__)

@dataclass
class VectorContext:
    """Context retrieved from vector similarity search."""
    query_text: str = ""
    query_embedding: List[float] = field(default_factory=list)
    search_results: List[SearchResult] = field(default_factory=list)
    total_results: int = 0
    search_time_seconds: float = 0.0
    similarity_threshold: float = 0.0
    collection_searched: str = ""
    filters_applied: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VectorSearchResult:
    """Result of vector similarity search operation."""
    success: bool
    query: str = ""
    vector_context: Optional[VectorContext] = None
    documents: List[Dict[str, Any]] = field(default_factory=list)
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    error_message: str = ""
    search_metadata: Dict[str, Any] = field(default_factory=dict)

class VectorSearchEngine:
    """
    Vector similarity search engine using Qdrant and local embeddings.
    Provides semantic search capabilities across documents, chunks, and entities.
    """
    
    def __init__(self, embedding_pipeline: Optional[EmbeddingPipeline] = None):
        """
        Initialize vector search engine.
        
        Args:
            embedding_pipeline: Optional pre-initialized embedding pipeline
        """
        self.config = get_config()
        self.embedding_pipeline = embedding_pipeline or EmbeddingPipeline()
        self._initialized = False
        
        # Default search parameters
        self.default_params = {
            "similarity_threshold": 0.7,
            "max_results": 20,
            "chunk_limit": 15,
            "entity_limit": 10
        }
        
        logger.info("VectorSearchEngine initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize the vector search engine.
        
        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing VectorSearchEngine...")
            
            # Initialize embedding pipeline if not already done
            if not self.embedding_pipeline.embedder.is_model_loaded():
                if not self.embedding_pipeline.initialize():
                    logger.error("Failed to initialize embedding pipeline")
                    return False
            
            self._initialized = True
            logger.info("VectorSearchEngine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing VectorSearchEngine: {str(e)}")
            return False
    
    async def search_documents(
        self,
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
        include_chunks: bool = True,
        include_entities: bool = True
    ) -> VectorSearchResult:
        """
        Search for semantically similar documents using vector similarity.
        
        Args:
            query: Natural language query
            limit: Maximum number of document results
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            filters: Optional payload filters for search
            include_chunks: Whether to include chunk-level results
            include_entities: Whether to include entity-level results
            
        Returns:
            VectorSearchResult with search results and metadata
        """
        start_time = time.time()
        
        if not self._initialized:
            return VectorSearchResult(
                success=False,
                query=query,
                error_message="VectorSearchEngine not initialized",
                processing_time_seconds=time.time() - start_time
            )
        
        try:
            logger.info(f"Performing vector search for query: '{query[:50]}...'")
            
            # Search documents
            document_results = await self._search_by_type(
                query=query,
                embedding_type=EmbeddingType.DOCUMENT,
                limit=limit,
                similarity_threshold=similarity_threshold,
                filters=filters
            )
            
            # Search chunks if requested
            chunk_results = []
            if include_chunks:
                chunk_results = await self._search_by_type(
                    query=query,
                    embedding_type=EmbeddingType.CHUNK,
                    limit=self.default_params["chunk_limit"],
                    similarity_threshold=similarity_threshold,
                    filters=filters
                )
            
            # Search entities if requested
            entity_results = []
            if include_entities:
                entity_results = await self._search_by_type(
                    query=query,
                    embedding_type=EmbeddingType.ENTITY,
                    limit=self.default_params["entity_limit"],
                    similarity_threshold=similarity_threshold,
                    filters=filters
                )
            
            # Create vector context
            vector_context = VectorContext(
                query_text=query,
                search_results=document_results + chunk_results + entity_results,
                total_results=len(document_results) + len(chunk_results) + len(entity_results),
                search_time_seconds=time.time() - start_time,
                similarity_threshold=similarity_threshold,
                collection_searched="documents,chunks,entities",
                filters_applied=filters or {}
            )
            
            # Format results
            documents = self._format_search_results(document_results, "document")
            chunks = self._format_search_results(chunk_results, "chunk")
            entities = self._format_search_results(entity_results, "entity")
            
            processing_time = time.time() - start_time
            
            logger.info(f"Vector search completed: {len(documents)} docs, {len(chunks)} chunks, {len(entities)} entities")
            
            return VectorSearchResult(
                success=True,
                query=query,
                vector_context=vector_context,
                documents=documents,
                chunks=chunks,
                entities=entities,
                processing_time_seconds=processing_time,
                search_metadata={
                    "similarity_threshold": similarity_threshold,
                    "filters_applied": filters or {},
                    "include_chunks": include_chunks,
                    "include_entities": include_entities,
                    "total_search_time": processing_time
                }
            )
            
        except Exception as e:
            error_msg = f"Vector search failed: {str(e)}"
            logger.error(error_msg)
            
            return VectorSearchResult(
                success=False,
                query=query,
                error_message=error_msg,
                processing_time_seconds=time.time() - start_time
            )
    
    async def _search_by_type(
        self,
        query: str,
        embedding_type: EmbeddingType,
        limit: int,
        similarity_threshold: float,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar items by embedding type.
        
        Args:
            query: Search query
            embedding_type: Type of embeddings to search
            limit: Maximum results
            similarity_threshold: Minimum similarity score
            filters: Optional payload filters
            
        Returns:
            List of SearchResult objects
        """
        try:
            search_results = self.embedding_pipeline.search_similar_documents(
                query_text=query,
                embedding_type=embedding_type,
                limit=limit,
                score_threshold=similarity_threshold,
                filters=filters
            )
            
            logger.info(f"Found {len(search_results)} {embedding_type.value} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching {embedding_type.value}: {str(e)}")
            return []
    
    def _format_search_results(
        self,
        search_results: List[SearchResult],
        result_type: str
    ) -> List[Dict[str, Any]]:
        """
        Format search results for consistent output.
        
        Args:
            search_results: Raw search results from Qdrant
            result_type: Type of results (document, chunk, entity)
            
        Returns:
            List of formatted result dictionaries
        """
        formatted_results = []
        
        for result in search_results:
            formatted_result = {
                "id": result.id,
                "score": result.score,
                "type": result_type,
                "document_id": result.payload.get("document_id", ""),
                "document_title": result.payload.get("document_title", ""),
                "content": result.payload.get("content", ""),
                "content_length": result.payload.get("content_length", 0),
                "created_at": result.payload.get("created_at", ""),
                "metadata": {}
            }
            
            # Add type-specific fields
            if result_type == "chunk":
                formatted_result["chunk_index"] = result.payload.get("chunk_index", 0)
                formatted_result["metadata"]["chunk_index"] = result.payload.get("chunk_index", 0)
            
            elif result_type == "entity":
                formatted_result["entity_name"] = result.payload.get("entity_name", "")
                formatted_result["entity_type"] = result.payload.get("entity_type", "")
                formatted_result["entity_description"] = result.payload.get("entity_description", "")
                formatted_result["metadata"].update({
                    "entity_name": result.payload.get("entity_name", ""),
                    "entity_type": result.payload.get("entity_type", ""),
                    "entity_description": result.payload.get("entity_description", "")
                })
            
            # Add any additional metadata from payload
            for key, value in result.payload.items():
                if key not in ["document_id", "document_title", "content", "content_length", 
                              "created_at", "chunk_index", "entity_name", "entity_type", 
                              "entity_description", "embedding_type"]:
                    formatted_result["metadata"][key] = value
            
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    async def search_with_ranking(
        self,
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        ranking_weights: Optional[Dict[str, float]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> VectorSearchResult:
        """
        Search with custom ranking and scoring.
        
        Args:
            query: Natural language query
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            ranking_weights: Custom weights for different result types
            filters: Optional payload filters
            
        Returns:
            VectorSearchResult with ranked results
        """
        # Default ranking weights
        default_weights = {
            "document": 1.0,
            "chunk": 0.8,
            "entity": 0.6
        }
        
        weights = ranking_weights or default_weights
        
        try:
            # Perform comprehensive search
            search_result = await self.search_documents(
                query=query,
                limit=limit * 2,  # Get more results for ranking
                similarity_threshold=similarity_threshold,
                filters=filters,
                include_chunks=True,
                include_entities=True
            )
            
            if not search_result.success:
                return search_result
            
            # Combine and rank all results
            all_results = []
            
            # Add documents with weights
            for doc in search_result.documents:
                doc["weighted_score"] = doc["score"] * weights.get("document", 1.0)
                all_results.append(doc)
            
            # Add chunks with weights
            for chunk in search_result.chunks:
                chunk["weighted_score"] = chunk["score"] * weights.get("chunk", 0.8)
                all_results.append(chunk)
            
            # Add entities with weights
            for entity in search_result.entities:
                entity["weighted_score"] = entity["score"] * weights.get("entity", 0.6)
                all_results.append(entity)
            
            # Sort by weighted score
            all_results.sort(key=lambda x: x["weighted_score"], reverse=True)
            
            # Limit results
            ranked_results = all_results[:limit]
            
            # Separate back into types
            ranked_documents = [r for r in ranked_results if r["type"] == "document"]
            ranked_chunks = [r for r in ranked_results if r["type"] == "chunk"]
            ranked_entities = [r for r in ranked_results if r["type"] == "entity"]
            
            # Update search result
            search_result.documents = ranked_documents
            search_result.chunks = ranked_chunks
            search_result.entities = ranked_entities
            search_result.search_metadata["ranking_applied"] = True
            search_result.search_metadata["ranking_weights"] = weights
            search_result.search_metadata["total_ranked_results"] = len(ranked_results)
            
            return search_result
            
        except Exception as e:
            error_msg = f"Ranked search failed: {str(e)}"
            logger.error(error_msg)
            
            return VectorSearchResult(
                success=False,
                query=query,
                error_message=error_msg
            )
    
    async def find_similar_to_document(
        self,
        document_id: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        exclude_same_document: bool = True
    ) -> VectorSearchResult:
        """
        Find documents similar to a specific document.
        
        Args:
            document_id: ID of the reference document
            limit: Maximum number of similar documents
            similarity_threshold: Minimum similarity score
            exclude_same_document: Whether to exclude the reference document
            
        Returns:
            VectorSearchResult with similar documents
        """
        try:
            # First, find the reference document
            filters = {"document_id": document_id}
            
            reference_results = self.embedding_pipeline.search_similar_documents(
                query_text="",  # We'll use the document's own embedding
                embedding_type=EmbeddingType.DOCUMENT,
                limit=1,
                score_threshold=0.0,
                filters=filters
            )
            
            if not reference_results:
                return VectorSearchResult(
                    success=False,
                    query=f"document_id:{document_id}",
                    error_message=f"Reference document '{document_id}' not found"
                )
            
            reference_doc = reference_results[0]
            
            # Use the reference document's content as query
            query_text = reference_doc.payload.get("content", "")
            
            if not query_text:
                return VectorSearchResult(
                    success=False,
                    query=f"document_id:{document_id}",
                    error_message="Reference document has no content for similarity search"
                )
            
            # Search for similar documents
            search_result = await self.search_documents(
                query=query_text,
                limit=limit + (1 if not exclude_same_document else 0),
                similarity_threshold=similarity_threshold,
                include_chunks=False,
                include_entities=False
            )
            
            if search_result.success and exclude_same_document:
                # Remove the reference document from results
                search_result.documents = [
                    doc for doc in search_result.documents 
                    if doc["document_id"] != document_id
                ][:limit]
            
            # Update metadata
            search_result.search_metadata["reference_document_id"] = document_id
            search_result.search_metadata["exclude_same_document"] = exclude_same_document
            
            return search_result
            
        except Exception as e:
            error_msg = f"Similar document search failed: {str(e)}"
            logger.error(error_msg)
            
            return VectorSearchResult(
                success=False,
                query=f"document_id:{document_id}",
                error_message=error_msg
            )
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the vector search collections.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            stats = self.embedding_pipeline.get_collection_stats()
            
            # Add search engine specific information
            search_stats = {
                "collections": stats,
                "default_parameters": self.default_params,
                "initialized": self._initialized,
                "embedding_model": self.embedding_pipeline.embedder.get_model_info() if self.embedding_pipeline.embedder.is_model_loaded() else {}
            }
            
            return search_stats
            
        except Exception as e:
            logger.error(f"Error getting search statistics: {str(e)}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on vector search engine.
        
        Returns:
            Dictionary with health check results
        """
        health_status = {
            "initialized": self._initialized,
            "embedding_pipeline_healthy": False,
            "collections_accessible": False,
            "model_loaded": False,
            "errors": []
        }
        
        try:
            # Check embedding pipeline
            pipeline_health = self.embedding_pipeline.health_check()
            health_status["embedding_pipeline_healthy"] = (
                pipeline_health["embedder_ready"] and 
                pipeline_health["qdrant_connected"]
            )
            
            health_status["model_loaded"] = pipeline_health["embedder_ready"]
            health_status["collections_accessible"] = pipeline_health["qdrant_connected"]
            
            if not health_status["embedding_pipeline_healthy"]:
                health_status["errors"].extend(pipeline_health["errors"])
            
            # Check collections
            collections_ready = pipeline_health.get("collections_ready", {})
            missing_collections = [
                collection_type for collection_type, info in collections_ready.items()
                if not info.get("exists", False)
            ]
            
            if missing_collections:
                health_status["errors"].append(f"Missing collections: {missing_collections}")
            
        except Exception as e:
            health_status["errors"].append(f"Health check failed: {str(e)}")
        
        return health_status

def create_vector_search_engine(embedding_pipeline: Optional[EmbeddingPipeline] = None) -> VectorSearchEngine:
    """
    Factory function to create a VectorSearchEngine instance.
    
    Args:
        embedding_pipeline: Optional pre-initialized embedding pipeline
        
    Returns:
        VectorSearchEngine instance
    """
    return VectorSearchEngine(embedding_pipeline=embedding_pipeline)
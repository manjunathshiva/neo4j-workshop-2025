"""
Qdrant cloud integration for vector storage and retrieval.
Manages collections, vector upload, and search operations.
"""

import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

try:
    from ..connections import get_database_manager
    from ..config import get_config
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from connections import get_database_manager
    from config import get_config

logger = logging.getLogger(__name__)

class CollectionStatus(Enum):
    """Collection status enumeration."""
    NOT_EXISTS = "not_exists"
    EXISTS = "exists"
    CREATING = "creating"
    ERROR = "error"

@dataclass
class VectorPoint:
    """Represents a vector point for Qdrant storage."""
    id: str
    vector: List[float]
    payload: Dict[str, Any]

@dataclass
class SearchResult:
    """Result from vector search."""
    id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None

@dataclass
class CollectionInfo:
    """Information about a Qdrant collection."""
    name: str
    status: CollectionStatus
    vectors_count: int
    indexed_vectors_count: int
    points_count: int
    segments_count: int
    config: Dict[str, Any]
    payload_schema: Dict[str, Any]

@dataclass
class UploadResult:
    """Result of vector upload operation."""
    success: bool
    uploaded_count: int
    failed_count: int
    operation_id: Optional[str]
    processing_time: float
    errors: List[str]

class QdrantManager:
    """
    Manages Qdrant cloud operations for vector storage and retrieval.
    Handles collection management, vector upload, and search operations.
    """
    
    def __init__(self):
        """Initialize Qdrant manager with database connection."""
        self.config = get_config()
        self.db_manager = get_database_manager()
        self._client: Optional[QdrantClient] = None
        self._collections_cache: Dict[str, CollectionInfo] = {}
        self._cache_timestamp = 0
        self._cache_ttl = 300  # 5 minutes
        
        logger.info("QdrantManager initialized")
    
    def _get_client(self) -> QdrantClient:
        """Get Qdrant client with connection validation."""
        if not self.db_manager.is_initialized():
            raise ConnectionError("Database manager not initialized")
        
        return self.db_manager.qdrant.get_client()
    
    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine",
        on_disk_payload: bool = True,
        replication_factor: int = 1,
        write_consistency_factor: int = 1
    ) -> bool:
        """
        Create a new collection in Qdrant cloud.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors
            distance: Distance metric (Cosine, Dot, Euclid)
            on_disk_payload: Store payload on disk for memory efficiency
            replication_factor: Number of replicas
            write_consistency_factor: Write consistency requirement
            
        Returns:
            True if collection created successfully
        """
        try:
            client = self._get_client()
            
            logger.info(f"Creating collection '{collection_name}' with vector size {vector_size}")
            
            # Check if collection already exists
            if self.collection_exists(collection_name):
                logger.info(f"Collection '{collection_name}' already exists")
                return True
            
            # Try to create collection with full configuration first
            try:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=getattr(models.Distance, distance.upper())
                    ),
                    optimizers_config=models.OptimizersConfig(
                        deleted_threshold=0.2,
                        vacuum_min_vector_number=1000,
                        default_segment_number=2,
                        max_segment_size=20000,
                        memmap_threshold=20000,
                        indexing_threshold=20000,
                        flush_interval_sec=5,
                        max_optimization_threads=2
                    ),
                    wal_config=models.WalConfig(
                        wal_capacity_mb=32,
                        wal_segments_ahead=0
                    ),
                    hnsw_config=models.HnswConfig(
                        m=16,
                        ef_construct=100,
                        full_scan_threshold=10000,
                        max_indexing_threads=2,
                        on_disk=False,
                        payload_m=None
                    ),
                    quantization_config=None,  # Disable quantization for accuracy
                    replication_factor=replication_factor,
                    write_consistency_factor=write_consistency_factor,
                    on_disk_payload=on_disk_payload
                )
            except Exception as config_error:
                logger.warning(f"Full configuration failed: {config_error}")
                logger.info("Trying simplified collection creation...")
                
                # Fallback to basic configuration
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=getattr(models.Distance, distance.upper())
                    ),
                    replication_factor=replication_factor,
                    write_consistency_factor=write_consistency_factor,
                    on_disk_payload=on_disk_payload
                )
            
            # Clear cache to force refresh
            self._clear_collections_cache()
            
            logger.info(f"Collection '{collection_name}' created successfully")
            return True
            
        except UnexpectedResponse as e:
            if "already exists" in str(e).lower():
                logger.info(f"Collection '{collection_name}' already exists")
                return True
            else:
                logger.error(f"Failed to create collection '{collection_name}': {str(e)}")
                return False
        except Exception as e:
            logger.error(f"Error creating collection '{collection_name}': {str(e)}")
            return False
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from Qdrant.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            True if collection deleted successfully
        """
        try:
            client = self._get_client()
            
            if not self.collection_exists(collection_name):
                logger.info(f"Collection '{collection_name}' does not exist")
                return True
            
            client.delete_collection(collection_name)
            
            # Clear cache
            self._clear_collections_cache()
            
            logger.info(f"Collection '{collection_name}' deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting collection '{collection_name}': {str(e)}")
            return False

    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists in Qdrant.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if collection exists
        """
        try:
            client = self._get_client()
            collections = client.get_collections()
            return any(col.name == collection_name for col in collections.collections)
        except Exception as e:
            logger.error(f"Error checking collection existence: {str(e)}")
            return False
    
    def get_collection_info(self, collection_name: str, use_cache: bool = True) -> Optional[CollectionInfo]:
        """
        Get detailed information about a collection.
        
        Args:
            collection_name: Name of the collection
            use_cache: Whether to use cached information
            
        Returns:
            CollectionInfo object or None if not found
        """
        # Check cache first
        if use_cache and self._is_cache_valid():
            cached_info = self._collections_cache.get(collection_name)
            if cached_info:
                return cached_info
        
        try:
            client = self._get_client()
            
            # Get collection info
            collection = client.get_collection(collection_name)
            
            # Get collection statistics
            try:
                info = client.get_collection(collection_name)
            except AttributeError:
                # Fallback for different API versions
                info = collection
            
            collection_info = CollectionInfo(
                name=collection_name,
                status=CollectionStatus.EXISTS,
                vectors_count=getattr(info, 'vectors_count', 0) or 0,
                indexed_vectors_count=getattr(info, 'indexed_vectors_count', 0) or 0,
                points_count=getattr(info, 'points_count', 0) or 0,
                segments_count=getattr(info, 'segments_count', 0) or 0,
                config={
                    "vector_size": collection.config.params.vectors.size,
                    "distance": collection.config.params.vectors.distance.name,
                    "on_disk_payload": collection.config.params.on_disk_payload
                },
                payload_schema=getattr(collection, 'payload_schema', {}) or {}
            )
            
            # Cache the result
            self._collections_cache[collection_name] = collection_info
            self._cache_timestamp = time.time()
            
            return collection_info
            
        except UnexpectedResponse as e:
            if "404" in str(e) or "not found" in str(e).lower():
                return None
            else:
                logger.error(f"Error getting collection info for '{collection_name}': {str(e)}")
                return None
        except Exception as e:
            logger.error(f"Error getting collection info for '{collection_name}': {str(e)}")
            return None
    
    def upload_vectors(
        self,
        collection_name: str,
        vectors: List[VectorPoint],
        batch_size: int = 100,
        parallel: int = 1,
        wait: bool = True
    ) -> UploadResult:
        """
        Upload vectors to a Qdrant collection with batch processing.
        
        Args:
            collection_name: Name of the target collection
            vectors: List of VectorPoint objects to upload
            batch_size: Number of vectors per batch
            parallel: Number of parallel upload threads
            wait: Whether to wait for indexing completion
            
        Returns:
            UploadResult with upload statistics
        """
        start_time = time.time()
        uploaded_count = 0
        failed_count = 0
        errors = []
        operation_id = None
        
        try:
            client = self._get_client()
            
            logger.info(f"Uploading {len(vectors)} vectors to collection '{collection_name}'")
            logger.info(f"Batch size: {batch_size}, Parallel: {parallel}")
            
            # Ensure collection exists
            if not self.collection_exists(collection_name):
                errors.append(f"Collection '{collection_name}' does not exist")
                return UploadResult(
                    success=False,
                    uploaded_count=0,
                    failed_count=len(vectors),
                    operation_id=None,
                    processing_time=time.time() - start_time,
                    errors=errors
                )
            
            # Process vectors in batches
            total_batches = (len(vectors) + batch_size - 1) // batch_size
            
            for i in range(0, len(vectors), batch_size):
                batch_vectors = vectors[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_vectors)} vectors)")
                
                try:
                    # Convert to Qdrant points
                    points = [
                        models.PointStruct(
                            id=vector.id,
                            vector=vector.vector,
                            payload=vector.payload
                        )
                        for vector in batch_vectors
                    ]
                    
                    # Upload batch
                    operation_info = client.upsert(
                        collection_name=collection_name,
                        points=points,
                        wait=wait
                    )
                    
                    if hasattr(operation_info, 'operation_id'):
                        operation_id = operation_info.operation_id
                    
                    uploaded_count += len(batch_vectors)
                    logger.info(f"Batch {batch_num} uploaded successfully")
                    
                except Exception as e:
                    error_msg = f"Batch {batch_num} failed: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    failed_count += len(batch_vectors)
            
            processing_time = time.time() - start_time
            success = failed_count == 0
            
            logger.info(f"Upload completed: {uploaded_count} successful, {failed_count} failed")
            logger.info(f"Processing time: {processing_time:.2f}s")
            
            # Clear cache to force refresh
            self._clear_collections_cache()
            
            return UploadResult(
                success=success,
                uploaded_count=uploaded_count,
                failed_count=failed_count,
                operation_id=operation_id,
                processing_time=processing_time,
                errors=errors
            )
            
        except Exception as e:
            error_msg = f"Upload operation failed: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            
            return UploadResult(
                success=False,
                uploaded_count=uploaded_count,
                failed_count=len(vectors) - uploaded_count,
                operation_id=operation_id,
                processing_time=time.time() - start_time,
                errors=errors
            )
    
    def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        payload_filter: Optional[Dict[str, Any]] = None,
        with_vectors: bool = False,
        with_payload: bool = True
    ) -> List[SearchResult]:
        """
        Search for similar vectors in a collection.
        
        Args:
            collection_name: Name of the collection to search
            query_vector: Query vector for similarity search
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            payload_filter: Filter conditions for payload
            with_vectors: Include vectors in results
            with_payload: Include payload in results
            
        Returns:
            List of SearchResult objects
        """
        try:
            client = self._get_client()
            
            logger.info(f"Searching collection '{collection_name}' with limit {limit}")
            
            # Debug: Check collection info
            try:
                collection_info = client.get_collection(collection_name)
                logger.info(f"Collection '{collection_name}' has {collection_info.points_count} points")
            except Exception as e:
                logger.warning(f"Could not get collection info: {e}")
            
            # Build filter if provided
            search_filter = None
            if payload_filter:
                search_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                        for key, value in payload_filter.items()
                    ]
                )
            
            # Perform search
            search_results = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_vectors=with_vectors,
                with_payload=with_payload
            )
            
            # Convert to SearchResult objects
            results = []
            for result in search_results:
                search_result = SearchResult(
                    id=str(result.id),
                    score=result.score,
                    payload=result.payload or {},
                    vector=result.vector if with_vectors else None
                )
                results.append(search_result)
            
            logger.info(f"Found {len(results)} similar vectors")
            if len(results) == 0:
                logger.warning(f"No results found. Query vector dimension: {len(query_vector)}, threshold: {score_threshold}")
            else:
                logger.info(f"Top result score: {results[0].score:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching collection '{collection_name}': {str(e)}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from Qdrant.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            True if deletion successful
        """
        try:
            client = self._get_client()
            
            logger.info(f"Deleting collection '{collection_name}'")
            
            client.delete_collection(collection_name)
            
            # Clear cache
            self._clear_collections_cache()
            
            logger.info(f"Collection '{collection_name}' deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting collection '{collection_name}': {str(e)}")
            return False
    
    def list_collections(self, refresh_cache: bool = False) -> List[CollectionInfo]:
        """
        List all collections with detailed information.
        
        Args:
            refresh_cache: Force refresh of cached information
            
        Returns:
            List of CollectionInfo objects
        """
        if refresh_cache:
            self._clear_collections_cache()
        
        try:
            client = self._get_client()
            
            collections_response = client.get_collections()
            collections_info = []
            
            for collection in collections_response.collections:
                info = self.get_collection_info(collection.name, use_cache=not refresh_cache)
                if info:
                    collections_info.append(info)
            
            return collections_info
            
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            return []
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get detailed statistics for a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary with collection statistics
        """
        try:
            client = self._get_client()
            
            info = client.collection_info(collection_name)
            
            return {
                "vectors_count": info.vectors_count or 0,
                "indexed_vectors_count": info.indexed_vectors_count or 0,
                "points_count": info.points_count or 0,
                "segments_count": info.segments_count or 0,
                "disk_data_size": info.disk_data_size or 0,
                "ram_data_size": info.ram_data_size or 0,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance.name
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting stats for collection '{collection_name}': {str(e)}")
            return {}
    
    def _is_cache_valid(self) -> bool:
        """Check if collections cache is still valid."""
        return (time.time() - self._cache_timestamp) < self._cache_ttl
    
    def _clear_collections_cache(self):
        """Clear the collections cache."""
        self._collections_cache.clear()
        self._cache_timestamp = 0
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Qdrant connection and collections.
        
        Returns:
            Dictionary with health check results
        """
        health_status = {
            "connected": False,
            "collections_accessible": False,
            "collections_count": 0,
            "total_vectors": 0,
            "errors": []
        }
        
        try:
            client = self._get_client()
            
            # Test basic connectivity
            collections_response = client.get_collections()
            health_status["connected"] = True
            health_status["collections_accessible"] = True
            health_status["collections_count"] = len(collections_response.collections)
            
            # Count total vectors across all collections
            total_vectors = 0
            for collection in collections_response.collections:
                try:
                    info = client.collection_info(collection.name)
                    total_vectors += info.vectors_count or 0
                except:
                    pass  # Skip collections with access issues
            
            health_status["total_vectors"] = total_vectors
            
        except Exception as e:
            health_status["errors"].append(str(e))
        
        return health_status

def create_qdrant_manager() -> QdrantManager:
    """
    Factory function to create a QdrantManager instance.
    
    Returns:
        Configured QdrantManager instance
    """
    return QdrantManager()
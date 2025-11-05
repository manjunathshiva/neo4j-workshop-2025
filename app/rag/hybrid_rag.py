"""
Hybrid RAG engine that combines graph-based and vector-based retrieval.
Implements fusion of Neo4j graph search with Qdrant vector similarity search.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np

try:
    from .graph_rag import GraphRAGPipeline, GraphRAGResult, GraphContext
    from .vector_search import VectorSearchEngine, VectorSearchResult, VectorContext
    from ..embeddings.embedding_pipeline import EmbeddingPipeline
    from ..graph.graph_manager import GraphManager
    from ..config import get_config
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from rag.graph_rag import GraphRAGPipeline, GraphRAGResult, GraphContext
    from rag.vector_search import VectorSearchEngine, VectorSearchResult, VectorContext
    from embeddings.embedding_pipeline import EmbeddingPipeline
    from graph.graph_manager import GraphManager
    from config import get_config

logger = logging.getLogger(__name__)

class FusionMethod(Enum):
    """Methods for combining graph and vector search results."""
    WEIGHTED_AVERAGE = "weighted_average"
    RANK_FUSION = "rank_fusion"
    SCORE_NORMALIZATION = "score_normalization"
    RECIPROCAL_RANK = "reciprocal_rank"

@dataclass
class FusionWeights:
    """Weights for combining different types of search results."""
    graph_weight: float = 0.6
    vector_weight: float = 0.4
    document_boost: float = 1.0
    chunk_boost: float = 0.8
    entity_boost: float = 0.7
    relationship_boost: float = 0.9

@dataclass
class HybridContext:
    """Combined context from both graph and vector retrieval."""
    graph_context: Optional[GraphContext] = None
    vector_context: Optional[VectorContext] = None
    fusion_method: FusionMethod = FusionMethod.WEIGHTED_AVERAGE
    fusion_weights: FusionWeights = field(default_factory=FusionWeights)
    combined_sources: List[Dict[str, Any]] = field(default_factory=list)
    deduplication_stats: Dict[str, int] = field(default_factory=dict)
    total_unique_sources: int = 0
    fusion_time_seconds: float = 0.0

@dataclass
class HybridRAGResult:
    """Result of hybrid RAG processing combining graph and vector approaches."""
    success: bool
    query: str = ""
    answer: str = ""
    method: str = "hybrid_rag"
    confidence: float = 0.0
    hybrid_context: Optional[HybridContext] = None
    graph_result: Optional[GraphRAGResult] = None
    vector_result: Optional[VectorSearchResult] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    error_message: str = ""
    reasoning_steps: List[str] = field(default_factory=list)
    fusion_metadata: Dict[str, Any] = field(default_factory=dict)
    
class ResultFusion:
    """
    Handles fusion and ranking of results from different retrieval methods.
    Implements various fusion strategies for combining graph and vector results.
    """
    
    def __init__(self, fusion_method: FusionMethod = FusionMethod.WEIGHTED_AVERAGE):
        """
        Initialize result fusion engine.
        
        Args:
            fusion_method: Method to use for combining results
        """
        self.fusion_method = fusion_method
        self.default_weights = FusionWeights()
        
        logger.info(f"ResultFusion initialized with method: {fusion_method.value}")
    
    def fuse_results(
        self,
        graph_sources: List[Dict[str, Any]],
        vector_documents: List[Dict[str, Any]],
        vector_chunks: List[Dict[str, Any]],
        vector_entities: List[Dict[str, Any]],
        weights: Optional[FusionWeights] = None,
        max_results: int = 20
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Fuse results from graph and vector searches.
        
        Args:
            graph_sources: Sources from graph RAG
            vector_documents: Document results from vector search
            vector_chunks: Chunk results from vector search
            vector_entities: Entity results from vector search
            weights: Custom fusion weights
            max_results: Maximum number of results to return
            
        Returns:
            Tuple of (fused_results, deduplication_stats)
        """
        fusion_weights = weights or self.default_weights
        
        try:
            logger.info(f"Fusing results: {len(graph_sources)} graph, {len(vector_documents)} docs, "
                       f"{len(vector_chunks)} chunks, {len(vector_entities)} entities")
            
            # Normalize and score all results
            all_results = []
            
            # Process graph sources
            for source in graph_sources:
                result = self._normalize_graph_source(source, fusion_weights.graph_weight)
                result["source_type"] = "graph"
                all_results.append(result)
            
            # Process vector documents
            for doc in vector_documents:
                result = self._normalize_vector_result(doc, fusion_weights.vector_weight * fusion_weights.document_boost)
                result["source_type"] = "vector_document"
                all_results.append(result)
            
            # Process vector chunks
            for chunk in vector_chunks:
                result = self._normalize_vector_result(chunk, fusion_weights.vector_weight * fusion_weights.chunk_boost)
                result["source_type"] = "vector_chunk"
                all_results.append(result)
            
            # Process vector entities
            for entity in vector_entities:
                result = self._normalize_vector_result(entity, fusion_weights.vector_weight * fusion_weights.entity_boost)
                result["source_type"] = "vector_entity"
                all_results.append(result)
            
            # Apply fusion method
            if self.fusion_method == FusionMethod.WEIGHTED_AVERAGE:
                fused_results = self._weighted_average_fusion(all_results)
            elif self.fusion_method == FusionMethod.RANK_FUSION:
                fused_results = self._rank_fusion(all_results)
            elif self.fusion_method == FusionMethod.SCORE_NORMALIZATION:
                fused_results = self._score_normalization_fusion(all_results)
            elif self.fusion_method == FusionMethod.RECIPROCAL_RANK:
                fused_results = self._reciprocal_rank_fusion(all_results)
            else:
                fused_results = self._weighted_average_fusion(all_results)
            
            # Deduplicate results
            deduplicated_results, dedup_stats = self._deduplicate_results(fused_results)
            
            # Limit results
            final_results = deduplicated_results[:max_results]
            
            logger.info(f"Fusion completed: {len(final_results)} final results after deduplication")
            
            return final_results, dedup_stats
            
        except Exception as e:
            logger.error(f"Error in result fusion: {str(e)}")
            return [], {"error": 1}
    
    def _normalize_graph_source(self, source: Dict[str, Any], base_weight: float) -> Dict[str, Any]:
        """Normalize graph source to common format."""
        return {
            "id": source.get("id", ""),
            "document_id": source.get("id", ""),
            "title": source.get("entity", source.get("relationship", "Unknown")),
            "content": str(source),
            "score": source.get("confidence", 0.5),
            "normalized_score": source.get("confidence", 0.5) * base_weight,
            "original_source": source,
            "content_type": "graph_entity" if "entity" in source else "graph_relationship"
        }
    
    def _normalize_vector_result(self, result: Dict[str, Any], base_weight: float) -> Dict[str, Any]:
        """Normalize vector result to common format."""
        return {
            "id": result.get("id", ""),
            "document_id": result.get("document_id", ""),
            "title": result.get("document_title", result.get("entity_name", "Unknown")),
            "content": result.get("content", ""),
            "score": result.get("score", 0.0),
            "normalized_score": result.get("score", 0.0) * base_weight,
            "original_source": result,
            "content_type": result.get("type", "unknown")
        }
    
    def _weighted_average_fusion(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply weighted average fusion."""
        results.sort(key=lambda x: x["normalized_score"], reverse=True)
        return results
    
    def _rank_fusion(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply rank-based fusion (RRF - Reciprocal Rank Fusion)."""
        source_groups = {}
        for result in results:
            source_type = result["source_type"]
            if source_type not in source_groups:
                source_groups[source_type] = []
            source_groups[source_type].append(result)
        
        # Sort each group by score
        for source_type in source_groups:
            source_groups[source_type].sort(key=lambda x: x["score"], reverse=True)
        
        # Calculate RRF scores
        k = 60  # RRF parameter
        for source_type, group_results in source_groups.items():
            for rank, result in enumerate(group_results):
                rrf_score = 1.0 / (k + rank + 1)
                result["rrf_score"] = rrf_score
                result["rank_in_group"] = rank + 1
        
        # Combine and sort by RRF score
        all_results = []
        for group_results in source_groups.values():
            all_results.extend(group_results)
        
        all_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return all_results
    
    def _deduplicate_results(
        self,
        results: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Deduplicate results based on document ID and content similarity."""
        seen_documents = set()
        seen_content_hashes = set()
        deduplicated = []
        
        stats = {
            "total_input": len(results),
            "document_duplicates": 0,
            "content_duplicates": 0,
            "final_count": 0
        }
        
        for result in results:
            document_id = result.get("document_id", "")
            content = result.get("content", "")
            
            # Create content hash for similarity detection
            content_hash = hash(content.strip().lower()[:200])  # Use first 200 chars
            
            # Check for document ID duplication
            if document_id and document_id in seen_documents:
                stats["document_duplicates"] += 1
                continue
            
            # Check for content duplication
            if content_hash in seen_content_hashes:
                stats["content_duplicates"] += 1
                continue
            
            # Add to deduplicated results
            deduplicated.append(result)
            
            if document_id:
                seen_documents.add(document_id)
            seen_content_hashes.add(content_hash)
        
        stats["final_count"] = len(deduplicated)
        
        logger.info(f"Deduplication: {stats['total_input']} -> {stats['final_count']} "
                   f"(removed {stats['document_duplicates']} doc duplicates, "
                   f"{stats['content_duplicates']} content duplicates)")
        
        return deduplicated, stats

class HybridRAGEngine:
    """
    Complete Hybrid RAG engine that combines graph and vector retrieval.
    Orchestrates both retrieval methods and fuses results for comprehensive answers.
    """
    
    def __init__(
        self,
        graph_manager: GraphManager,
        embedding_pipeline: Optional[EmbeddingPipeline] = None,
        fusion_method: FusionMethod = FusionMethod.WEIGHTED_AVERAGE
    ):
        """
        Initialize Hybrid RAG engine.
        
        Args:
            graph_manager: Initialized graph manager
            embedding_pipeline: Optional embedding pipeline
            fusion_method: Method for fusing results
        """
        self.graph_manager = graph_manager
        self.embedding_pipeline = embedding_pipeline or EmbeddingPipeline()
        
        # Initialize components
        self.graph_rag = GraphRAGPipeline(graph_manager)
        self.vector_search = VectorSearchEngine(self.embedding_pipeline)
        self.result_fusion = ResultFusion(fusion_method)
        
        self.config = get_config()
        self._initialized = False
        
        # Default hybrid parameters
        self.default_params = {
            "max_graph_nodes": 30,
            "max_vector_results": 20,
            "similarity_threshold": 0.7,
            "fusion_weights": FusionWeights(),
            "max_context_length": 6000,
            "enable_graph_search": True,
            "enable_vector_search": True
        }
        
        logger.info("HybridRAGEngine initialized")
    
    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize the Hybrid RAG engine and all components.
        
        Returns:
            Dictionary with initialization results
        """
        try:
            logger.info("Initializing HybridRAGEngine...")
            
            # Initialize vector search engine
            vector_init = await self.vector_search.initialize()
            if not vector_init:
                return {
                    "success": False,
                    "error": "Failed to initialize vector search engine",
                    "component": "VectorSearchEngine"
                }
            
            # Graph RAG should already be initialized through graph_manager
            # Test basic functionality
            try:
                test_stats = await self.graph_manager.get_graph_statistics()
                graph_available = test_stats.get("success", False)
            except:
                graph_available = False
            
            if not graph_available:
                logger.warning("Graph manager not fully available, hybrid mode will use vector search only")
            
            self._initialized = True
            
            logger.info("HybridRAGEngine initialized successfully")
            
            return {
                "success": True,
                "message": "Hybrid RAG Engine initialized successfully",
                "components": {
                    "vector_search": vector_init,
                    "graph_available": graph_available,
                    "fusion_method": self.result_fusion.fusion_method.value
                }
            }
            
        except Exception as e:
            logger.error(f"Error initializing HybridRAGEngine: {str(e)}")
            return {
                "success": False,
                "error": f"HybridRAGEngine initialization failed: {str(e)}",
                "component": "HybridRAGEngine"
            }
    
    async def query(
        self,
        question: str,
        enable_graph_search: bool = True,
        enable_vector_search: bool = True,
        fusion_weights: Optional[FusionWeights] = None,
        max_results: int = 15,
        similarity_threshold: float = 0.7
    ) -> HybridRAGResult:
        """
        Process a question using hybrid RAG approach.
        
        Args:
            question: Natural language question
            enable_graph_search: Whether to use graph-based retrieval
            enable_vector_search: Whether to use vector-based retrieval
            fusion_weights: Custom weights for result fusion
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity score for vector search
            
        Returns:
            HybridRAGResult with comprehensive answer and context
        """
        start_time = time.time()
        reasoning_steps = []
        
        if not self._initialized:
            return HybridRAGResult(
                success=False,
                query=question,
                error_message="HybridRAGEngine not initialized",
                processing_time_seconds=time.time() - start_time
            )
        
        try:
            reasoning_steps.append("Starting hybrid RAG processing")
            
            # Initialize results
            graph_result = None
            vector_result = None
            
            # Perform graph-based retrieval
            if enable_graph_search:
                reasoning_steps.append("Performing graph-based retrieval")
                try:
                    graph_result = await self.graph_rag.query(
                        question=question,
                        max_context_nodes=self.default_params["max_graph_nodes"],
                        max_traversal_depth=2,
                        use_text_to_cypher=True
                    )
                    
                    if graph_result.success:
                        reasoning_steps.append(f"Graph search found {graph_result.graph_context.total_nodes} entities, "
                                             f"{graph_result.graph_context.total_relationships} relationships")
                    else:
                        reasoning_steps.append(f"Graph search failed: {graph_result.error_message}")
                        
                except Exception as e:
                    reasoning_steps.append(f"Graph search error: {str(e)}")
                    logger.warning(f"Graph search failed: {str(e)}")
            
            # Perform vector-based retrieval
            if enable_vector_search:
                reasoning_steps.append("Performing vector-based retrieval")
                try:
                    vector_result = await self.vector_search.search_documents(
                        query=question,
                        limit=self.default_params["max_vector_results"],
                        similarity_threshold=similarity_threshold,
                        include_chunks=True,
                        include_entities=True
                    )
                    
                    if vector_result.success:
                        reasoning_steps.append(f"Vector search found {len(vector_result.documents)} documents, "
                                             f"{len(vector_result.chunks)} chunks, {len(vector_result.entities)} entities")
                    else:
                        reasoning_steps.append(f"Vector search failed: {vector_result.error_message}")
                        
                except Exception as e:
                    reasoning_steps.append(f"Vector search error: {str(e)}")
                    logger.warning(f"Vector search failed: {str(e)}")
            
            # Check if we have any results
            has_graph_results = graph_result and graph_result.success and (
                graph_result.graph_context.entities or graph_result.graph_context.query_results
            )
            has_vector_results = vector_result and vector_result.success and (
                vector_result.documents or vector_result.chunks or vector_result.entities
            )
            
            if not has_graph_results and not has_vector_results:
                reasoning_steps.append("No relevant information found in either graph or vector search")
                return HybridRAGResult(
                    success=False,
                    query=question,
                    error_message="No relevant information found in knowledge base",
                    graph_result=graph_result,
                    vector_result=vector_result,
                    processing_time_seconds=time.time() - start_time,
                    reasoning_steps=reasoning_steps
                )
            
            # Fuse results
            reasoning_steps.append("Fusing graph and vector search results")
            fusion_start = time.time()
            
            # Extract sources for fusion
            graph_sources = []
            if has_graph_results:
                graph_sources = graph_result.sources or []
                # Also add entities and relationships as sources
                if graph_result.graph_context.entities:
                    for entity in graph_result.graph_context.entities:
                        graph_sources.append({
                            "id": entity.get("id", ""),
                            "entity": entity.get("name", "Unknown"),
                            "confidence": entity.get("confidence", 0.5),
                            "type": "entity"
                        })
            
            vector_documents = vector_result.documents if has_vector_results else []
            vector_chunks = vector_result.chunks if has_vector_results else []
            vector_entities = vector_result.entities if has_vector_results else []
            
            # Perform fusion
            fused_sources, dedup_stats = self.result_fusion.fuse_results(
                graph_sources=graph_sources,
                vector_documents=vector_documents,
                vector_chunks=vector_chunks,
                vector_entities=vector_entities,
                weights=fusion_weights or self.default_params["fusion_weights"],
                max_results=max_results
            )
            
            fusion_time = time.time() - fusion_start
            
            # Build hybrid context
            hybrid_context = HybridContext(
                graph_context=graph_result.graph_context if has_graph_results else None,
                vector_context=vector_result.vector_context if has_vector_results else None,
                fusion_method=self.result_fusion.fusion_method,
                fusion_weights=fusion_weights or self.default_params["fusion_weights"],
                combined_sources=fused_sources,
                deduplication_stats=dedup_stats,
                total_unique_sources=len(fused_sources),
                fusion_time_seconds=fusion_time
            )
            
            # Generate answer using the best available method
            reasoning_steps.append("Generating answer from hybrid context")
            if has_graph_results and graph_result.answer:
                # Use graph RAG answer as base
                answer = graph_result.answer
            else:
                # Generate simple answer from available context
                answer = self._generate_simple_answer(question, fused_sources)
            
            # Calculate confidence
            confidence = self._calculate_hybrid_confidence(
                graph_result, vector_result, len(fused_sources)
            )
            
            processing_time = time.time() - start_time
            reasoning_steps.append(f"Hybrid RAG processing completed in {processing_time:.2f} seconds")
            
            logger.info(f"Hybrid RAG query completed: {question}")
            
            return HybridRAGResult(
                success=True,
                query=question,
                answer=answer,
                confidence=confidence,
                hybrid_context=hybrid_context,
                graph_result=graph_result,
                vector_result=vector_result,
                sources=fused_sources,
                processing_time_seconds=processing_time,
                reasoning_steps=reasoning_steps,
                fusion_metadata={
                    "fusion_method": self.result_fusion.fusion_method.value,
                    "fusion_time": fusion_time,
                    "graph_enabled": enable_graph_search,
                    "vector_enabled": enable_vector_search,
                    "has_graph_results": has_graph_results,
                    "has_vector_results": has_vector_results
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Hybrid RAG query failed: {str(e)}"
            reasoning_steps.append(error_msg)
            
            logger.error(f"Error in Hybrid RAG query '{question}': {str(e)}")
            
            return HybridRAGResult(
                success=False,
                query=question,
                error_message=error_msg,
                graph_result=graph_result,
                vector_result=vector_result,
                processing_time_seconds=processing_time,
                reasoning_steps=reasoning_steps
            )
    
    def _generate_simple_answer(self, question: str, sources: List[Dict[str, Any]]) -> str:
        """Generate a simple answer from fused sources."""
        if not sources:
            return "No relevant information found to answer the question."
        
        # Extract key information from top sources
        top_sources = sources[:5]
        answer_parts = [f"Based on the available information:"]
        
        for i, source in enumerate(top_sources, 1):
            title = source.get("title", "Unknown")
            content = source.get("content", "")[:100]
            source_type = source.get("source_type", "unknown")
            
            answer_parts.append(f"{i}. From {source_type}: {title} - {content}...")
        
        return "\n".join(answer_parts)
    
    def _calculate_hybrid_confidence(
        self,
        graph_result: Optional[GraphRAGResult],
        vector_result: Optional[VectorSearchResult],
        num_fused_sources: int
    ) -> float:
        """Calculate confidence score for hybrid RAG result."""
        confidence = 0.5  # Base confidence
        
        # Boost from graph results
        if graph_result and graph_result.success:
            confidence += graph_result.confidence * 0.4
        
        # Boost from vector results
        if vector_result and vector_result.success:
            # Calculate average vector score
            all_scores = []
            for doc in vector_result.documents:
                all_scores.append(doc.get("score", 0.0))
            for chunk in vector_result.chunks:
                all_scores.append(chunk.get("score", 0.0))
            for entity in vector_result.entities:
                all_scores.append(entity.get("score", 0.0))
            
            if all_scores:
                avg_vector_score = sum(all_scores) / len(all_scores)
                confidence += avg_vector_score * 0.3
        
        # Boost from fusion quality
        if num_fused_sources > 5:
            confidence += 0.1
        elif num_fused_sources > 10:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of hybrid RAG system."""
        health_status = {
            "overall_healthy": False,
            "components": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Check initialization
            health_status["components"]["initialization"] = {
                "healthy": self._initialized,
                "message": "Initialized" if self._initialized else "Not initialized"
            }
            
            if not self._initialized:
                return health_status
            
            # Check vector search
            vector_health = self.vector_search.health_check()
            health_status["components"]["vector_search"] = {
                "healthy": vector_health["initialized"] and vector_health["embedding_pipeline_healthy"],
                "details": vector_health
            }
            
            # Check graph availability
            try:
                graph_stats = await self.graph_manager.get_graph_statistics()
                graph_healthy = graph_stats.get("success", False)
            except:
                graph_healthy = False
            
            health_status["components"]["graph_search"] = {
                "healthy": graph_healthy,
                "message": "Graph accessible" if graph_healthy else "Graph not accessible"
            }
            
            # Overall health
            component_health = [comp["healthy"] for comp in health_status["components"].values()]
            health_status["overall_healthy"] = any(component_health)  # At least one method should work
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error in hybrid RAG health check: {str(e)}")
            health_status["error"] = str(e)
            return health_status

def create_hybrid_rag_engine(
    graph_manager: GraphManager,
    embedding_pipeline: Optional[EmbeddingPipeline] = None,
    fusion_method: FusionMethod = FusionMethod.WEIGHTED_AVERAGE
) -> HybridRAGEngine:
    """
    Factory function to create a HybridRAGEngine instance.
    
    Args:
        graph_manager: Initialized graph manager
        embedding_pipeline: Optional embedding pipeline
        fusion_method: Method for fusing results
        
    Returns:
        HybridRAGEngine instance
    """
    return HybridRAGEngine(
        graph_manager=graph_manager,
        embedding_pipeline=embedding_pipeline,
        fusion_method=fusion_method
    )
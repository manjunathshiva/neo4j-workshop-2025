"""
Graph RAG Manager - High-level interface for Graph RAG operations.
Provides a unified interface for graph-based retrieval and generation.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    from .graph_rag import GraphRAGPipeline, GraphRAGResult
    from .text_to_cypher import TextToCypherGenerator
    from ..graph.graph_manager import GraphManager
    from ..connections import get_database_manager
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from rag.graph_rag import GraphRAGPipeline, GraphRAGResult
    from rag.text_to_cypher import TextToCypherGenerator
    from graph.graph_manager import GraphManager
    from connections import get_database_manager

logger = logging.getLogger(__name__)

class GraphRAGManager:
    """
    High-level manager for Graph RAG operations.
    Provides a unified interface for graph-based question answering.
    """
    
    def __init__(self):
        """Initialize Graph RAG Manager."""
        self.db_manager = get_database_manager()
        self.graph_manager = None
        self.rag_pipeline = None
        self.cypher_generator = None
        self._initialized = False
        
        logger.info("GraphRAGManager created")
    
    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize the Graph RAG Manager and all components.
        
        Returns:
            Dictionary with initialization results
        """
        try:
            logger.info("Initializing GraphRAGManager...")
            
            # Initialize database connections first
            db_init_result = await self.db_manager.initialize(validate_only=False)
            
            if not db_init_result["overall_success"]:
                return {
                    "success": False,
                    "error": f"Database initialization failed: {db_init_result.get('errors', 'Unknown error')}",
                    "component": "DatabaseManager"
                }
            
            # Initialize graph manager
            self.graph_manager = GraphManager()
            graph_init_result = await self.graph_manager.initialize()
            
            if not graph_init_result["success"]:
                return {
                    "success": False,
                    "error": f"Graph manager initialization failed: {graph_init_result.get('error', 'Unknown error')}",
                    "component": "GraphManager"
                }
            
            # Initialize RAG pipeline
            self.rag_pipeline = GraphRAGPipeline(self.graph_manager)
            
            # Initialize Cypher generator
            self.cypher_generator = TextToCypherGenerator()
            
            self._initialized = True
            
            logger.info("GraphRAGManager initialized successfully")
            
            return {
                "success": True,
                "message": "Graph RAG Manager initialized successfully",
                "components": {
                    "graph_manager": graph_init_result,
                    "rag_pipeline": self.rag_pipeline.get_pipeline_info(),
                    "cypher_generator": self.cypher_generator.get_model_info()
                }
            }
            
        except Exception as e:
            logger.error(f"Error initializing GraphRAGManager: {str(e)}")
            return {
                "success": False,
                "error": f"GraphRAGManager initialization failed: {str(e)}",
                "component": "GraphRAGManager"
            }
    
    async def query(
        self,
        question: str,
        max_context_nodes: int = 50,
        max_traversal_depth: int = 2,
        use_text_to_cypher: bool = True,
        include_reasoning: bool = True
    ) -> Dict[str, Any]:
        """
        Process a natural language question using Graph RAG.
        
        Args:
            question: Natural language question
            max_context_nodes: Maximum nodes to include in context
            max_traversal_depth: Maximum graph traversal depth
            use_text_to_cypher: Whether to use Text-to-Cypher generation
            include_reasoning: Whether to include reasoning steps in response
            
        Returns:
            Dictionary with query results and metadata
        """
        if not self._initialized:
            return {
                "success": False,
                "error": "GraphRAGManager not initialized",
                "query": question
            }
        
        try:
            logger.info(f"Processing Graph RAG query: {question}")
            
            # Process query using RAG pipeline
            rag_result = await self.rag_pipeline.query(
                question=question,
                max_context_nodes=max_context_nodes,
                max_traversal_depth=max_traversal_depth,
                use_text_to_cypher=use_text_to_cypher
            )
            
            # Format response
            response = {
                "success": rag_result.success,
                "query": rag_result.query,
                "answer": rag_result.answer,
                "method": "graph_rag",
                "confidence": rag_result.confidence,
                "processing_time_seconds": rag_result.processing_time_seconds,
                "sources": rag_result.sources
            }
            
            if not rag_result.success:
                response["error"] = rag_result.error_message
            
            # Add context information
            if rag_result.graph_context:
                response["context"] = {
                    "total_entities": rag_result.graph_context.total_nodes,
                    "total_relationships": rag_result.graph_context.total_relationships,
                    "cypher_query": rag_result.graph_context.cypher_query,
                    "traversal_depth": rag_result.graph_context.traversal_depth
                }
                
                # Include detailed context if requested
                if include_reasoning:
                    response["context"]["entities"] = rag_result.graph_context.entities[:10]  # Limit for response size
                    response["context"]["relationships"] = rag_result.graph_context.relationships[:10]
            
            # Add reasoning steps if requested
            if include_reasoning and rag_result.reasoning_steps:
                response["reasoning_steps"] = rag_result.reasoning_steps
            
            # Add Cypher generation details if available
            if rag_result.cypher_generation:
                response["cypher_generation"] = {
                    "success": rag_result.cypher_generation.success,
                    "query": rag_result.cypher_generation.cypher_query,
                    "confidence": rag_result.cypher_generation.confidence,
                    "model_used": rag_result.cypher_generation.model_used
                }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing Graph RAG query '{question}': {str(e)}")
            return {
                "success": False,
                "error": f"Query processing failed: {str(e)}",
                "query": question,
                "method": "graph_rag"
            }
    
    async def generate_cypher(
        self,
        question: str,
        include_explanation: bool = False
    ) -> Dict[str, Any]:
        """
        Generate Cypher query from natural language question.
        
        Args:
            question: Natural language question
            include_explanation: Whether to include query explanation
            
        Returns:
            Dictionary with Cypher generation results
        """
        if not self._initialized or not self.cypher_generator:
            return {
                "success": False,
                "error": "Cypher generator not initialized",
                "query": question
            }
        
        try:
            # Generate Cypher query
            result = self.cypher_generator.generate_cypher(question)
            
            response = {
                "success": result.success,
                "query": question,
                "cypher_query": result.cypher_query,
                "parameters": result.parameters,
                "confidence": result.confidence,
                "model_used": result.model_used,
                "processing_time_seconds": result.processing_time_seconds
            }
            
            if not result.success:
                response["error"] = result.error_message
            else:
                response["reasoning"] = result.reasoning
                
                # Add explanation if requested
                if include_explanation and result.cypher_query:
                    explanation = self.cypher_generator.explain_cypher(result.cypher_query)
                    response["explanation"] = explanation
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating Cypher for question '{question}': {str(e)}")
            return {
                "success": False,
                "error": f"Cypher generation failed: {str(e)}",
                "query": question
            }
    
    async def execute_cypher(
        self,
        cypher_query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a Cypher query directly against the graph.
        
        Args:
            cypher_query: Cypher query to execute
            parameters: Optional query parameters
            
        Returns:
            Dictionary with query execution results
        """
        if not self._initialized or not self.graph_manager:
            return {
                "success": False,
                "error": "Graph manager not initialized",
                "query": cypher_query
            }
        
        try:
            result = await self.graph_manager.query_graph(cypher_query, parameters)
            
            return {
                "success": result["success"],
                "cypher_query": cypher_query,
                "parameters": parameters or {},
                "records": result.get("records", []),
                "record_count": result.get("record_count", 0),
                "error": result.get("error")
            }
            
        except Exception as e:
            logger.error(f"Error executing Cypher query: {str(e)}")
            return {
                "success": False,
                "error": f"Query execution failed: {str(e)}",
                "cypher_query": cypher_query,
                "parameters": parameters or {}
            }
    
    async def get_entity_context(
        self,
        entity_name: str,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Get context information for a specific entity.
        
        Args:
            entity_name: Name of the entity to get context for
            max_depth: Maximum traversal depth
            
        Returns:
            Dictionary with entity context
        """
        if not self._initialized or not self.graph_manager:
            return {
                "success": False,
                "error": "Graph manager not initialized",
                "entity_name": entity_name
            }
        
        try:
            # Search for the entity
            search_result = await self.graph_manager.search_entities(entity_name, limit=5)
            
            if not search_result["success"] or not search_result["records"]:
                return {
                    "success": False,
                    "error": f"Entity '{entity_name}' not found",
                    "entity_name": entity_name
                }
            
            # Get the first matching entity
            entity_record = search_result["records"][0]
            entity = entity_record.get("entity", {})
            entity_id = entity.get("id")
            
            if not entity_id:
                return {
                    "success": False,
                    "error": "Entity ID not found",
                    "entity_name": entity_name
                }
            
            # Get entity neighborhood
            neighborhood_result = await self.graph_manager.get_entity_neighborhood(entity_id)
            
            if not neighborhood_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to get entity neighborhood: {neighborhood_result.get('error', 'Unknown error')}",
                    "entity_name": entity_name
                }
            
            # Get traversal information
            traversal_result = await self.graph_manager.traverse_from_entity(
                entity_id=entity_id,
                max_depth=max_depth
            )
            
            return {
                "success": True,
                "entity_name": entity_name,
                "entity": entity,
                "neighborhood": neighborhood_result["records"],
                "traversal": traversal_result.get("traversal_result", {}),
                "context_summary": {
                    "direct_connections": len(neighborhood_result.get("records", [])),
                    "traversal_depth": max_depth,
                    "total_connected_entities": len(traversal_result.get("traversal_result", {}).get("all_nodes", []))
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting entity context for '{entity_name}': {str(e)}")
            return {
                "success": False,
                "error": f"Failed to get entity context: {str(e)}",
                "entity_name": entity_name
            }
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the knowledge graph.
        
        Returns:
            Dictionary with graph statistics
        """
        if not self._initialized or not self.graph_manager:
            return {
                "success": False,
                "error": "Graph manager not initialized"
            }
        
        try:
            stats_result = await self.graph_manager.get_graph_statistics()
            
            if stats_result["success"]:
                return {
                    "success": True,
                    "statistics": stats_result,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": stats_result.get("error", "Failed to get statistics")
                }
                
        except Exception as e:
            logger.error(f"Error getting graph statistics: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to get graph statistics: {str(e)}"
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the Graph RAG Manager.
        
        Returns:
            Dictionary with status information
        """
        return {
            "initialized": self._initialized,
            "components": {
                "graph_manager": self.graph_manager is not None,
                "rag_pipeline": self.rag_pipeline is not None,
                "cypher_generator": self.cypher_generator is not None
            },
            "database_status": self.db_manager.get_connection_summary() if self.db_manager else {},
            "model_info": self.cypher_generator.get_model_info() if self.cypher_generator else {}
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of Graph RAG system.
        
        Returns:
            Dictionary with health check results
        """
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
            
            # Check database connections
            db_validation = self.db_manager.validate_connections()
            health_status["components"]["database"] = {
                "healthy": db_validation["overall_valid"],
                "neo4j": db_validation["neo4j"]["valid"],
                "message": "Database connections healthy" if db_validation["overall_valid"] else "Database connection issues"
            }
            
            # Check LLM availability
            llm_available = self.cypher_generator._llm is not None
            health_status["components"]["llm"] = {
                "healthy": llm_available,
                "model": self.cypher_generator._model_name if llm_available else "Not available",
                "message": "LLM available" if llm_available else "LLM not available"
            }
            
            # Test basic graph query
            try:
                test_result = await self.graph_manager.query_graph("MATCH (n) RETURN count(n) as node_count LIMIT 1")
                graph_query_healthy = test_result["success"]
                node_count = test_result["records"][0]["node_count"] if test_result.get("records") else 0
            except:
                graph_query_healthy = False
                node_count = 0
            
            health_status["components"]["graph_query"] = {
                "healthy": graph_query_healthy,
                "node_count": node_count,
                "message": f"Graph accessible ({node_count} nodes)" if graph_query_healthy else "Graph query failed"
            }
            
            # Overall health
            component_health = [comp["healthy"] for comp in health_status["components"].values()]
            health_status["overall_healthy"] = all(component_health)
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error in health check: {str(e)}")
            health_status["error"] = str(e)
            return health_status
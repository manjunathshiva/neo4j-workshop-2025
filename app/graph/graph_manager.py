"""
Graph manager that integrates knowledge extraction with Neo4j graph building.
Provides high-level interface for building and querying the knowledge graph.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    from .neo4j_builder import Neo4jGraphBuilder
    from .neo4j_schema import Neo4jSchema
    from .cypher_builder import CommonQueries, CypherQueryBuilder
    from ..extractors.knowledge_graph import KnowledgeGraph
    from ..extractors.entity_extractor import Entity, EntityType
    from ..extractors.relationship_extractor import Relationship
    from ..connections import get_database_manager
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from graph.neo4j_builder import Neo4jGraphBuilder
    from graph.neo4j_schema import Neo4jSchema
    from graph.cypher_builder import CommonQueries, CypherQueryBuilder
    from extractors.knowledge_graph import KnowledgeGraph
    from extractors.entity_extractor import Entity, EntityType
    from extractors.relationship_extractor import Relationship
    from connections import get_database_manager

logger = logging.getLogger(__name__)

class GraphManager:
    """
    High-level manager for the Neo4j knowledge graph.
    Integrates knowledge extraction with graph building and querying.
    """
    
    def __init__(self):
        """Initialize graph manager."""
        self.neo4j_builder = Neo4jGraphBuilder()
        self.schema = Neo4jSchema()
        self.knowledge_graph = KnowledgeGraph()
        self.db_manager = get_database_manager()
        
        logger.info("GraphManager initialized")
    
    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize the graph manager and Neo4j schema.
        
        Returns:
            Dictionary with initialization results
        """
        logger.info("Initializing GraphManager...")
        
        try:
            # Initialize Neo4j schema
            schema_result = await self.neo4j_builder.initialize_schema()
            
            if schema_result["success"]:
                logger.info("GraphManager initialized successfully")
                return {
                    "success": True,
                    "message": "GraphManager initialized successfully",
                    "schema_initialization": schema_result
                }
            else:
                logger.warning("Schema initialization had issues")
                return {
                    "success": True,  # Still usable even with schema warnings
                    "message": "GraphManager initialized with warnings",
                    "schema_initialization": schema_result,
                    "warnings": schema_result.get("errors", [])
                }
                
        except Exception as e:
            logger.error(f"Error initializing GraphManager: {str(e)}")
            return {
                "success": False,
                "error": f"GraphManager initialization failed: {str(e)}"
            }
    
    async def process_document_to_graph(
        self,
        document_id: str,
        title: str,
        content: str,
        file_type: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a document and add it to the knowledge graph.
        
        Args:
            document_id: Unique document identifier
            title: Document title
            content: Document content
            file_type: File type (PDF, TXT, etc.)
            metadata: Additional metadata
            
        Returns:
            Dictionary with processing results
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing document to graph: {document_id}")
            
            # Step 1: Create document node in Neo4j
            logger.info(f"Creating document node for {document_id}")
            doc_result = await self.neo4j_builder.create_document_node(
                document_id=document_id,
                title=title,
                content=content,
                file_type=file_type,
                metadata=metadata
            )
            
            if doc_result.success:
                logger.info(f"Document node {doc_result.operation} successfully: {document_id}")
            
            if not doc_result.success:
                logger.error(f"Document node creation failed for {document_id}: {doc_result.error}")
                return {
                    "success": False,
                    "error": f"Failed to create document node: {doc_result.error}",
                    "document_id": document_id
                }
            
            # Step 2: Extract knowledge from document content
            knowledge_result = self.knowledge_graph.extract_knowledge_from_text(
                text=content,
                document_id=document_id,
                chunk_id=f"{document_id}_main"
            )
            
            if not knowledge_result["success"]:
                logger.error(f"Knowledge extraction failed for {document_id}: {knowledge_result['error_message']}")
                return {
                    "success": False,
                    "error": f"Knowledge extraction failed: {knowledge_result['error_message']}",
                    "document_id": document_id,
                    "document_created": doc_result.operation == "created"
                }
            
            # Step 3: Add extracted entities to Neo4j
            entities_added = 0
            entities_updated = 0
            
            for entity in self.knowledge_graph._entities.values():
                if entity.source_document_id == document_id:
                    entity_result = await self.neo4j_builder.create_entity_node(entity)
                    if entity_result.success:
                        if entity_result.operation == "created":
                            entities_added += 1
                        else:
                            entities_updated += 1
                        
                        # Create CONTAINS relationship between document and entity
                        await self.neo4j_builder.create_document_entity_relationship(
                            document_id=document_id,
                            entity_id=entity.id,
                            confidence=entity.confidence,
                            extraction_method=entity.metadata.get("extraction_method", "llm")
                        )
            
            # Step 4: Add extracted relationships to Neo4j
            relationships_added = 0
            relationships_updated = 0
            
            for relationship in self.knowledge_graph._relationships.values():
                if relationship.source_document_id == document_id:
                    rel_result = await self.neo4j_builder.create_entity_relationship(relationship)
                    if rel_result.success:
                        if rel_result.operation == "created":
                            relationships_added += 1
                        else:
                            relationships_updated += 1
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "success": True,
                "document_id": document_id,
                "document_operation": doc_result.operation,
                "entities_added": entities_added,
                "entities_updated": entities_updated,
                "relationships_added": relationships_added,
                "relationships_updated": relationships_updated,
                "total_entities_extracted": knowledge_result["entities_extracted"],
                "total_relationships_extracted": knowledge_result["relationships_extracted"],
                "processing_time_seconds": processing_time,
                "knowledge_extraction": knowledge_result
            }
            
            logger.info(f"Document processed successfully: {document_id} "
                       f"({entities_added + entities_updated} entities, "
                       f"{relationships_added + relationships_updated} relationships)")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error processing document {document_id}: {str(e)}")
            
            return {
                "success": False,
                "error": f"Document processing failed: {str(e)}",
                "document_id": document_id,
                "processing_time_seconds": processing_time
            }
    
    async def query_graph(self, cypher_query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a Cypher query against the graph.
        
        Args:
            cypher_query: Cypher query string
            parameters: Optional query parameters
            
        Returns:
            Dictionary with query results
        """
        try:
            if not parameters:
                parameters = {}
            
            async with self.db_manager.get_neo4j_session() as session:
                result = session.run(cypher_query, parameters)
                
                records = []
                for record in result:
                    # Convert record to dictionary
                    record_dict = {}
                    for key in record.keys():
                        value = record[key]
                        # Handle Neo4j node/relationship objects
                        if hasattr(value, '__dict__'):
                            record_dict[key] = dict(value)
                        else:
                            record_dict[key] = value
                    records.append(record_dict)
                
                return {
                    "success": True,
                    "records": records,
                    "record_count": len(records),
                    "query": cypher_query,
                    "parameters": parameters
                }
                
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return {
                "success": False,
                "error": f"Query execution failed: {str(e)}",
                "query": cypher_query,
                "parameters": parameters
            }
    
    async def search_entities(self, search_term: str, limit: int = 20) -> Dict[str, Any]:
        """
        Search entities using full-text search.
        
        Args:
            search_term: Search term
            limit: Maximum number of results
            
        Returns:
            Dictionary with search results
        """
        try:
            query = CommonQueries.search_entities_by_text(search_term, limit)
            return await self.query_graph(query.query, query.parameters)
            
        except Exception as e:
            logger.error(f"Error searching entities: {str(e)}")
            return {
                "success": False,
                "error": f"Entity search failed: {str(e)}",
                "search_term": search_term
            }
    
    async def get_entity_neighborhood(
        self,
        entity_id: str,
        relationship_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get the neighborhood of an entity.
        
        Args:
            entity_id: Entity ID
            relationship_types: Optional relationship types to filter by
            
        Returns:
            Dictionary with neighborhood data
        """
        try:
            query = CommonQueries.get_entity_neighborhood(entity_id, relationship_types)
            return await self.query_graph(query.query, query.parameters)
            
        except Exception as e:
            logger.error(f"Error getting entity neighborhood: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to get entity neighborhood: {str(e)}",
                "entity_id": entity_id
            }
    
    async def find_path_between_entities(
        self,
        source_id: str,
        target_id: str
    ) -> Dict[str, Any]:
        """
        Find shortest path between two entities.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            
        Returns:
            Dictionary with path information
        """
        try:
            query = CommonQueries.find_shortest_path_between_entities(source_id, target_id)
            return await self.query_graph(query.query, query.parameters)
            
        except Exception as e:
            logger.error(f"Error finding path between entities: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to find path: {str(e)}",
                "source_id": source_id,
                "target_id": target_id
            }
    
    async def get_document_entities(self, document_id: str) -> Dict[str, Any]:
        """
        Get all entities in a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Dictionary with document entities
        """
        try:
            query = CommonQueries.find_entities_in_document(document_id)
            return await self.query_graph(query.query, query.parameters)
            
        except Exception as e:
            logger.error(f"Error getting document entities: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to get document entities: {str(e)}",
                "document_id": document_id
            }
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive graph statistics.
        
        Returns:
            Dictionary with graph statistics
        """
        try:
            # Get Neo4j statistics
            neo4j_stats = await self.neo4j_builder.get_graph_statistics()
            
            # Get knowledge graph statistics
            kg_stats = self.knowledge_graph.get_graph_statistics()
            
            return {
                "success": True,
                "neo4j_statistics": neo4j_stats,
                "knowledge_graph_statistics": kg_stats.__dict__,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting graph statistics: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to get statistics: {str(e)}"
            }
    
    async def get_most_connected_entities(self, limit: int = 10) -> Dict[str, Any]:
        """
        Get the most connected entities in the graph.
        
        Args:
            limit: Maximum number of entities to return
            
        Returns:
            Dictionary with most connected entities
        """
        try:
            query = CommonQueries.get_most_connected_entities(limit)
            return await self.query_graph(query.query, query.parameters)
            
        except Exception as e:
            logger.error(f"Error getting most connected entities: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to get most connected entities: {str(e)}"
            }
    
    async def traverse_from_entity(
        self,
        entity_id: str,
        max_depth: int = 2,
        relationship_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Traverse the graph starting from an entity.
        
        Args:
            entity_id: Starting entity ID
            max_depth: Maximum traversal depth
            relationship_types: Optional relationship types to follow
            
        Returns:
            Dictionary with traversal results
        """
        try:
            result = await self.neo4j_builder.traverse_graph(
                start_node_id=entity_id,
                max_depth=max_depth,
                relationship_types=relationship_types,
                node_labels=["Entity"]
            )
            
            return {
                "success": True,
                "traversal_result": result
            }
            
        except Exception as e:
            logger.error(f"Error traversing from entity: {str(e)}")
            return {
                "success": False,
                "error": f"Graph traversal failed: {str(e)}",
                "entity_id": entity_id
            }
    
    async def get_subgraph_for_entities(
        self,
        entity_ids: List[str],
        include_intermediate: bool = True
    ) -> Dict[str, Any]:
        """
        Get a subgraph containing specified entities.
        
        Args:
            entity_ids: List of entity IDs
            include_intermediate: Whether to include intermediate nodes
            
        Returns:
            Dictionary with subgraph data
        """
        try:
            result = await self.neo4j_builder.get_connected_subgraph(
                node_ids=entity_ids,
                include_intermediate=include_intermediate
            )
            
            return {
                "success": True,
                "subgraph": result
            }
            
        except Exception as e:
            logger.error(f"Error getting subgraph: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to get subgraph: {str(e)}",
                "entity_ids": entity_ids
            }
    
    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get schema information for the graph.
        
        Returns:
            Dictionary with schema information
        """
        return {
            "node_labels": self.schema.get_all_node_labels(),
            "relationship_types": self.schema.get_all_relationship_types(),
            "schema_summary": self.schema.get_schema_summary()
        }
    
    async def clear_graph(self) -> Dict[str, Any]:
        """
        Clear all data from the graph (use with caution).
        
        Returns:
            Dictionary with operation result
        """
        try:
            async with self.db_manager.get_neo4j_session() as session:
                # Delete all nodes and relationships
                await session.run("MATCH (n) DETACH DELETE n")
                
                # Clear in-memory knowledge graph
                self.knowledge_graph.clear_graph()
                
                # Reset operation stats
                self.neo4j_builder.reset_operation_stats()
                
                logger.warning("Graph cleared - all data deleted")
                
                return {
                    "success": True,
                    "message": "Graph cleared successfully"
                }
                
        except Exception as e:
            logger.error(f"Error clearing graph: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to clear graph: {str(e)}"
            }
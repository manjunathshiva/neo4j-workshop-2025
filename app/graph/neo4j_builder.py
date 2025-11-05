"""
Neo4j graph builder for creating and managing nodes and relationships.
Implements node creation, update operations, and graph indexing.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from neo4j import Driver, Session
from neo4j.exceptions import Neo4jError, ConstraintError

try:
    from .neo4j_schema import Neo4jSchema, NodeType, RelationshipType
    from ..extractors.entity_extractor import Entity, EntityType
    from ..extractors.relationship_extractor import Relationship
    from ..connections import get_database_manager
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from graph.neo4j_schema import Neo4jSchema, NodeType, RelationshipType
    from extractors.entity_extractor import Entity, EntityType
    from extractors.relationship_extractor import Relationship
    from connections import get_database_manager

logger = logging.getLogger(__name__)

@dataclass
class NodeOperationResult:
    """Result of a node operation."""
    success: bool
    node_id: Optional[str] = None
    operation: str = ""  # "created", "updated", "found"
    message: str = ""
    error: Optional[str] = None
    properties: Dict[str, Any] = None

@dataclass
class GraphOperationStats:
    """Statistics for graph operations."""
    nodes_created: int = 0
    nodes_updated: int = 0
    nodes_found: int = 0
    relationships_created: int = 0
    relationships_updated: int = 0
    errors: int = 0
    processing_time_seconds: float = 0.0

class Neo4jGraphBuilder:
    """
    Neo4j graph builder for the knowledge graph system.
    Handles node creation, updates, and graph structure management.
    """
    
    def __init__(self):
        """Initialize Neo4j graph builder."""
        self.schema = Neo4jSchema()
        self.db_manager = get_database_manager()
        
        # Operation tracking
        self._operation_stats = GraphOperationStats()
        
        logger.info("Neo4jGraphBuilder initialized")
    
    def _flatten_metadata(self, metadata: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """
        Flatten nested metadata for Neo4j storage.
        
        Args:
            metadata: Nested metadata dictionary
            prefix: Prefix for flattened keys
            
        Returns:
            Flattened dictionary with primitive values only
        """
        flattened = {}
        
        for key, value in metadata.items():
            new_key = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                flattened.update(self._flatten_metadata(value, f"{new_key}_"))
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                if value and isinstance(value[0], (str, int, float)):
                    flattened[new_key] = ",".join(str(v) for v in value)
                else:
                    flattened[f"{new_key}_count"] = len(value)
            elif isinstance(value, (str, int, float, bool)):
                # Keep primitive types as-is
                flattened[new_key] = value
            else:
                # Convert other types to string
                flattened[new_key] = str(value)
        
        return flattened
    
    async def initialize_schema(self) -> Dict[str, Any]:
        """
        Initialize Neo4j schema by creating indexes and constraints.
        
        Returns:
            Dictionary with initialization results
        """
        start_time = datetime.now()
        results = {
            "success": False,
            "indexes_created": 0,
            "constraints_created": 0,
            "errors": [],
            "processing_time_seconds": 0.0
        }
        
        try:
            async with self.db_manager.get_neo4j_session() as session:
                # Create constraints first (they may create indexes automatically)
                logger.info("Creating Neo4j constraints...")
                for constraint_query in self.schema.constraints:
                    try:
                        session.run(constraint_query)
                        results["constraints_created"] += 1
                        logger.debug(f"Created constraint: {constraint_query}")
                    except ConstraintError as e:
                        if "already exists" in str(e).lower():
                            logger.debug(f"Constraint already exists: {constraint_query}")
                        else:
                            logger.warning(f"Error creating constraint: {str(e)}")
                            results["errors"].append(f"Constraint error: {str(e)}")
                    except Exception as e:
                        logger.warning(f"Error creating constraint: {str(e)}")
                        results["errors"].append(f"Constraint error: {str(e)}")
                
                # Create indexes
                logger.info("Creating Neo4j indexes...")
                for index_query in self.schema.indexes:
                    try:
                        session.run(index_query)
                        results["indexes_created"] += 1
                        logger.debug(f"Created index: {index_query}")
                    except Neo4jError as e:
                        if "already exists" in str(e).lower():
                            logger.debug(f"Index already exists: {index_query}")
                        else:
                            logger.warning(f"Error creating index: {str(e)}")
                            results["errors"].append(f"Index error: {str(e)}")
                    except Exception as e:
                        logger.warning(f"Error creating index: {str(e)}")
                        results["errors"].append(f"Index error: {str(e)}")
                
                results["success"] = True
                processing_time = (datetime.now() - start_time).total_seconds()
                results["processing_time_seconds"] = processing_time
                
                logger.info(f"Schema initialization completed: {results['constraints_created']} constraints, {results['indexes_created']} indexes")
                
        except Exception as e:
            logger.error(f"Error initializing Neo4j schema: {str(e)}")
            results["errors"].append(f"Schema initialization failed: {str(e)}")
            results["processing_time_seconds"] = (datetime.now() - start_time).total_seconds()
        
        return results
    
    async def create_document_node(
        self,
        document_id: str,
        title: str,
        content: str,
        file_type: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> NodeOperationResult:
        """
        Create or update a document node in Neo4j.
        
        Args:
            document_id: Unique document identifier
            title: Document title
            content: Document content
            file_type: File type (PDF, TXT, etc.)
            metadata: Additional metadata
            
        Returns:
            NodeOperationResult with operation details
        """
        try:
            # Prepare node properties
            properties = {
                "id": document_id,
                "title": title,
                "content": content,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            if file_type:
                properties["file_type"] = file_type
            
            if metadata:
                # Flatten metadata to avoid Neo4j nested object issues
                properties.update(self._flatten_metadata(metadata))
            
            # Validate properties against schema
            validation = self.schema.validate_node_properties(NodeType.DOCUMENT.value, properties)
            if not validation["valid"]:
                return NodeOperationResult(
                    success=False,
                    error=f"Schema validation failed: {validation['error']}"
                )
            
            async with self.db_manager.get_neo4j_session() as session:
                # Use MERGE to create or update
                # Build dynamic SET clauses for all properties
                set_clauses = []
                for key in properties.keys():
                    if key != 'id':  # Skip id as it's used in MERGE
                        set_clauses.append(f"d.{key} = ${key}")
                
                set_clause = ", ".join(set_clauses)
                
                query = f"""
                MERGE (d:Document {{id: $id}})
                ON CREATE SET {set_clause}
                ON MATCH SET {set_clause}
                RETURN d.id as id, 
                       CASE WHEN d.created_at = $created_at THEN 'created' ELSE 'updated' END as operation
                """
                
                result = session.run(query, **properties)
                record = result.single()
                
                if record:
                    operation = record["operation"]
                    if operation == "created":
                        self._operation_stats.nodes_created += 1
                    else:
                        self._operation_stats.nodes_updated += 1
                    
                    logger.info(f"Document node {operation}: {document_id}")
                    
                    return NodeOperationResult(
                        success=True,
                        node_id=record["id"],
                        operation=operation,
                        message=f"Document node {operation} successfully",
                        properties=properties
                    )
                else:
                    return NodeOperationResult(
                        success=False,
                        error="No result returned from Neo4j"
                    )
                    
        except Exception as e:
            logger.error(f"Error creating document node {document_id}: {str(e)}")
            self._operation_stats.errors += 1
            return NodeOperationResult(
                success=False,
                error=f"Failed to create document node: {str(e)}"
            )
    
    async def create_entity_node(self, entity: Entity) -> NodeOperationResult:
        """
        Create or update an entity node in Neo4j.
        
        Args:
            entity: Entity object to create node from
            
        Returns:
            NodeOperationResult with operation details
        """
        try:
            # Prepare node properties
            properties = {
                "id": entity.id,
                "name": entity.name,
                "type": entity.entity_type.value,
                "created_at": entity.created_at.isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Add optional properties with defaults to avoid missing parameter errors
            properties["description"] = entity.description if entity.description else ""
            properties["confidence"] = entity.confidence if entity.confidence > 0 else 0.0
            properties["context"] = entity.context if entity.context else ""
            properties["source_document_id"] = entity.source_document_id if entity.source_document_id else ""
            properties["source_chunk_id"] = entity.source_chunk_id if entity.source_chunk_id else ""
            properties["start_char"] = entity.start_char if entity.start_char >= 0 else -1
            properties["end_char"] = entity.end_char if entity.end_char >= 0 else -1
            
            # Handle metadata
            if entity.metadata:
                properties["extraction_method"] = entity.metadata.get("extraction_method", "")
                properties["llm_provider"] = entity.metadata.get("llm_provider", "")
                # Convert metadata dict to JSON string for Neo4j compatibility
                import json
                properties["metadata_json"] = json.dumps(entity.metadata)
            else:
                properties["extraction_method"] = ""
                properties["llm_provider"] = ""
                properties["metadata_json"] = "{}"
            
            # Validate properties against schema
            validation = self.schema.validate_node_properties(NodeType.ENTITY.value, properties)
            if not validation["valid"]:
                return NodeOperationResult(
                    success=False,
                    error=f"Schema validation failed: {validation['error']}"
                )
            
            async with self.db_manager.get_neo4j_session() as session:
                # Use MERGE to create or update
                query = """
                MERGE (e:Entity {id: $id})
                ON CREATE SET 
                    e.name = $name,
                    e.type = $type,
                    e.description = $description,
                    e.confidence = $confidence,
                    e.context = $context,
                    e.source_document_id = $source_document_id,
                    e.source_chunk_id = $source_chunk_id,
                    e.start_char = $start_char,
                    e.end_char = $end_char,
                    e.extraction_method = $extraction_method,
                    e.llm_provider = $llm_provider,
                    e.metadata_json = $metadata_json,
                    e.created_at = $created_at,
                    e.updated_at = $updated_at
                ON MATCH SET
                    e.name = $name,
                    e.type = $type,
                    e.description = CASE WHEN $description IS NOT NULL THEN $description ELSE e.description END,
                    e.confidence = CASE WHEN $confidence > e.confidence THEN $confidence ELSE e.confidence END,
                    e.context = CASE WHEN $context IS NOT NULL THEN $context ELSE e.context END,
                    e.source_document_id = $source_document_id,
                    e.source_chunk_id = $source_chunk_id,
                    e.start_char = $start_char,
                    e.end_char = $end_char,
                    e.extraction_method = $extraction_method,
                    e.llm_provider = $llm_provider,
                    e.metadata_json = $metadata_json,
                    e.updated_at = $updated_at
                RETURN e.id as id, 
                       CASE WHEN e.created_at = $created_at THEN 'created' ELSE 'updated' END as operation
                """
                
                result = session.run(query, **properties)
                record = result.single()
                
                if record:
                    operation = record["operation"]
                    if operation == "created":
                        self._operation_stats.nodes_created += 1
                    else:
                        self._operation_stats.nodes_updated += 1
                    
                    logger.debug(f"Entity node {operation}: {entity.id} ({entity.name})")
                    
                    return NodeOperationResult(
                        success=True,
                        node_id=record["id"],
                        operation=operation,
                        message=f"Entity node {operation} successfully",
                        properties=properties
                    )
                else:
                    return NodeOperationResult(
                        success=False,
                        error="No result returned from Neo4j"
                    )
                    
        except Exception as e:
            logger.error(f"Error creating entity node {entity.id}: {str(e)}")
            self._operation_stats.errors += 1
            return NodeOperationResult(
                success=False,
                error=f"Failed to create entity node: {str(e)}"
            )
    
    async def create_concept_node(
        self,
        concept_id: str,
        name: str,
        definition: str = "",
        category: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> NodeOperationResult:
        """
        Create or update a concept node in Neo4j.
        
        Args:
            concept_id: Unique concept identifier
            name: Concept name
            definition: Concept definition
            category: Concept category
            metadata: Additional metadata
            
        Returns:
            NodeOperationResult with operation details
        """
        try:
            # Prepare node properties
            properties = {
                "id": concept_id,
                "name": name,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            if definition:
                properties["definition"] = definition
            if category:
                properties["category"] = category
            if metadata:
                properties["metadata"] = metadata
            
            # Validate properties against schema
            validation = self.schema.validate_node_properties(NodeType.CONCEPT.value, properties)
            if not validation["valid"]:
                return NodeOperationResult(
                    success=False,
                    error=f"Schema validation failed: {validation['error']}"
                )
            
            async with self.db_manager.get_neo4j_session() as session:
                # Use MERGE to create or update
                query = """
                MERGE (c:Concept {id: $id})
                ON CREATE SET 
                    c.name = $name,
                    c.definition = $definition,
                    c.category = $category,
                    c.metadata = $metadata,
                    c.created_at = $created_at,
                    c.updated_at = $updated_at
                ON MATCH SET
                    c.name = $name,
                    c.definition = CASE WHEN $definition IS NOT NULL THEN $definition ELSE c.definition END,
                    c.category = CASE WHEN $category IS NOT NULL THEN $category ELSE c.category END,
                    c.metadata = $metadata,
                    c.updated_at = $updated_at
                RETURN c.id as id, 
                       CASE WHEN c.created_at = $created_at THEN 'created' ELSE 'updated' END as operation
                """
                
                result = session.run(query, **properties)
                record = result.single()
                
                if record:
                    operation = record["operation"]
                    if operation == "created":
                        self._operation_stats.nodes_created += 1
                    else:
                        self._operation_stats.nodes_updated += 1
                    
                    logger.info(f"Concept node {operation}: {concept_id}")
                    
                    return NodeOperationResult(
                        success=True,
                        node_id=record["id"],
                        operation=operation,
                        message=f"Concept node {operation} successfully",
                        properties=properties
                    )
                else:
                    return NodeOperationResult(
                        success=False,
                        error="No result returned from Neo4j"
                    )
                    
        except Exception as e:
            logger.error(f"Error creating concept node {concept_id}: {str(e)}")
            self._operation_stats.errors += 1
            return NodeOperationResult(
                success=False,
                error=f"Failed to create concept node: {str(e)}"
            )
    
    async def get_node_by_id(self, node_id: str, node_label: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get a node by its ID.
        
        Args:
            node_id: Node ID to search for
            node_label: Optional node label to filter by
            
        Returns:
            Node properties dictionary or None if not found
        """
        try:
            async with self.db_manager.get_neo4j_session() as session:
                if node_label:
                    query = f"MATCH (n:{node_label} {{id: $id}}) RETURN n"
                else:
                    query = "MATCH (n {id: $id}) RETURN n, labels(n) as labels"
                
                result = session.run(query, id=node_id)
                record = result.single()
                
                if record:
                    node = record["n"]
                    node_dict = dict(node)
                    if not node_label:
                        node_dict["labels"] = record["labels"]
                    return node_dict
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting node {node_id}: {str(e)}")
            return None
    
    async def update_node_properties(
        self,
        node_id: str,
        properties: Dict[str, Any],
        node_label: Optional[str] = None
    ) -> NodeOperationResult:
        """
        Update properties of an existing node.
        
        Args:
            node_id: Node ID to update
            properties: Properties to update
            node_label: Optional node label for performance
            
        Returns:
            NodeOperationResult with operation details
        """
        try:
            # Add updated timestamp
            properties["updated_at"] = datetime.now().isoformat()
            
            async with self.db_manager.get_neo4j_session() as session:
                if node_label:
                    query = f"""
                    MATCH (n:{node_label} {{id: $id}})
                    SET n += $properties
                    RETURN n.id as id
                    """
                else:
                    query = """
                    MATCH (n {id: $id})
                    SET n += $properties
                    RETURN n.id as id
                    """
                
                result = session.run(query, id=node_id, properties=properties)
                record = result.single()
                
                if record:
                    self._operation_stats.nodes_updated += 1
                    logger.debug(f"Node updated: {node_id}")
                    
                    return NodeOperationResult(
                        success=True,
                        node_id=record["id"],
                        operation="updated",
                        message="Node properties updated successfully",
                        properties=properties
                    )
                else:
                    return NodeOperationResult(
                        success=False,
                        error=f"Node not found: {node_id}"
                    )
                    
        except Exception as e:
            logger.error(f"Error updating node {node_id}: {str(e)}")
            self._operation_stats.errors += 1
            return NodeOperationResult(
                success=False,
                error=f"Failed to update node: {str(e)}"
            )
    
    async def delete_node(self, node_id: str, node_label: Optional[str] = None) -> NodeOperationResult:
        """
        Delete a node and all its relationships.
        
        Args:
            node_id: Node ID to delete
            node_label: Optional node label for performance
            
        Returns:
            NodeOperationResult with operation details
        """
        try:
            async with self.db_manager.get_neo4j_session() as session:
                if node_label:
                    query = f"""
                    MATCH (n:{node_label} {{id: $id}})
                    DETACH DELETE n
                    RETURN count(n) as deleted_count
                    """
                else:
                    query = """
                    MATCH (n {id: $id})
                    DETACH DELETE n
                    RETURN count(n) as deleted_count
                    """
                
                result = session.run(query, id=node_id)
                record = result.single()
                
                if record and record["deleted_count"] > 0:
                    logger.info(f"Node deleted: {node_id}")
                    
                    return NodeOperationResult(
                        success=True,
                        node_id=node_id,
                        operation="deleted",
                        message="Node deleted successfully"
                    )
                else:
                    return NodeOperationResult(
                        success=False,
                        error=f"Node not found: {node_id}"
                    )
                    
        except Exception as e:
            logger.error(f"Error deleting node {node_id}: {str(e)}")
            self._operation_stats.errors += 1
            return NodeOperationResult(
                success=False,
                error=f"Failed to delete node: {str(e)}"
            )
    
    async def get_nodes_by_type(self, node_label: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get nodes by their label/type.
        
        Args:
            node_label: Node label to filter by
            limit: Maximum number of nodes to return
            
        Returns:
            List of node property dictionaries
        """
        try:
            async with self.db_manager.get_neo4j_session() as session:
                query = f"""
                MATCH (n:{node_label})
                RETURN n
                ORDER BY n.created_at DESC
                LIMIT $limit
                """
                
                result = session.run(query, limit=limit)
                nodes = []
                
                for record in result:
                    node = record["n"]
                    nodes.append(dict(node))
                
                return nodes
                
        except Exception as e:
            logger.error(f"Error getting nodes by type {node_label}: {str(e)}")
            return []
    
    async def search_nodes(
        self,
        search_term: str,
        node_labels: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search nodes using full-text search.
        
        Args:
            search_term: Term to search for
            node_labels: Optional list of node labels to filter by
            limit: Maximum number of results
            
        Returns:
            List of matching nodes with relevance scores
        """
        try:
            async with self.db_manager.get_neo4j_session() as session:
                if node_labels:
                    # Use specific full-text indexes
                    results = []
                    for label in node_labels:
                        if label == "Document":
                            query = """
                            CALL db.index.fulltext.queryNodes('document_content_fulltext', $search_term)
                            YIELD node, score
                            RETURN node, score, labels(node) as labels
                            ORDER BY score DESC
                            LIMIT $limit
                            """
                        elif label == "Entity":
                            query = """
                            CALL db.index.fulltext.queryNodes('entity_search_fulltext', $search_term)
                            YIELD node, score
                            RETURN node, score, labels(node) as labels
                            ORDER BY score DESC
                            LIMIT $limit
                            """
                        elif label == "Concept":
                            query = """
                            CALL db.index.fulltext.queryNodes('concept_search_fulltext', $search_term)
                            YIELD node, score
                            RETURN node, score, labels(node) as labels
                            ORDER BY score DESC
                            LIMIT $limit
                            """
                        else:
                            continue
                        
                        result = session.run(query, search_term=search_term, limit=limit)
                        for record in result:
                            node_dict = dict(record["node"])
                            node_dict["_score"] = record["score"]
                            node_dict["_labels"] = record["labels"]
                            results.append(node_dict)
                    
                    # Sort all results by score
                    results.sort(key=lambda x: x["_score"], reverse=True)
                    return results[:limit]
                
                else:
                    # Search all full-text indexes
                    query = """
                    CALL db.index.fulltext.queryNodes('document_content_fulltext', $search_term)
                    YIELD node, score
                    RETURN node, score, labels(node) as labels
                    UNION ALL
                    CALL db.index.fulltext.queryNodes('entity_search_fulltext', $search_term)
                    YIELD node, score
                    RETURN node, score, labels(node) as labels
                    UNION ALL
                    CALL db.index.fulltext.queryNodes('concept_search_fulltext', $search_term)
                    YIELD node, score
                    RETURN node, score, labels(node) as labels
                    ORDER BY score DESC
                    LIMIT $limit
                    """
                    
                    result = session.run(query, search_term=search_term, limit=limit)
                    nodes = []
                    
                    for record in result:
                        node_dict = dict(record["node"])
                        node_dict["_score"] = record["score"]
                        node_dict["_labels"] = record["labels"]
                        nodes.append(node_dict)
                    
                    return nodes
                
        except Exception as e:
            logger.error(f"Error searching nodes: {str(e)}")
            return []
    
    def get_operation_stats(self) -> GraphOperationStats:
        """Get current operation statistics."""
        return self._operation_stats
    
    def reset_operation_stats(self):
        """Reset operation statistics."""
        self._operation_stats = GraphOperationStats()
        logger.info("Operation statistics reset")
    
    async def create_relationship(
        self,
        source_node_id: str,
        target_node_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> NodeOperationResult:
        """
        Create a relationship between two nodes.
        
        Args:
            source_node_id: ID of the source node
            target_node_id: ID of the target node
            relationship_type: Type of relationship
            properties: Optional relationship properties
            
        Returns:
            NodeOperationResult with operation details
        """
        try:
            if not properties:
                properties = {}
            
            # Add timestamp
            properties["created_at"] = datetime.now().isoformat()
            
            async with self.db_manager.get_neo4j_session() as session:
                # First verify both nodes exist
                verify_query = """
                MATCH (source {id: $source_id}), (target {id: $target_id})
                RETURN source, target, labels(source) as source_labels, labels(target) as target_labels
                """
                
                result = session.run(verify_query, source_id=source_node_id, target_id=target_node_id)
                record = result.single()
                
                if not record:
                    return NodeOperationResult(
                        success=False,
                        error=f"One or both nodes not found: {source_node_id}, {target_node_id}"
                    )
                
                # Validate relationship schema if possible
                source_labels = record["source_labels"]
                target_labels = record["target_labels"]
                
                if source_labels and target_labels:
                    validation = self.schema.validate_relationship_properties(
                        relationship_type, source_labels[0], target_labels[0], properties
                    )
                    if not validation["valid"]:
                        logger.warning(f"Relationship schema validation warning: {validation['error']}")
                
                # Create relationship using MERGE to avoid duplicates
                create_query = f"""
                MATCH (source {{id: $source_id}}), (target {{id: $target_id}})
                MERGE (source)-[r:{relationship_type}]->(target)
                ON CREATE SET r += $properties
                ON MATCH SET r += $properties, r.updated_at = $updated_at
                RETURN r, 
                       CASE WHEN r.created_at = $created_at THEN 'created' ELSE 'updated' END as operation
                """
                
                properties["updated_at"] = datetime.now().isoformat()
                created_at = properties["created_at"]
                
                result = session.run(
                    create_query, 
                    source_id=source_node_id, 
                    target_id=target_node_id,
                    properties=properties,
                    created_at=created_at,
                    updated_at=properties["updated_at"]
                )
                record = result.single()
                
                if record:
                    operation = record["operation"]
                    if operation == "created":
                        self._operation_stats.relationships_created += 1
                    else:
                        self._operation_stats.relationships_updated += 1
                    
                    relationship_id = f"{source_node_id}_{relationship_type}_{target_node_id}"
                    logger.debug(f"Relationship {operation}: {relationship_id}")
                    
                    return NodeOperationResult(
                        success=True,
                        node_id=relationship_id,
                        operation=operation,
                        message=f"Relationship {operation} successfully",
                        properties=properties
                    )
                else:
                    return NodeOperationResult(
                        success=False,
                        error="No result returned from relationship creation"
                    )
                    
        except Exception as e:
            logger.error(f"Error creating relationship {source_node_id} -> {target_node_id}: {str(e)}")
            self._operation_stats.errors += 1
            return NodeOperationResult(
                success=False,
                error=f"Failed to create relationship: {str(e)}"
            )
    
    async def create_entity_relationship(self, relationship: Relationship) -> NodeOperationResult:
        """
        Create a relationship from a Relationship object.
        
        Args:
            relationship: Relationship object to create
            
        Returns:
            NodeOperationResult with operation details
        """
        # Map relationship type to Neo4j relationship type
        neo4j_rel_type = self._map_relationship_type(relationship.relationship_type.value)
        
        # Prepare relationship properties
        properties = {
            "relationship_type": relationship.relationship_type.value,
            "confidence": relationship.confidence,
            "created_at": relationship.created_at.isoformat()
        }
        
        if relationship.description:
            properties["description"] = relationship.description
        if relationship.context:
            properties["context"] = relationship.context
        if relationship.source_document_id:
            properties["source_document_id"] = relationship.source_document_id
        if relationship.source_chunk_id:
            properties["source_chunk_id"] = relationship.source_chunk_id
        if relationship.metadata:
            # Flatten metadata - Neo4j doesn't support nested dictionaries
            # Extract common metadata fields as top-level properties
            if isinstance(relationship.metadata, dict):
                for key, value in relationship.metadata.items():
                    # Only add primitive types (string, int, float, bool)
                    if isinstance(value, (str, int, float, bool)):
                        properties[f"meta_{key}"] = value
                    elif isinstance(value, list) and all(isinstance(v, (str, int, float, bool)) for v in value):
                        properties[f"meta_{key}"] = value
                # Store full metadata as JSON string for reference
                import json
                properties["metadata_json"] = json.dumps(relationship.metadata)
            else:
                properties["metadata_json"] = str(relationship.metadata)
        
        return await self.create_relationship(
            source_node_id=relationship.source_entity_id,
            target_node_id=relationship.target_entity_id,
            relationship_type=neo4j_rel_type,
            properties=properties
        )
    
    def _map_relationship_type(self, relationship_type: str) -> str:
        """Map relationship type to Neo4j relationship type."""
        # Map common relationship types to Neo4j schema
        mapping = {
            "WORKS_FOR": "WORKS_FOR",
            "LOCATED_IN": "LOCATED_IN", 
            "PART_OF": "PART_OF",
            "FOUNDED_BY": "FOUNDED_BY",
            "OWNS": "OWNS",
            "COLLABORATES_WITH": "COLLABORATES_WITH",
            "LEADS": "LEADS",
            "MEMBER_OF": "MEMBER_OF",
            "CREATED_BY": "CREATED_BY",
            "USED_BY": "USED_BY"
        }
        
        return mapping.get(relationship_type, "RELATED_TO")
    
    async def create_document_entity_relationship(
        self,
        document_id: str,
        entity_id: str,
        confidence: float = 1.0,
        extraction_method: str = "llm"
    ) -> NodeOperationResult:
        """
        Create a CONTAINS relationship between a document and an entity.
        
        Args:
            document_id: Document node ID
            entity_id: Entity node ID
            confidence: Confidence score
            extraction_method: Method used for extraction
            
        Returns:
            NodeOperationResult with operation details
        """
        properties = {
            "confidence": confidence,
            "extraction_method": extraction_method
        }
        
        return await self.create_relationship(
            source_node_id=document_id,
            target_node_id=entity_id,
            relationship_type="CONTAINS",
            properties=properties
        )
    
    async def get_node_relationships(
        self,
        node_id: str,
        direction: str = "both",
        relationship_types: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get relationships for a specific node.
        
        Args:
            node_id: Node ID to get relationships for
            direction: "incoming", "outgoing", or "both"
            relationship_types: Optional list of relationship types to filter by
            limit: Maximum number of relationships to return
            
        Returns:
            List of relationship dictionaries with connected nodes
        """
        try:
            async with self.db_manager.get_neo4j_session() as session:
                # Build query based on direction
                if direction == "incoming":
                    match_pattern = "(connected)-[r]->(n {id: $node_id})"
                elif direction == "outgoing":
                    match_pattern = "(n {id: $node_id})-[r]->(connected)"
                else:  # both
                    match_pattern = "(n {id: $node_id})-[r]-(connected)"
                
                # Add relationship type filter if specified
                if relationship_types:
                    type_filter = "|".join(relationship_types)
                    match_pattern = match_pattern.replace("[r]", f"[r:{type_filter}]")
                
                query = f"""
                MATCH {match_pattern}
                RETURN r, connected, type(r) as relationship_type, labels(connected) as connected_labels
                ORDER BY r.created_at DESC
                LIMIT $limit
                """
                
                result = session.run(query, node_id=node_id, limit=limit)
                relationships = []
                
                for record in result:
                    rel_dict = dict(record["r"])
                    connected_dict = dict(record["connected"])
                    
                    relationships.append({
                        "relationship": rel_dict,
                        "relationship_type": record["relationship_type"],
                        "connected_node": connected_dict,
                        "connected_labels": record["connected_labels"]
                    })
                
                return relationships
                
        except Exception as e:
            logger.error(f"Error getting relationships for node {node_id}: {str(e)}")
            return []
    
    async def traverse_graph(
        self,
        start_node_id: str,
        max_depth: int = 2,
        relationship_types: Optional[List[str]] = None,
        node_labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Traverse the graph from a starting node.
        
        Args:
            start_node_id: Starting node ID
            max_depth: Maximum traversal depth
            relationship_types: Optional relationship types to follow
            node_labels: Optional node labels to include
            
        Returns:
            Dictionary with traversal results
        """
        try:
            async with self.db_manager.get_neo4j_session() as session:
                # Build traversal query
                match_pattern = f"(start {{id: $start_id}})"
                
                if relationship_types:
                    type_filter = "|".join(relationship_types)
                    rel_pattern = f"[r:{type_filter}*1..{max_depth}]"
                else:
                    rel_pattern = f"[r*1..{max_depth}]"
                
                if node_labels:
                    label_filter = "|".join(node_labels)
                    end_pattern = f"(end:{label_filter})"
                else:
                    end_pattern = "(end)"
                
                query = f"""
                MATCH path = {match_pattern}-{rel_pattern}-{end_pattern}
                WHERE start.id <> end.id
                RETURN path, 
                       start, 
                       end, 
                       relationships(path) as rels,
                       nodes(path) as path_nodes,
                       length(path) as path_length
                ORDER BY path_length, end.name
                LIMIT 200
                """
                
                result = session.run(query, start_id=start_node_id)
                
                paths = []
                all_nodes = {}
                all_relationships = {}
                
                for record in result:
                    path_length = record["path_length"]
                    start_node = dict(record["start"])
                    end_node = dict(record["end"])
                    path_nodes = [dict(node) for node in record["path_nodes"]]
                    relationships = [dict(rel) for rel in record["rels"]]
                    
                    # Store nodes and relationships
                    for node in path_nodes:
                        all_nodes[node["id"]] = node
                    
                    for i, rel in enumerate(relationships):
                        rel_id = f"{path_nodes[i]['id']}_{path_nodes[i+1]['id']}"
                        all_relationships[rel_id] = rel
                    
                    paths.append({
                        "start_node": start_node,
                        "end_node": end_node,
                        "path_length": path_length,
                        "nodes": path_nodes,
                        "relationships": relationships
                    })
                
                return {
                    "start_node_id": start_node_id,
                    "max_depth": max_depth,
                    "total_paths": len(paths),
                    "unique_nodes": len(all_nodes),
                    "unique_relationships": len(all_relationships),
                    "paths": paths,
                    "all_nodes": list(all_nodes.values()),
                    "all_relationships": list(all_relationships.values())
                }
                
        except Exception as e:
            logger.error(f"Error traversing graph from {start_node_id}: {str(e)}")
            return {
                "error": f"Graph traversal failed: {str(e)}",
                "start_node_id": start_node_id
            }
    
    async def find_shortest_path(
        self,
        source_node_id: str,
        target_node_id: str,
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Find the shortest path between two nodes.
        
        Args:
            source_node_id: Source node ID
            target_node_id: Target node ID
            relationship_types: Optional relationship types to follow
            max_depth: Maximum search depth
            
        Returns:
            Dictionary with shortest path or None if no path found
        """
        try:
            async with self.db_manager.get_neo4j_session() as session:
                if relationship_types:
                    type_filter = "|".join(relationship_types)
                    rel_pattern = f"[r:{type_filter}*1..{max_depth}]"
                else:
                    rel_pattern = f"[r*1..{max_depth}]"
                
                query = f"""
                MATCH path = shortestPath((source {{id: $source_id}})-{rel_pattern}-(target {{id: $target_id}}))
                RETURN path,
                       nodes(path) as path_nodes,
                       relationships(path) as path_relationships,
                       length(path) as path_length
                """
                
                result = session.run(query, source_id=source_node_id, target_id=target_node_id)
                record = result.single()
                
                if record:
                    path_nodes = [dict(node) for node in record["path_nodes"]]
                    path_relationships = [dict(rel) for rel in record["path_relationships"]]
                    path_length = record["path_length"]
                    
                    return {
                        "source_node_id": source_node_id,
                        "target_node_id": target_node_id,
                        "path_length": path_length,
                        "nodes": path_nodes,
                        "relationships": path_relationships,
                        "found": True
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Error finding shortest path {source_node_id} -> {target_node_id}: {str(e)}")
            return None
    
    async def get_connected_subgraph(
        self,
        node_ids: List[str],
        include_intermediate: bool = True
    ) -> Dict[str, Any]:
        """
        Get a subgraph containing specified nodes and their connections.
        
        Args:
            node_ids: List of node IDs to include
            include_intermediate: Whether to include intermediate nodes in paths
            
        Returns:
            Dictionary with subgraph data
        """
        try:
            async with self.db_manager.get_neo4j_session() as session:
                if include_intermediate:
                    # Find all paths between the specified nodes
                    query = """
                    MATCH (n1), (n2)
                    WHERE n1.id IN $node_ids AND n2.id IN $node_ids AND n1.id <> n2.id
                    MATCH path = shortestPath((n1)-[*1..3]-(n2))
                    RETURN path,
                           nodes(path) as path_nodes,
                           relationships(path) as path_relationships
                    """
                else:
                    # Only direct connections between specified nodes
                    query = """
                    MATCH (n1)-[r]-(n2)
                    WHERE n1.id IN $node_ids AND n2.id IN $node_ids
                    RETURN n1, n2, r
                    """
                
                result = session.run(query, node_ids=node_ids)
                
                all_nodes = {}
                all_relationships = {}
                
                if include_intermediate:
                    for record in result:
                        path_nodes = [dict(node) for node in record["path_nodes"]]
                        path_relationships = [dict(rel) for rel in record["path_relationships"]]
                        
                        # Store all nodes in paths
                        for node in path_nodes:
                            all_nodes[node["id"]] = node
                        
                        # Store all relationships
                        for i, rel in enumerate(path_relationships):
                            rel_id = f"{path_nodes[i]['id']}_{path_nodes[i+1]['id']}"
                            all_relationships[rel_id] = rel
                else:
                    for record in result:
                        n1 = dict(record["n1"])
                        n2 = dict(record["n2"])
                        rel = dict(record["r"])
                        
                        all_nodes[n1["id"]] = n1
                        all_nodes[n2["id"]] = n2
                        
                        rel_id = f"{n1['id']}_{n2['id']}"
                        all_relationships[rel_id] = rel
                
                return {
                    "requested_nodes": node_ids,
                    "total_nodes": len(all_nodes),
                    "total_relationships": len(all_relationships),
                    "nodes": list(all_nodes.values()),
                    "relationships": list(all_relationships.values()),
                    "include_intermediate": include_intermediate
                }
                
        except Exception as e:
            logger.error(f"Error getting connected subgraph: {str(e)}")
            return {
                "error": f"Failed to get subgraph: {str(e)}",
                "requested_nodes": node_ids
            }
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the graph.
        
        Returns:
            Dictionary with graph statistics
        """
        try:
            async with self.db_manager.get_neo4j_session() as session:
                # Get node counts by label
                node_stats_query = """
                MATCH (n)
                RETURN labels(n) as labels, count(n) as count
                ORDER BY count DESC
                """
                
                result = session.run(node_stats_query)
                node_counts = {}
                total_nodes = 0
                
                for record in result:
                    labels = record["labels"]
                    count = record["count"]
                    total_nodes += count
                    
                    for label in labels:
                        if label not in node_counts:
                            node_counts[label] = 0
                        node_counts[label] += count
                
                # Get relationship counts by type
                rel_stats_query = """
                MATCH ()-[r]->()
                RETURN type(r) as relationship_type, count(r) as count
                ORDER BY count DESC
                """
                
                result = session.run(rel_stats_query)
                relationship_counts = {}
                total_relationships = 0
                
                for record in result:
                    rel_type = record["relationship_type"]
                    count = record["count"]
                    relationship_counts[rel_type] = count
                    total_relationships += count
                
                return {
                    "total_nodes": total_nodes,
                    "total_relationships": total_relationships,
                    "node_counts_by_label": node_counts,
                    "relationship_counts_by_type": relationship_counts,
                    "operation_stats": self._operation_stats.__dict__
                }
                
        except Exception as e:
            logger.error(f"Error getting graph statistics: {str(e)}")
            return {
                "error": f"Failed to get statistics: {str(e)}",
                "operation_stats": self._operation_stats.__dict__
            }
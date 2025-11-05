"""
Cypher query builder for common Neo4j operations.
Provides utilities for building complex Cypher queries programmatically.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of Cypher queries."""
    MATCH = "MATCH"
    CREATE = "CREATE"
    MERGE = "MERGE"
    DELETE = "DELETE"
    SET = "SET"
    REMOVE = "REMOVE"

class Direction(Enum):
    """Relationship direction."""
    OUTGOING = "->"
    INCOMING = "<-"
    UNDIRECTED = "-"

@dataclass
class CypherQuery:
    """Represents a complete Cypher query."""
    query: str
    parameters: Dict[str, Any]
    description: str = ""

class CypherQueryBuilder:
    """
    Builder for constructing Cypher queries programmatically.
    Provides methods for common graph operations and complex queries.
    """
    
    def __init__(self):
        """Initialize Cypher query builder."""
        self.reset()
        logger.debug("CypherQueryBuilder initialized")
    
    def reset(self):
        """Reset the builder to start a new query."""
        self._clauses = []
        self._parameters = {}
        self._description = ""
        return self
    
    def match(self, pattern: str, where: Optional[str] = None) -> 'CypherQueryBuilder':
        """
        Add a MATCH clause.
        
        Args:
            pattern: Cypher pattern to match
            where: Optional WHERE condition
            
        Returns:
            Self for method chaining
        """
        clause = f"MATCH {pattern}"
        if where:
            clause += f" WHERE {where}"
        self._clauses.append(clause)
        return self
    
    def create(self, pattern: str) -> 'CypherQueryBuilder':
        """
        Add a CREATE clause.
        
        Args:
            pattern: Cypher pattern to create
            
        Returns:
            Self for method chaining
        """
        self._clauses.append(f"CREATE {pattern}")
        return self
    
    def merge(self, pattern: str, on_create: Optional[str] = None, on_match: Optional[str] = None) -> 'CypherQueryBuilder':
        """
        Add a MERGE clause with optional ON CREATE/MATCH.
        
        Args:
            pattern: Cypher pattern to merge
            on_create: Optional ON CREATE SET clause
            on_match: Optional ON MATCH SET clause
            
        Returns:
            Self for method chaining
        """
        clause = f"MERGE {pattern}"
        if on_create:
            clause += f" ON CREATE SET {on_create}"
        if on_match:
            clause += f" ON MATCH SET {on_match}"
        self._clauses.append(clause)
        return self
    
    def where(self, condition: str) -> 'CypherQueryBuilder':
        """
        Add a WHERE clause.
        
        Args:
            condition: WHERE condition
            
        Returns:
            Self for method chaining
        """
        self._clauses.append(f"WHERE {condition}")
        return self
    
    def set_properties(self, assignments: str) -> 'CypherQueryBuilder':
        """
        Add a SET clause.
        
        Args:
            assignments: Property assignments
            
        Returns:
            Self for method chaining
        """
        self._clauses.append(f"SET {assignments}")
        return self
    
    def delete(self, variables: str, detach: bool = False) -> 'CypherQueryBuilder':
        """
        Add a DELETE clause.
        
        Args:
            variables: Variables to delete
            detach: Whether to use DETACH DELETE
            
        Returns:
            Self for method chaining
        """
        delete_type = "DETACH DELETE" if detach else "DELETE"
        self._clauses.append(f"{delete_type} {variables}")
        return self
    
    def return_clause(self, expressions: str) -> 'CypherQueryBuilder':
        """
        Add a RETURN clause.
        
        Args:
            expressions: Return expressions
            
        Returns:
            Self for method chaining
        """
        self._clauses.append(f"RETURN {expressions}")
        return self
    
    def order_by(self, expressions: str) -> 'CypherQueryBuilder':
        """
        Add an ORDER BY clause.
        
        Args:
            expressions: Order expressions
            
        Returns:
            Self for method chaining
        """
        self._clauses.append(f"ORDER BY {expressions}")
        return self
    
    def limit(self, count: int) -> 'CypherQueryBuilder':
        """
        Add a LIMIT clause.
        
        Args:
            count: Limit count
            
        Returns:
            Self for method chaining
        """
        self._clauses.append(f"LIMIT {count}")
        return self
    
    def skip(self, count: int) -> 'CypherQueryBuilder':
        """
        Add a SKIP clause.
        
        Args:
            count: Skip count
            
        Returns:
            Self for method chaining
        """
        self._clauses.append(f"SKIP {count}")
        return self
    
    def with_clause(self, expressions: str) -> 'CypherQueryBuilder':
        """
        Add a WITH clause.
        
        Args:
            expressions: WITH expressions
            
        Returns:
            Self for method chaining
        """
        self._clauses.append(f"WITH {expressions}")
        return self
    
    def union(self, all_union: bool = False) -> 'CypherQueryBuilder':
        """
        Add a UNION clause.
        
        Args:
            all_union: Whether to use UNION ALL
            
        Returns:
            Self for method chaining
        """
        union_type = "UNION ALL" if all_union else "UNION"
        self._clauses.append(union_type)
        return self
    
    def add_parameter(self, name: str, value: Any) -> 'CypherQueryBuilder':
        """
        Add a query parameter.
        
        Args:
            name: Parameter name
            value: Parameter value
            
        Returns:
            Self for method chaining
        """
        self._parameters[name] = value
        return self
    
    def add_parameters(self, parameters: Dict[str, Any]) -> 'CypherQueryBuilder':
        """
        Add multiple query parameters.
        
        Args:
            parameters: Dictionary of parameters
            
        Returns:
            Self for method chaining
        """
        self._parameters.update(parameters)
        return self
    
    def set_description(self, description: str) -> 'CypherQueryBuilder':
        """
        Set query description.
        
        Args:
            description: Query description
            
        Returns:
            Self for method chaining
        """
        self._description = description
        return self
    
    def build(self) -> CypherQuery:
        """
        Build the final Cypher query.
        
        Returns:
            CypherQuery object with query string and parameters
        """
        query = "\n".join(self._clauses)
        return CypherQuery(
            query=query,
            parameters=self._parameters.copy(),
            description=self._description
        )
    
    def build_string(self) -> str:
        """
        Build just the query string.
        
        Returns:
            Cypher query string
        """
        return "\n".join(self._clauses)

class CommonQueries:
    """
    Collection of common Cypher queries for the knowledge graph.
    Provides pre-built queries for frequent operations.
    """
    
    @staticmethod
    def find_entity_by_name(entity_name: str, entity_type: Optional[str] = None) -> CypherQuery:
        """Find entity by name and optionally by type."""
        builder = CypherQueryBuilder()
        
        if entity_type:
            builder.match("(e:Entity {name: $name, type: $type})")
            builder.add_parameter("type", entity_type)
        else:
            builder.match("(e:Entity {name: $name})")
        
        builder.return_clause("e")
        builder.add_parameter("name", entity_name)
        builder.set_description(f"Find entity by name: {entity_name}")
        
        return builder.build()
    
    @staticmethod
    def find_entities_in_document(document_id: str) -> CypherQuery:
        """Find all entities contained in a document."""
        builder = CypherQueryBuilder()
        builder.match("(d:Document {id: $doc_id})-[:CONTAINS]->(e:Entity)")
        builder.return_clause("e, d")
        builder.order_by("e.name")
        builder.add_parameter("doc_id", document_id)
        builder.set_description(f"Find entities in document: {document_id}")
        
        return builder.build()
    
    @staticmethod
    def find_related_entities(entity_id: str, max_depth: int = 2) -> CypherQuery:
        """Find entities related to a given entity within max depth."""
        builder = CypherQueryBuilder()
        builder.match(f"(start:Entity {{id: $entity_id}})-[r*1..{max_depth}]-(related:Entity)")
        builder.where("start.id <> related.id")
        builder.return_clause("related, r, length(r) as depth")
        builder.order_by("depth, related.name")
        builder.limit(50)
        builder.add_parameter("entity_id", entity_id)
        builder.set_description(f"Find entities related to: {entity_id}")
        
        return builder.build()
    
    @staticmethod
    def find_shortest_path_between_entities(source_id: str, target_id: str) -> CypherQuery:
        """Find shortest path between two entities."""
        builder = CypherQueryBuilder()
        builder.match("(source:Entity {id: $source_id}), (target:Entity {id: $target_id})")
        builder.match("path = shortestPath((source)-[*1..5]-(target))")
        builder.return_clause("path, nodes(path) as path_nodes, relationships(path) as path_rels, length(path) as path_length")
        builder.add_parameters({"source_id": source_id, "target_id": target_id})
        builder.set_description(f"Shortest path: {source_id} -> {target_id}")
        
        return builder.build()
    
    @staticmethod
    def search_entities_by_text(search_term: str, limit: int = 20) -> CypherQuery:
        """Search entities using full-text search."""
        builder = CypherQueryBuilder()
        builder.match("CALL db.index.fulltext.queryNodes('entity_search_fulltext', $search_term) YIELD node, score")
        builder.return_clause("node as entity, score")
        builder.order_by("score DESC")
        builder.limit(limit)
        builder.add_parameter("search_term", search_term)
        builder.set_description(f"Full-text search for entities: {search_term}")
        
        return builder.build()
    
    @staticmethod
    def get_entity_neighborhood(entity_id: str, relationship_types: Optional[List[str]] = None) -> CypherQuery:
        """Get immediate neighborhood of an entity."""
        builder = CypherQueryBuilder()
        
        if relationship_types:
            rel_filter = "|".join(relationship_types)
            pattern = f"(center:Entity {{id: $entity_id}})-[r:{rel_filter}]-(neighbor)"
        else:
            pattern = "(center:Entity {id: $entity_id})-[r]-(neighbor)"
        
        builder.match(pattern)
        builder.return_clause("center, r, neighbor, type(r) as relationship_type")
        builder.order_by("relationship_type, neighbor.name")
        builder.add_parameter("entity_id", entity_id)
        builder.set_description(f"Get neighborhood of entity: {entity_id}")
        
        return builder.build()
    
    @staticmethod
    def get_document_statistics() -> CypherQuery:
        """Get statistics about documents in the graph."""
        builder = CypherQueryBuilder()
        builder.match("(d:Document)")
        builder.match("(d)-[:CONTAINS]->(e:Entity)")
        builder.return_clause("""
            d.id as document_id,
            d.title as title,
            count(e) as entity_count,
            collect(DISTINCT e.type) as entity_types
        """)
        builder.order_by("entity_count DESC")
        builder.set_description("Get document statistics")
        
        return builder.build()
    
    @staticmethod
    def get_most_connected_entities(limit: int = 10) -> CypherQuery:
        """Get entities with the most connections."""
        builder = CypherQueryBuilder()
        builder.match("(e:Entity)-[r]-()")
        builder.return_clause("e, count(r) as connection_count")
        builder.order_by("connection_count DESC")
        builder.limit(limit)
        builder.set_description("Get most connected entities")
        
        return builder.build()
    
    @staticmethod
    def find_entities_by_type_and_confidence(
        entity_type: str, 
        min_confidence: float = 0.5
    ) -> CypherQuery:
        """Find entities by type with minimum confidence."""
        builder = CypherQueryBuilder()
        builder.match("(e:Entity)")
        builder.where("e.type = $entity_type AND e.confidence >= $min_confidence")
        builder.return_clause("e")
        builder.order_by("e.confidence DESC, e.name")
        builder.add_parameters({
            "entity_type": entity_type,
            "min_confidence": min_confidence
        })
        builder.set_description(f"Find {entity_type} entities with confidence >= {min_confidence}")
        
        return builder.build()
    
    @staticmethod
    def get_relationship_statistics() -> CypherQuery:
        """Get statistics about relationships in the graph."""
        builder = CypherQueryBuilder()
        builder.match("()-[r]->()")
        builder.return_clause("type(r) as relationship_type, count(r) as count")
        builder.order_by("count DESC")
        builder.set_description("Get relationship statistics")
        
        return builder.build()
    
    @staticmethod
    def find_co_occurring_entities(document_id: str) -> CypherQuery:
        """Find entities that co-occur in the same document."""
        builder = CypherQueryBuilder()
        builder.match("(d:Document {id: $doc_id})-[:CONTAINS]->(e1:Entity)")
        builder.match("(d)-[:CONTAINS]->(e2:Entity)")
        builder.where("e1.id <> e2.id")
        builder.return_clause("e1, e2, d")
        builder.order_by("e1.name, e2.name")
        builder.add_parameter("doc_id", document_id)
        builder.set_description(f"Find co-occurring entities in document: {document_id}")
        
        return builder.build()
    
    @staticmethod
    def create_entity_node(entity_data: Dict[str, Any]) -> CypherQuery:
        """Create an entity node with properties."""
        builder = CypherQueryBuilder()
        builder.merge("(e:Entity {id: $id})")
        
        # Build property assignments for ON CREATE and ON MATCH
        properties = []
        for key, value in entity_data.items():
            if key != "id":
                properties.append(f"e.{key} = ${key}")
        
        property_assignments = ", ".join(properties)
        builder.merge(
            "(e:Entity {id: $id})",
            on_create=property_assignments + ", e.created_at = timestamp()",
            on_match=property_assignments + ", e.updated_at = timestamp()"
        )
        builder.return_clause("e")
        builder.add_parameters(entity_data)
        builder.set_description(f"Create/update entity: {entity_data.get('name', 'unknown')}")
        
        return builder.build()

def build_text_to_cypher_prompt(
    user_question: str,
    schema_info: Dict[str, Any],
    examples: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Build a prompt for Text-to-Cypher generation.
    
    Args:
        user_question: Natural language question
        schema_info: Graph schema information
        examples: Optional examples of question-cypher pairs
        
    Returns:
        Formatted prompt for LLM
    """
    prompt = f"""You are a Neo4j Cypher expert. Convert the natural language question to a precise Cypher query.

GRAPH SCHEMA:
Node Labels: {', '.join(schema_info.get('node_labels', []))}
Relationship Types: {', '.join(schema_info.get('relationship_types', []))}

SCHEMA DETAILS:
- Document nodes have properties: id, title, content, file_type, created_at
- Entity nodes have properties: id, name, type, description, confidence, source_document_id
- Concept nodes have properties: id, name, definition, category

RELATIONSHIPS:
- (Document)-[:CONTAINS]->(Entity)
- (Entity)-[:RELATED_TO]->(Entity)
- (Entity)-[:IS_A]->(Concept)
- Various specific relationships: WORKS_FOR, LOCATED_IN, PART_OF, etc.

QUERY GUIDELINES:
1. Use MATCH for retrieval queries
2. Always include LIMIT to prevent large result sets
3. Use WHERE clauses for filtering
4. Return meaningful node properties
5. Use ORDER BY for sorted results

"""
    
    if examples:
        prompt += "EXAMPLES:\n"
        for example in examples:
            prompt += f"Question: {example['question']}\n"
            prompt += f"Cypher: {example['cypher']}\n\n"
    
    prompt += f"""USER QUESTION: {user_question}

Generate a Cypher query that answers this question. Return only the Cypher query, no explanation:"""
    
    return prompt
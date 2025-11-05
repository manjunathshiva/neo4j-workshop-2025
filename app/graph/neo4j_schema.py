"""
Neo4j graph schema definition and management.
Defines node types, relationship types, and indexing for the knowledge graph.
"""

import logging
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class NodeType(Enum):
    """Neo4j node types for the knowledge graph."""
    DOCUMENT = "Document"
    ENTITY = "Entity"
    CONCEPT = "Concept"

class RelationshipType(Enum):
    """Neo4j relationship types for the knowledge graph."""
    CONTAINS = "CONTAINS"
    RELATED_TO = "RELATED_TO"
    IS_A = "IS_A"
    WORKS_FOR = "WORKS_FOR"
    LOCATED_IN = "LOCATED_IN"
    PART_OF = "PART_OF"
    FOUNDED_BY = "FOUNDED_BY"
    OWNS = "OWNS"
    COLLABORATES_WITH = "COLLABORATES_WITH"
    LEADS = "LEADS"
    MEMBER_OF = "MEMBER_OF"
    CREATED_BY = "CREATED_BY"
    USED_BY = "USED_BY"

@dataclass
class NodeSchema:
    """Schema definition for a Neo4j node type."""
    label: str
    required_properties: List[str]
    optional_properties: List[str]
    indexes: List[str]
    constraints: List[str]

@dataclass
class RelationshipSchema:
    """Schema definition for a Neo4j relationship type."""
    type: str
    from_labels: List[str]
    to_labels: List[str]
    properties: List[str]

class Neo4jSchema:
    """
    Neo4j schema manager for the knowledge graph.
    Defines node types, relationship types, indexes, and constraints.
    """
    
    def __init__(self):
        """Initialize Neo4j schema definitions."""
        self.node_schemas = self._define_node_schemas()
        self.relationship_schemas = self._define_relationship_schemas()
        self.indexes = self._define_indexes()
        self.constraints = self._define_constraints()
        
        logger.info("Neo4j schema initialized")
    
    def _define_node_schemas(self) -> Dict[str, NodeSchema]:
        """Define schemas for all node types."""
        return {
            NodeType.DOCUMENT.value: NodeSchema(
                label="Document",
                required_properties=["id", "title", "content"],
                optional_properties=[
                    "file_type", "file_size", "created_at", "updated_at",
                    "processing_status", "chunk_count", "entity_count",
                    "source_path", "metadata"
                ],
                indexes=["id", "title", "created_at"],
                constraints=["id"]  # Unique constraint on id
            ),
            
            NodeType.ENTITY.value: NodeSchema(
                label="Entity",
                required_properties=["id", "name", "type"],
                optional_properties=[
                    "description", "confidence", "context",
                    "source_document_id", "source_chunk_id",
                    "start_char", "end_char", "created_at",
                    "extraction_method", "metadata"
                ],
                indexes=["id", "name", "type", "source_document_id"],
                constraints=["id"]  # Unique constraint on id
            ),
            
            NodeType.CONCEPT.value: NodeSchema(
                label="Concept",
                required_properties=["id", "name"],
                optional_properties=[
                    "definition", "category", "description",
                    "related_terms", "confidence", "created_at",
                    "source_documents", "metadata"
                ],
                indexes=["id", "name", "category"],
                constraints=["id"]  # Unique constraint on id
            )
        }
    
    def _define_relationship_schemas(self) -> Dict[str, RelationshipSchema]:
        """Define schemas for all relationship types."""
        return {
            RelationshipType.CONTAINS.value: RelationshipSchema(
                type="CONTAINS",
                from_labels=["Document"],
                to_labels=["Entity"],
                properties=["confidence", "extraction_method", "created_at"]
            ),
            
            RelationshipType.RELATED_TO.value: RelationshipSchema(
                type="RELATED_TO",
                from_labels=["Entity"],
                to_labels=["Entity"],
                properties=["relationship_type", "confidence", "description", "context", "created_at"]
            ),
            
            RelationshipType.IS_A.value: RelationshipSchema(
                type="IS_A",
                from_labels=["Entity"],
                to_labels=["Concept"],
                properties=["confidence", "context", "created_at"]
            ),
            
            RelationshipType.WORKS_FOR.value: RelationshipSchema(
                type="WORKS_FOR",
                from_labels=["Entity"],
                to_labels=["Entity"],
                properties=["confidence", "description", "context", "created_at"]
            ),
            
            RelationshipType.LOCATED_IN.value: RelationshipSchema(
                type="LOCATED_IN",
                from_labels=["Entity"],
                to_labels=["Entity"],
                properties=["confidence", "description", "context", "created_at"]
            ),
            
            RelationshipType.PART_OF.value: RelationshipSchema(
                type="PART_OF",
                from_labels=["Entity"],
                to_labels=["Entity"],
                properties=["confidence", "description", "context", "created_at"]
            ),
            
            RelationshipType.FOUNDED_BY.value: RelationshipSchema(
                type="FOUNDED_BY",
                from_labels=["Entity"],
                to_labels=["Entity"],
                properties=["confidence", "description", "context", "created_at"]
            ),
            
            RelationshipType.OWNS.value: RelationshipSchema(
                type="OWNS",
                from_labels=["Entity"],
                to_labels=["Entity"],
                properties=["confidence", "description", "context", "created_at"]
            ),
            
            RelationshipType.COLLABORATES_WITH.value: RelationshipSchema(
                type="COLLABORATES_WITH",
                from_labels=["Entity"],
                to_labels=["Entity"],
                properties=["confidence", "description", "context", "created_at"]
            ),
            
            RelationshipType.LEADS.value: RelationshipSchema(
                type="LEADS",
                from_labels=["Entity"],
                to_labels=["Entity"],
                properties=["confidence", "description", "context", "created_at"]
            ),
            
            RelationshipType.MEMBER_OF.value: RelationshipSchema(
                type="MEMBER_OF",
                from_labels=["Entity"],
                to_labels=["Entity"],
                properties=["confidence", "description", "context", "created_at"]
            ),
            
            RelationshipType.CREATED_BY.value: RelationshipSchema(
                type="CREATED_BY",
                from_labels=["Entity"],
                to_labels=["Entity"],
                properties=["confidence", "description", "context", "created_at"]
            ),
            
            RelationshipType.USED_BY.value: RelationshipSchema(
                type="USED_BY",
                from_labels=["Entity"],
                to_labels=["Entity"],
                properties=["confidence", "description", "context", "created_at"]
            )
        }
    
    def _define_indexes(self) -> List[str]:
        """Define indexes for performance optimization."""
        return [
            # Document indexes
            "CREATE INDEX document_id_index IF NOT EXISTS FOR (d:Document) ON (d.id)",
            "CREATE INDEX document_title_index IF NOT EXISTS FOR (d:Document) ON (d.title)",
            "CREATE INDEX document_created_at_index IF NOT EXISTS FOR (d:Document) ON (d.created_at)",
            "CREATE INDEX document_file_type_index IF NOT EXISTS FOR (d:Document) ON (d.file_type)",
            
            # Entity indexes
            "CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX entity_source_document_index IF NOT EXISTS FOR (e:Entity) ON (e.source_document_id)",
            "CREATE INDEX entity_confidence_index IF NOT EXISTS FOR (e:Entity) ON (e.confidence)",
            
            # Concept indexes
            "CREATE INDEX concept_id_index IF NOT EXISTS FOR (c:Concept) ON (c.id)",
            "CREATE INDEX concept_name_index IF NOT EXISTS FOR (c:Concept) ON (c.name)",
            "CREATE INDEX concept_category_index IF NOT EXISTS FOR (c:Concept) ON (c.category)",
            
            # Full-text search indexes
            "CREATE FULLTEXT INDEX document_content_fulltext IF NOT EXISTS FOR (d:Document) ON EACH [d.title, d.content]",
            "CREATE FULLTEXT INDEX entity_search_fulltext IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.description, e.context]",
            "CREATE FULLTEXT INDEX concept_search_fulltext IF NOT EXISTS FOR (c:Concept) ON EACH [c.name, c.definition, c.description]"
        ]
    
    def _define_constraints(self) -> List[str]:
        """Define constraints for data integrity."""
        return [
            # Unique constraints
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
            
            # Node key constraints (composite uniqueness)
            "CREATE CONSTRAINT document_node_key IF NOT EXISTS FOR (d:Document) REQUIRE (d.id, d.title) IS NODE KEY",
            "CREATE CONSTRAINT entity_node_key IF NOT EXISTS FOR (e:Entity) REQUIRE (e.id, e.name, e.type) IS NODE KEY",
            "CREATE CONSTRAINT concept_node_key IF NOT EXISTS FOR (c:Concept) REQUIRE (c.id, c.name) IS NODE KEY",
            
            # Property existence constraints
            "CREATE CONSTRAINT document_id_exists IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS NOT NULL",
            "CREATE CONSTRAINT document_title_exists IF NOT EXISTS FOR (d:Document) REQUIRE d.title IS NOT NULL",
            "CREATE CONSTRAINT entity_id_exists IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS NOT NULL",
            "CREATE CONSTRAINT entity_name_exists IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS NOT NULL",
            "CREATE CONSTRAINT entity_type_exists IF NOT EXISTS FOR (e:Entity) REQUIRE e.type IS NOT NULL",
            "CREATE CONSTRAINT concept_id_exists IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS NOT NULL",
            "CREATE CONSTRAINT concept_name_exists IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS NOT NULL"
        ]
    
    def get_node_schema(self, node_type: str) -> Optional[NodeSchema]:
        """Get schema for a specific node type."""
        return self.node_schemas.get(node_type)
    
    def get_relationship_schema(self, relationship_type: str) -> Optional[RelationshipSchema]:
        """Get schema for a specific relationship type."""
        return self.relationship_schemas.get(relationship_type)
    
    def get_all_node_labels(self) -> List[str]:
        """Get all defined node labels."""
        return list(self.node_schemas.keys())
    
    def get_all_relationship_types(self) -> List[str]:
        """Get all defined relationship types."""
        return list(self.relationship_schemas.keys())
    
    def validate_node_properties(self, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate node properties against schema.
        
        Args:
            label: Node label
            properties: Properties to validate
            
        Returns:
            Dictionary with validation results
        """
        schema = self.get_node_schema(label)
        if not schema:
            return {
                "valid": False,
                "error": f"Unknown node label: {label}",
                "missing_required": [],
                "extra_properties": []
            }
        
        # Check required properties
        missing_required = []
        for required_prop in schema.required_properties:
            if required_prop not in properties:
                missing_required.append(required_prop)
        
        # Check for extra properties (not in required or optional)
        all_allowed = set(schema.required_properties + schema.optional_properties)
        extra_properties = []
        for prop in properties.keys():
            if prop not in all_allowed:
                extra_properties.append(prop)
        
        is_valid = len(missing_required) == 0
        
        return {
            "valid": is_valid,
            "error": None if is_valid else f"Missing required properties: {missing_required}",
            "missing_required": missing_required,
            "extra_properties": extra_properties,
            "warnings": [f"Extra property: {prop}" for prop in extra_properties] if extra_properties else []
        }
    
    def validate_relationship_properties(
        self, 
        relationship_type: str, 
        from_label: str, 
        to_label: str,
        properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate relationship properties and labels against schema.
        
        Args:
            relationship_type: Type of relationship
            from_label: Source node label
            to_label: Target node label
            properties: Relationship properties
            
        Returns:
            Dictionary with validation results
        """
        schema = self.get_relationship_schema(relationship_type)
        if not schema:
            return {
                "valid": False,
                "error": f"Unknown relationship type: {relationship_type}",
                "label_valid": False,
                "property_warnings": []
            }
        
        # Validate labels
        label_valid = (from_label in schema.from_labels and to_label in schema.to_labels)
        
        # Check properties (all are optional for relationships)
        property_warnings = []
        for prop in properties.keys():
            if prop not in schema.properties:
                property_warnings.append(f"Unexpected property: {prop}")
        
        return {
            "valid": label_valid,
            "error": None if label_valid else f"Invalid labels: {from_label} -> {to_label} for {relationship_type}",
            "label_valid": label_valid,
            "property_warnings": property_warnings,
            "allowed_from_labels": schema.from_labels,
            "allowed_to_labels": schema.to_labels
        }
    
    def get_schema_summary(self) -> Dict[str, Any]:
        """Get a summary of the complete schema."""
        return {
            "node_types": {
                label: {
                    "required_properties": schema.required_properties,
                    "optional_properties": schema.optional_properties,
                    "indexes": schema.indexes,
                    "constraints": schema.constraints
                }
                for label, schema in self.node_schemas.items()
            },
            "relationship_types": {
                rel_type: {
                    "from_labels": schema.from_labels,
                    "to_labels": schema.to_labels,
                    "properties": schema.properties
                }
                for rel_type, schema in self.relationship_schemas.items()
            },
            "total_indexes": len(self.indexes),
            "total_constraints": len(self.constraints)
        }
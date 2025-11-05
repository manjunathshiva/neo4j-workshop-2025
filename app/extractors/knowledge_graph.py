"""
Knowledge graph representation and management.
Combines entities and relationships into a unified graph structure.
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from .entity_extractor import Entity, EntityType, EntityExtractor
from .relationship_extractor import Relationship, RelationshipType, RelationshipExtractor

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeGraphStats:
    """Statistics about the knowledge graph."""
    total_entities: int = 0
    total_relationships: int = 0
    entity_type_counts: Dict[str, int] = field(default_factory=dict)
    relationship_type_counts: Dict[str, int] = field(default_factory=dict)
    avg_entity_confidence: float = 0.0
    avg_relationship_confidence: float = 0.0
    connected_components: int = 0
    isolated_entities: int = 0
    most_connected_entities: List[Tuple[str, int]] = field(default_factory=list)

class KnowledgeGraph:
    """
    Knowledge graph that combines entities and relationships.
    Provides unified interface for knowledge extraction and graph operations.
    """
    
    def __init__(self):
        """Initialize knowledge graph with extractors."""
        self.entity_extractor = EntityExtractor()
        self.relationship_extractor = RelationshipExtractor()
        
        # Graph storage
        self._entities: Dict[str, Entity] = {}
        self._relationships: Dict[str, Relationship] = {}
        
        # Graph structure for efficient queries
        self._entity_relationships: Dict[str, Set[str]] = defaultdict(set)  # entity_id -> relationship_ids
        self._adjacency_list: Dict[str, Set[str]] = defaultdict(set)  # entity_id -> connected_entity_ids
        
        logger.info("KnowledgeGraph initialized")
    
    def extract_knowledge_from_text(
        self,
        text: str,
        document_id: str = "",
        chunk_id: str = "",
        entity_types: Optional[List[EntityType]] = None,
        relationship_types: Optional[List[RelationshipType]] = None
    ) -> Dict[str, Any]:
        """
        Extract entities and relationships from text and add to knowledge graph.
        
        Args:
            text: Text to extract knowledge from
            document_id: Source document identifier
            chunk_id: Source chunk identifier
            entity_types: Entity types to extract (default: all common types)
            relationship_types: Relationship types to extract (default: all common types)
            
        Returns:
            Dictionary with extraction results and statistics
        """
        start_time = datetime.now()
        
        if not text.strip():
            return {
                "success": False,
                "error_message": "Empty text provided",
                "entities_extracted": 0,
                "relationships_extracted": 0
            }
        
        try:
            # Extract entities first
            logger.info(f"Extracting entities from text (length: {len(text)})")
            entity_result = self.entity_extractor.extract_entities(
                text=text,
                document_id=document_id,
                chunk_id=chunk_id,
                entity_types=entity_types
            )
            
            if not entity_result.success:
                return {
                    "success": False,
                    "error_message": f"Entity extraction failed: {entity_result.error_message}",
                    "entities_extracted": 0,
                    "relationships_extracted": 0
                }
            
            # Deduplicate entities across documents
            deduplicated_entities = self.entity_extractor.deduplicate_across_documents(entity_result.entities)
            
            # Add entities to graph
            entities_added = 0
            for entity in deduplicated_entities:
                if self.add_entity(entity):
                    entities_added += 1
            
            # Extract relationships if we have enough entities
            relationships_added = 0
            if len(deduplicated_entities) >= 2:
                logger.info(f"Extracting relationships between {len(deduplicated_entities)} entities")
                relationship_result = self.relationship_extractor.extract_relationships(
                    text=text,
                    entities=deduplicated_entities,
                    document_id=document_id,
                    chunk_id=chunk_id,
                    relationship_types=relationship_types
                )
                
                if relationship_result.success:
                    # Deduplicate relationships across documents
                    deduplicated_relationships = self.relationship_extractor.deduplicate_across_documents(
                        relationship_result.relationships
                    )
                    
                    # Add relationships to graph
                    for relationship in deduplicated_relationships:
                        if self.add_relationship(relationship):
                            relationships_added += 1
                else:
                    logger.warning(f"Relationship extraction failed: {relationship_result.error_message}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Knowledge extraction completed: {entities_added} entities, {relationships_added} relationships")
            
            return {
                "success": True,
                "entities_extracted": entities_added,
                "relationships_extracted": relationships_added,
                "processing_time_seconds": processing_time,
                "total_entities_in_graph": len(self._entities),
                "total_relationships_in_graph": len(self._relationships),
                "entity_extraction_result": entity_result.metadata,
                "relationship_extraction_result": relationship_result.metadata if len(deduplicated_entities) >= 2 else {}
            }
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error extracting knowledge from text: {str(e)}")
            
            return {
                "success": False,
                "error_message": f"Knowledge extraction failed: {str(e)}",
                "entities_extracted": 0,
                "relationships_extracted": 0,
                "processing_time_seconds": processing_time
            }
    
    def add_entity(self, entity: Entity) -> bool:
        """
        Add entity to the knowledge graph.
        
        Args:
            entity: Entity to add
            
        Returns:
            True if entity was added (new), False if updated (existing)
        """
        is_new = entity.id not in self._entities
        self._entities[entity.id] = entity
        
        # Initialize adjacency list entry
        if entity.id not in self._adjacency_list:
            self._adjacency_list[entity.id] = set()
        
        return is_new
    
    def add_relationship(self, relationship: Relationship) -> bool:
        """
        Add relationship to the knowledge graph.
        
        Args:
            relationship: Relationship to add
            
        Returns:
            True if relationship was added (new), False if updated (existing)
        """
        # Validate that both entities exist
        if (relationship.source_entity_id not in self._entities or 
            relationship.target_entity_id not in self._entities):
            logger.warning(f"Cannot add relationship {relationship.id}: missing entities")
            return False
        
        is_new = relationship.id not in self._relationships
        self._relationships[relationship.id] = relationship
        
        # Update graph structure
        source_id = relationship.source_entity_id
        target_id = relationship.target_entity_id
        
        # Add to entity-relationship mapping
        self._entity_relationships[source_id].add(relationship.id)
        self._entity_relationships[target_id].add(relationship.id)
        
        # Add to adjacency list (bidirectional for graph traversal)
        self._adjacency_list[source_id].add(target_id)
        self._adjacency_list[target_id].add(source_id)
        
        return is_new
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self._entities.get(entity_id)
    
    def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """Get relationship by ID."""
        return self._relationships.get(relationship_id)
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get all entities of a specific type."""
        return [entity for entity in self._entities.values() 
                if entity.entity_type == entity_type]
    
    def get_relationships_by_type(self, relationship_type: RelationshipType) -> List[Relationship]:
        """Get all relationships of a specific type."""
        return [relationship for relationship in self._relationships.values() 
                if relationship.relationship_type == relationship_type]
    
    def get_entity_relationships(self, entity_id: str) -> List[Relationship]:
        """Get all relationships involving a specific entity."""
        if entity_id not in self._entity_relationships:
            return []
        
        relationship_ids = self._entity_relationships[entity_id]
        return [self._relationships[rel_id] for rel_id in relationship_ids 
                if rel_id in self._relationships]
    
    def get_connected_entities(self, entity_id: str, max_depth: int = 1) -> Dict[str, Any]:
        """
        Get entities connected to a specific entity within max_depth hops.
        
        Args:
            entity_id: Starting entity ID
            max_depth: Maximum depth to traverse (default: 1)
            
        Returns:
            Dictionary with connected entities and paths
        """
        if entity_id not in self._entities:
            return {"error": "Entity not found"}
        
        visited = set()
        result = {
            "center_entity": self._entities[entity_id].to_dict(),
            "connected_entities": [],
            "relationships": [],
            "depth_levels": {}
        }
        
        # BFS to find connected entities
        queue = [(entity_id, 0)]  # (entity_id, depth)
        visited.add(entity_id)
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            # Get connected entities
            for connected_id in self._adjacency_list.get(current_id, set()):
                if connected_id not in visited:
                    visited.add(connected_id)
                    queue.append((connected_id, depth + 1))
                    
                    # Add to result
                    connected_entity = self._entities[connected_id]
                    result["connected_entities"].append(connected_entity.to_dict())
                    
                    # Track depth level
                    if depth + 1 not in result["depth_levels"]:
                        result["depth_levels"][depth + 1] = []
                    result["depth_levels"][depth + 1].append(connected_id)
            
            # Get relationships for current entity
            for relationship in self.get_entity_relationships(current_id):
                if (relationship.source_entity_id in visited and 
                    relationship.target_entity_id in visited):
                    result["relationships"].append(relationship.to_dict())
        
        return result
    
    def search_entities(self, query: str, entity_types: Optional[List[EntityType]] = None) -> List[Entity]:
        """
        Search entities by name or description.
        
        Args:
            query: Search query
            entity_types: Filter by entity types (optional)
            
        Returns:
            List of matching entities
        """
        query_lower = query.lower().strip()
        if not query_lower:
            return []
        
        matches = []
        for entity in self._entities.values():
            # Filter by type if specified
            if entity_types and entity.entity_type not in entity_types:
                continue
            
            # Check name match
            if query_lower in entity.name.lower():
                matches.append(entity)
                continue
            
            # Check description match
            if entity.description and query_lower in entity.description.lower():
                matches.append(entity)
                continue
        
        # Sort by relevance (exact matches first, then by confidence)
        matches.sort(key=lambda e: (
            0 if query_lower == e.name.lower() else 1,  # Exact name match first
            -e.confidence  # Higher confidence first
        ))
        
        return matches
    
    def get_subgraph(self, entity_ids: List[str]) -> Dict[str, Any]:
        """
        Get subgraph containing specified entities and their relationships.
        
        Args:
            entity_ids: List of entity IDs to include
            
        Returns:
            Dictionary with subgraph data
        """
        # Filter valid entity IDs
        valid_entity_ids = [eid for eid in entity_ids if eid in self._entities]
        
        if not valid_entity_ids:
            return {"entities": [], "relationships": []}
        
        # Get entities
        entities = [self._entities[eid].to_dict() for eid in valid_entity_ids]
        
        # Get relationships between these entities
        relationships = []
        for relationship in self._relationships.values():
            if (relationship.source_entity_id in valid_entity_ids and 
                relationship.target_entity_id in valid_entity_ids):
                relationships.append(relationship.to_dict())
        
        return {
            "entities": entities,
            "relationships": relationships,
            "entity_count": len(entities),
            "relationship_count": len(relationships)
        }
    
    def get_graph_statistics(self) -> KnowledgeGraphStats:
        """Get comprehensive statistics about the knowledge graph."""
        entities = list(self._entities.values())
        relationships = list(self._relationships.values())
        
        # Entity type counts
        entity_type_counts = {}
        for entity_type in EntityType:
            entity_type_counts[entity_type.value] = sum(
                1 for e in entities if e.entity_type == entity_type
            )
        
        # Relationship type counts
        relationship_type_counts = {}
        for rel_type in RelationshipType:
            relationship_type_counts[rel_type.value] = sum(
                1 for r in relationships if r.relationship_type == rel_type
            )
        
        # Confidence averages
        entity_confidences = [e.confidence for e in entities if e.confidence > 0]
        relationship_confidences = [r.confidence for r in relationships if r.confidence > 0]
        
        avg_entity_confidence = sum(entity_confidences) / len(entity_confidences) if entity_confidences else 0.0
        avg_relationship_confidence = sum(relationship_confidences) / len(relationship_confidences) if relationship_confidences else 0.0
        
        # Graph connectivity analysis
        isolated_entities = sum(1 for entity_id in self._entities.keys() 
                              if len(self._adjacency_list.get(entity_id, set())) == 0)
        
        # Most connected entities
        entity_connections = [(entity_id, len(connections)) 
                            for entity_id, connections in self._adjacency_list.items()]
        entity_connections.sort(key=lambda x: x[1], reverse=True)
        most_connected = [(self._entities[eid].name, count) 
                         for eid, count in entity_connections[:5] if eid in self._entities]
        
        # Connected components (simplified - just count non-isolated entities)
        connected_entities = len(self._entities) - isolated_entities
        connected_components = 1 if connected_entities > 0 else 0
        
        return KnowledgeGraphStats(
            total_entities=len(entities),
            total_relationships=len(relationships),
            entity_type_counts=entity_type_counts,
            relationship_type_counts=relationship_type_counts,
            avg_entity_confidence=avg_entity_confidence,
            avg_relationship_confidence=avg_relationship_confidence,
            connected_components=connected_components,
            isolated_entities=isolated_entities,
            most_connected_entities=most_connected
        )
    
    def export_graph_data(self) -> Dict[str, Any]:
        """Export complete graph data for serialization or storage."""
        return {
            "entities": [entity.to_dict() for entity in self._entities.values()],
            "relationships": [relationship.to_dict() for relationship in self._relationships.values()],
            "statistics": self.get_graph_statistics().__dict__,
            "export_timestamp": datetime.now().isoformat()
        }
    
    def clear_graph(self):
        """Clear all entities and relationships from the graph."""
        self._entities.clear()
        self._relationships.clear()
        self._entity_relationships.clear()
        self._adjacency_list.clear()
        
        # Clear extractor registries
        self.entity_extractor.clear_entity_registry()
        self.relationship_extractor.clear_relationship_registry()
        
        logger.info("Knowledge graph cleared")
    
    def get_entity_count(self) -> int:
        """Get total number of entities in the graph."""
        return len(self._entities)
    
    def get_relationship_count(self) -> int:
        """Get total number of relationships in the graph."""
        return len(self._relationships)
    
    def has_entity(self, entity_id: str) -> bool:
        """Check if entity exists in the graph."""
        return entity_id in self._entities
    
    def has_relationship(self, relationship_id: str) -> bool:
        """Check if relationship exists in the graph."""
        return relationship_id in self._relationships
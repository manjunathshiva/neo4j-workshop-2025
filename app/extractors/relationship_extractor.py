"""
Relationship extraction using Groq LLM and LangChain.
Implements relationship detection between entities with confidence scoring.
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Import entity types
from .entity_extractor import Entity, EntityType

# Import config
try:
    from ..config import get_config
except ImportError:
    try:
        from config import get_config
    except ImportError:
        # Fallback for testing
        class MockConfig:
            class LLM:
                primary_llm = "groq"
                groq_api_key = "test"
                openai_api_key = None
                anthropic_api_key = None
            llm = LLM()
        
        def get_config():
            return MockConfig()

logger = logging.getLogger(__name__)

class RelationshipType(Enum):
    """Types of relationships that can be extracted."""
    WORKS_FOR = "WORKS_FOR"
    LOCATED_IN = "LOCATED_IN"
    PART_OF = "PART_OF"
    RELATED_TO = "RELATED_TO"
    FOUNDED_BY = "FOUNDED_BY"
    OWNS = "OWNS"
    COLLABORATES_WITH = "COLLABORATES_WITH"
    COMPETES_WITH = "COMPETES_WITH"
    LEADS = "LEADS"
    MEMBER_OF = "MEMBER_OF"
    CREATED_BY = "CREATED_BY"
    USED_BY = "USED_BY"
    INFLUENCES = "INFLUENCES"
    SIMILAR_TO = "SIMILAR_TO"
    OPPOSITE_TO = "OPPOSITE_TO"
    CAUSES = "CAUSES"
    RESULTED_IN = "RESULTED_IN"
    OCCURRED_IN = "OCCURRED_IN"
    ATTENDED_BY = "ATTENDED_BY"

@dataclass
class Relationship:
    """Represents a relationship between two entities."""
    id: str
    source_entity_id: str
    target_entity_id: str
    source_entity_name: str
    target_entity_name: str
    relationship_type: RelationshipType
    description: Optional[str] = None
    confidence: float = 0.0
    context: str = ""
    source_document_id: str = ""
    source_chunk_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary for serialization."""
        return {
            "id": self.id,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "source_entity_name": self.source_entity_name,
            "target_entity_name": self.target_entity_name,
            "relationship_type": self.relationship_type.value,
            "description": self.description,
            "confidence": self.confidence,
            "context": self.context,
            "source_document_id": self.source_document_id,
            "source_chunk_id": self.source_chunk_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class RelationshipExtractionResult:
    """Result of relationship extraction from text."""
    success: bool
    relationships: List[Relationship] = field(default_factory=list)
    error_message: Optional[str] = None
    processing_time_seconds: float = 0.0
    llm_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class RelationshipExtractor:
    """
    Relationship extraction using Groq LLM and LangChain.
    Detects relationships between entities with confidence scoring and validation.
    """
    
    def __init__(self):
        """Initialize relationship extractor with LLM configuration."""
        self.config = get_config()
        self.llm = self._initialize_llm()
        
        # Relationship extraction prompt template
        self.extraction_prompt = PromptTemplate(
            input_variables=["text", "entities", "relationship_types"],
            template="""You are an expert relationship extraction system. Analyze the given text and identify relationships between the provided entities.

ENTITIES IN TEXT:
{entities}

RELATIONSHIP TYPES TO EXTRACT:
{relationship_types}

INSTRUCTIONS:
1. Find relationships between the entities listed above
2. Only extract relationships that are explicitly mentioned or strongly implied in the text
3. For each relationship, provide:
   - source_entity: Name of the first entity (exactly as listed above)
   - target_entity: Name of the second entity (exactly as listed above)
   - relationship_type: One of the specified relationship types
   - description: Brief description of the relationship
   - confidence: Confidence score from 0.0 to 1.0
   - context: The text that supports this relationship

4. Return results as a JSON array with this exact format:
[
  {{
    "source_entity": "Entity Name 1",
    "target_entity": "Entity Name 2", 
    "relationship_type": "RELATIONSHIP_TYPE",
    "description": "brief description of the relationship",
    "confidence": 0.85,
    "context": "text that supports this relationship"
  }}
]

5. IMPORTANT RULES:
   - Only use entity names exactly as they appear in the entities list
   - Only use relationship types from the provided list
   - Confidence should reflect how certain you are about the relationship
   - Don't create relationships that aren't supported by the text
   - Avoid duplicate or redundant relationships

TEXT TO ANALYZE:
{text}

Return only the JSON array, no additional text:"""
        )
        
        # Relationship validation settings
        self.min_confidence = 0.4
        self.max_relationships_per_chunk = 30
        
        # Relationship storage for deduplication
        self._relationship_registry: Dict[str, Relationship] = {}
        
        logger.info(f"RelationshipExtractor initialized with {self.config.llm.primary_llm} LLM")
    
    def _initialize_llm(self):
        """Initialize the appropriate LLM based on configuration."""
        try:
            if self.config.llm.primary_llm == "groq" and self.config.llm.groq_api_key:
                return ChatGroq(
                    groq_api_key=self.config.llm.groq_api_key,
                    model_name="moonshotai/kimi-k2-instruct-0905",  # Good for relationship extraction
                    temperature=0.1,  # Low temperature for consistent extraction
                    max_tokens=2000
                )
            elif self.config.llm.primary_llm == "openai" and self.config.llm.openai_api_key:
                return ChatOpenAI(
                    openai_api_key=self.config.llm.openai_api_key,
                    model_name="gpt-3.5-turbo",
                    temperature=0.1,
                    max_tokens=2000
                )
            elif self.config.llm.primary_llm == "anthropic" and self.config.llm.anthropic_api_key:
                return ChatAnthropic(
                    anthropic_api_key=self.config.llm.anthropic_api_key,
                    model_name="claude-3-haiku-20240307",
                    temperature=0.1,
                    max_tokens=2000
                )
            else:
                # Fallback to Groq if available
                if self.config.llm.groq_api_key:
                    logger.warning(f"Primary LLM {self.config.llm.primary_llm} not available, falling back to Groq")
                    return ChatGroq(
                        groq_api_key=self.config.llm.groq_api_key,
                        model_name="moonshotai/kimi-k2-instruct-0905",
                        temperature=0.1,
                        max_tokens=2000
                    )
                else:
                    raise ValueError("No valid LLM configuration found")
                    
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise
    
    def extract_relationships(
        self,
        text: str,
        entities: List[Entity],
        document_id: str = "",
        chunk_id: str = "",
        relationship_types: Optional[List[RelationshipType]] = None
    ) -> RelationshipExtractionResult:
        """
        Extract relationships between entities in text.
        
        Args:
            text: Text to analyze for relationships
            entities: List of entities to find relationships between
            document_id: Source document identifier
            chunk_id: Source chunk identifier
            relationship_types: List of relationship types to extract (default: common types)
            
        Returns:
            RelationshipExtractionResult with extracted relationships
        """
        start_time = datetime.now()
        
        if not text.strip():
            return RelationshipExtractionResult(
                success=False,
                error_message="Empty text provided"
            )
        
        if len(entities) < 2:
            return RelationshipExtractionResult(
                success=True,
                relationships=[],
                processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                metadata={"message": "Less than 2 entities provided, no relationships possible"}
            )
        
        # Default to common relationship types if none specified
        if relationship_types is None:
            relationship_types = [
                RelationshipType.WORKS_FOR,
                RelationshipType.LOCATED_IN,
                RelationshipType.PART_OF,
                RelationshipType.RELATED_TO,
                RelationshipType.FOUNDED_BY,
                RelationshipType.OWNS,
                RelationshipType.COLLABORATES_WITH,
                RelationshipType.LEADS,
                RelationshipType.MEMBER_OF,
                RelationshipType.CREATED_BY,
                RelationshipType.USED_BY
            ]
        
        try:
            # Prepare entities string for prompt
            entities_str = "\n".join([f"- {entity.name} ({entity.entity_type.value})" 
                                    for entity in entities])
            
            # Prepare relationship types string for prompt
            relationship_types_str = "\n".join([f"- {rt.value}: {self._get_relationship_description(rt)}" 
                                              for rt in relationship_types])
            
            # Create prompt
            prompt = self.extraction_prompt.format(
                text=text,
                entities=entities_str,
                relationship_types=relationship_types_str
            )
            
            # Get LLM response
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            llm_response = response.content
            
            # Parse relationships from response
            relationships = self._parse_llm_response(
                llm_response, entities, document_id, chunk_id
            )
            
            # Apply confidence filtering and validation
            filtered_relationships = self._filter_and_validate_relationships(relationships, entities)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Extracted {len(filtered_relationships)} relationships from text (original: {len(relationships)})")
            
            return RelationshipExtractionResult(
                success=True,
                relationships=filtered_relationships,
                processing_time_seconds=processing_time,
                llm_response=llm_response,
                metadata={
                    "original_relationship_count": len(relationships),
                    "filtered_relationship_count": len(filtered_relationships),
                    "entity_count": len(entities),
                    "text_length": len(text),
                    "relationship_types_requested": [rt.value for rt in relationship_types]
                }
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error extracting relationships: {str(e)}")
            
            return RelationshipExtractionResult(
                success=False,
                error_message=f"Relationship extraction failed: {str(e)}",
                processing_time_seconds=processing_time
            )
    
    def _get_relationship_description(self, relationship_type: RelationshipType) -> str:
        """Get description for relationship type to help LLM understand."""
        descriptions = {
            RelationshipType.WORKS_FOR: "Person works for an organization",
            RelationshipType.LOCATED_IN: "Entity is located in a place",
            RelationshipType.PART_OF: "Entity is part of another entity",
            RelationshipType.RELATED_TO: "General relationship between entities",
            RelationshipType.FOUNDED_BY: "Organization founded by person",
            RelationshipType.OWNS: "Entity owns another entity",
            RelationshipType.COLLABORATES_WITH: "Entities work together",
            RelationshipType.COMPETES_WITH: "Entities compete with each other",
            RelationshipType.LEADS: "Person leads an organization or project",
            RelationshipType.MEMBER_OF: "Person is member of organization",
            RelationshipType.CREATED_BY: "Entity created by another entity",
            RelationshipType.USED_BY: "Entity used by another entity",
            RelationshipType.INFLUENCES: "Entity influences another entity",
            RelationshipType.SIMILAR_TO: "Entities are similar",
            RelationshipType.OPPOSITE_TO: "Entities are opposite",
            RelationshipType.CAUSES: "Entity causes another entity/event",
            RelationshipType.RESULTED_IN: "Entity resulted in another entity/event",
            RelationshipType.OCCURRED_IN: "Event occurred in a location",
            RelationshipType.ATTENDED_BY: "Event attended by person/organization"
        }
        return descriptions.get(relationship_type, "General relationship")
    
    def _parse_llm_response(
        self,
        response: str,
        entities: List[Entity],
        document_id: str,
        chunk_id: str
    ) -> List[Relationship]:
        """
        Parse LLM response to extract relationships.
        
        Args:
            response: Raw LLM response
            entities: List of entities for validation
            document_id: Source document ID
            chunk_id: Source chunk ID
            
        Returns:
            List of Relationship objects
        """
        relationships = []
        entity_names = {entity.name.lower(): entity for entity in entities}
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                logger.warning("No JSON array found in LLM response")
                return relationships
            
            json_str = json_match.group(0)
            relationship_data = json.loads(json_str)
            
            if not isinstance(relationship_data, list):
                logger.warning("LLM response is not a JSON array")
                return relationships
            
            for i, rel_dict in enumerate(relationship_data):
                try:
                    # Validate required fields
                    required_fields = ['source_entity', 'target_entity', 'relationship_type']
                    if not all(key in rel_dict for key in required_fields):
                        logger.warning(f"Relationship {i} missing required fields")
                        continue
                    
                    # Validate entity names
                    source_name = rel_dict['source_entity'].strip()
                    target_name = rel_dict['target_entity'].strip()
                    
                    source_entity = self._find_entity_by_name(source_name, entities)
                    target_entity = self._find_entity_by_name(target_name, entities)
                    
                    if not source_entity or not target_entity:
                        logger.warning(f"Relationship {i} references unknown entities: {source_name}, {target_name}")
                        continue
                    
                    # Parse relationship type
                    try:
                        relationship_type = RelationshipType(rel_dict['relationship_type'])
                    except ValueError:
                        logger.warning(f"Invalid relationship type: {rel_dict['relationship_type']}")
                        continue
                    
                    # Create relationship ID
                    relationship_id = self._generate_relationship_id(
                        source_entity.id, target_entity.id, relationship_type, document_id
                    )
                    
                    # Create relationship object
                    relationship = Relationship(
                        id=relationship_id,
                        source_entity_id=source_entity.id,
                        target_entity_id=target_entity.id,
                        source_entity_name=source_entity.name,
                        target_entity_name=target_entity.name,
                        relationship_type=relationship_type,
                        description=rel_dict.get('description', '').strip(),
                        confidence=float(rel_dict.get('confidence', 0.5)),
                        context=rel_dict.get('context', '').strip(),
                        source_document_id=document_id,
                        source_chunk_id=chunk_id,
                        metadata={
                            "extraction_method": "llm",
                            "llm_provider": self.config.llm.primary_llm,
                            "source_entity_type": source_entity.entity_type.value,
                            "target_entity_type": target_entity.entity_type.value
                        }
                    )
                    
                    relationships.append(relationship)
                    
                except Exception as e:
                    logger.warning(f"Error parsing relationship {i}: {str(e)}")
                    continue
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from LLM response: {str(e)}")
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
        
        return relationships
    
    def _find_entity_by_name(self, name: str, entities: List[Entity]) -> Optional[Entity]:
        """Find entity by name with fuzzy matching."""
        name_lower = name.lower().strip()
        
        # Exact match first
        for entity in entities:
            if entity.name.lower().strip() == name_lower:
                return entity
        
        # Partial match
        for entity in entities:
            entity_name_lower = entity.name.lower().strip()
            if name_lower in entity_name_lower or entity_name_lower in name_lower:
                return entity
        
        return None
    
    def _generate_relationship_id(
        self,
        source_id: str,
        target_id: str,
        relationship_type: RelationshipType,
        document_id: str
    ) -> str:
        """Generate a unique ID for a relationship."""
        # Create base ID from entity IDs and relationship type
        base_id = f"{source_id}_{relationship_type.value.lower()}_{target_id}"
        
        # Add document prefix if provided
        if document_id:
            doc_prefix = document_id.split('_')[0] if '_' in document_id else document_id[:8]
            base_id = f"{doc_prefix}_{base_id}"
        
        return base_id
    
    def _filter_and_validate_relationships(
        self,
        relationships: List[Relationship],
        entities: List[Entity]
    ) -> List[Relationship]:
        """
        Filter relationships by confidence and validate consistency.
        
        Args:
            relationships: List of relationships to filter
            entities: List of entities for validation
            
        Returns:
            Filtered and validated list of relationships
        """
        if not relationships:
            return relationships
        
        # Filter by confidence
        confident_relationships = [r for r in relationships if r.confidence >= self.min_confidence]
        
        # Remove self-relationships (entity related to itself)
        valid_relationships = [r for r in confident_relationships 
                             if r.source_entity_id != r.target_entity_id]
        
        # Deduplicate similar relationships
        deduplicated_relationships = self._deduplicate_relationships(valid_relationships)
        
        # Limit number of relationships per chunk
        if len(deduplicated_relationships) > self.max_relationships_per_chunk:
            # Sort by confidence and take top relationships
            deduplicated_relationships.sort(key=lambda x: x.confidence, reverse=True)
            deduplicated_relationships = deduplicated_relationships[:self.max_relationships_per_chunk]
        
        return deduplicated_relationships
    
    def _deduplicate_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """
        Deduplicate similar relationships within the same extraction.
        
        Args:
            relationships: List of relationships to deduplicate
            
        Returns:
            Deduplicated list of relationships
        """
        if len(relationships) <= 1:
            return relationships
        
        deduplicated = []
        
        for relationship in relationships:
            is_duplicate = False
            
            for existing in deduplicated:
                if self._are_relationships_similar(relationship, existing):
                    # Keep the relationship with higher confidence
                    if relationship.confidence > existing.confidence:
                        # Replace existing with current relationship
                        deduplicated.remove(existing)
                        deduplicated.append(relationship)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(relationship)
        
        return deduplicated
    
    def _are_relationships_similar(self, rel1: Relationship, rel2: Relationship) -> bool:
        """
        Check if two relationships are similar enough to be considered duplicates.
        
        Args:
            rel1: First relationship
            rel2: Second relationship
            
        Returns:
            True if relationships are similar
        """
        # Same entities and relationship type
        if (rel1.source_entity_id == rel2.source_entity_id and
            rel1.target_entity_id == rel2.target_entity_id and
            rel1.relationship_type == rel2.relationship_type):
            return True
        
        # Reverse relationship with compatible type
        if (rel1.source_entity_id == rel2.target_entity_id and
            rel1.target_entity_id == rel2.source_entity_id):
            # Check if relationship types are compatible in reverse
            if self._are_relationship_types_compatible(rel1.relationship_type, rel2.relationship_type):
                return True
        
        return False
    
    def _are_relationship_types_compatible(self, type1: RelationshipType, type2: RelationshipType) -> bool:
        """Check if two relationship types are compatible (e.g., WORKS_FOR and EMPLOYS)."""
        # Define compatible relationship pairs
        compatible_pairs = {
            (RelationshipType.WORKS_FOR, RelationshipType.OWNS),
            (RelationshipType.PART_OF, RelationshipType.OWNS),
            (RelationshipType.MEMBER_OF, RelationshipType.OWNS),
            (RelationshipType.CREATED_BY, RelationshipType.OWNS),
            (RelationshipType.FOUNDED_BY, RelationshipType.OWNS)
        }
        
        return (type1, type2) in compatible_pairs or (type2, type1) in compatible_pairs
    
    def deduplicate_across_documents(self, new_relationships: List[Relationship]) -> List[Relationship]:
        """
        Deduplicate relationships across all processed documents.
        
        Args:
            new_relationships: New relationships to check against registry
            
        Returns:
            List of relationships after cross-document deduplication
        """
        deduplicated = []
        
        for relationship in new_relationships:
            # Check against existing relationships in registry
            existing_relationship = self._find_similar_relationship_in_registry(relationship)
            
            if existing_relationship:
                # Update existing relationship with new information
                self._merge_relationship_information(existing_relationship, relationship)
                deduplicated.append(existing_relationship)
            else:
                # Add new relationship to registry
                self._relationship_registry[relationship.id] = relationship
                deduplicated.append(relationship)
        
        return deduplicated
    
    def _find_similar_relationship_in_registry(self, relationship: Relationship) -> Optional[Relationship]:
        """Find similar relationship in the global registry."""
        for existing_relationship in self._relationship_registry.values():
            if self._are_relationships_similar(relationship, existing_relationship):
                return existing_relationship
        return None
    
    def _merge_relationship_information(self, existing: Relationship, new: Relationship):
        """Merge information from new relationship into existing relationship."""
        # Update confidence with weighted average
        total_confidence = (existing.confidence + new.confidence) / 2
        existing.confidence = min(total_confidence, 1.0)
        
        # Merge descriptions
        if new.description and new.description not in existing.description:
            if existing.description:
                existing.description += f"; {new.description}"
            else:
                existing.description = new.description
        
        # Add source document to metadata
        if "source_documents" not in existing.metadata:
            existing.metadata["source_documents"] = [existing.source_document_id]
        
        if new.source_document_id not in existing.metadata["source_documents"]:
            existing.metadata["source_documents"].append(new.source_document_id)
    
    def get_relationship_registry(self) -> Dict[str, Relationship]:
        """Get the current relationship registry for inspection."""
        return self._relationship_registry.copy()
    
    def clear_relationship_registry(self):
        """Clear the relationship registry (for testing or reset)."""
        self._relationship_registry.clear()
        logger.info("Relationship registry cleared")
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about relationship extraction."""
        if not self._relationship_registry:
            return {"total_relationships": 0}
        
        relationships = list(self._relationship_registry.values())
        
        # Count by type
        type_counts = {}
        for rel_type in RelationshipType:
            type_counts[rel_type.value] = sum(1 for r in relationships if r.relationship_type == rel_type)
        
        # Calculate confidence statistics
        confidences = [r.confidence for r in relationships]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "total_relationships": len(relationships),
            "type_counts": type_counts,
            "avg_confidence": avg_confidence,
            "min_confidence": min(confidences) if confidences else 0.0,
            "max_confidence": max(confidences) if confidences else 0.0,
            "llm_provider": self.config.llm.primary_llm
        }
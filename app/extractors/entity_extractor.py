"""
Entity extraction using Groq LLM and LangChain.
Implements named entity recognition for persons, organizations, locations, and concepts.
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

class EntityType(Enum):
    """Types of entities that can be extracted."""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    CONCEPT = "CONCEPT"
    EVENT = "EVENT"
    PRODUCT = "PRODUCT"
    DATE = "DATE"
    MONEY = "MONEY"
    MISC = "MISC"

@dataclass
class Entity:
    """Represents an extracted entity."""
    id: str
    name: str
    entity_type: EntityType
    description: Optional[str] = None
    confidence: float = 0.0
    context: str = ""
    source_document_id: str = ""
    source_chunk_id: str = ""
    start_char: int = -1
    end_char: int = -1
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "description": self.description,
            "confidence": self.confidence,
            "context": self.context,
            "source_document_id": self.source_document_id,
            "source_chunk_id": self.source_chunk_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class EntityExtractionResult:
    """Result of entity extraction from text."""
    success: bool
    entities: List[Entity] = field(default_factory=list)
    error_message: Optional[str] = None
    processing_time_seconds: float = 0.0
    llm_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class EntityExtractor:
    """
    Entity extraction using Groq LLM and LangChain.
    Implements named entity recognition with confidence scoring and deduplication.
    """
    
    def __init__(self):
        """Initialize entity extractor with LLM configuration."""
        self.config = get_config()
        self.llm = self._initialize_llm()
        
        # Entity extraction prompt template
        self.extraction_prompt = PromptTemplate(
            input_variables=["text", "entity_types"],
            template="""You are an expert named entity recognition system. Extract entities from the given text and classify them into the specified types.

ENTITY TYPES TO EXTRACT:
{entity_types}

INSTRUCTIONS:
1. Extract ALL entities that belong to the specified types
2. For each entity, provide:
   - name: The exact text of the entity as it appears
   - type: One of the specified entity types
   - description: Brief description of what this entity represents
   - confidence: Confidence score from 0.0 to 1.0
   - context: The surrounding text that provides context

3. Return results as a JSON array with this exact format:
[
  {{
    "name": "entity name",
    "type": "ENTITY_TYPE",
    "description": "brief description",
    "confidence": 0.95,
    "context": "surrounding text context"
  }}
]

CRITICAL: Your response must be ONLY the JSON array. Do not include any explanatory text, markdown formatting, or code blocks. Start with [ and end with ].

4. IMPORTANT RULES:
   - Only extract entities that clearly belong to the specified types
   - Avoid extracting common words unless they are clearly entities
   - For PERSON: Extract full names, not just first names or titles
   - For ORGANIZATION: Include companies, institutions, government bodies
   - For LOCATION: Include cities, countries, regions, landmarks
   - For CONCEPT: Include important ideas, theories, methodologies
   - Confidence should reflect how certain you are about the classification

TEXT TO ANALYZE:
{text}

Return only the JSON array, no additional text:"""
        )
        
        # Deduplication settings
        self.similarity_threshold = 0.85
        self.max_entities_per_chunk = 50
        
        # Entity storage for deduplication across documents
        self._entity_registry: Dict[str, Entity] = {}
        
        logger.info(f"EntityExtractor initialized with {self.config.llm.primary_llm} LLM")
    
    def _initialize_llm(self):
        """Initialize the appropriate LLM based on configuration."""
        try:
            if self.config.llm.primary_llm == "groq" and self.config.llm.groq_api_key:
                return ChatGroq(
                    groq_api_key=self.config.llm.groq_api_key,
                    model_name="moonshotai/kimi-k2-instruct-0905",  # Good for NER tasks
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
    
    def extract_entities(
        self, 
        text: str, 
        document_id: str = "", 
        chunk_id: str = "",
        entity_types: Optional[List[EntityType]] = None
    ) -> EntityExtractionResult:
        """
        Extract entities from text using LLM.
        
        Args:
            text: Text to extract entities from
            document_id: Source document identifier
            chunk_id: Source chunk identifier
            entity_types: List of entity types to extract (default: all types)
            
        Returns:
            EntityExtractionResult with extracted entities
        """
        start_time = datetime.now()
        
        if not text.strip():
            return EntityExtractionResult(
                success=False,
                error_message="Empty text provided"
            )
        
        # Default to all entity types if none specified
        if entity_types is None:
            entity_types = [
                EntityType.PERSON,
                EntityType.ORGANIZATION, 
                EntityType.LOCATION,
                EntityType.CONCEPT,
                EntityType.EVENT
            ]
        
        try:
            # Prepare entity types string for prompt
            entity_types_str = "\n".join([f"- {et.value}: {self._get_entity_type_description(et)}" 
                                        for et in entity_types])
            
            # Create prompt
            prompt = self.extraction_prompt.format(
                text=text,
                entity_types=entity_types_str
            )
            
            # Get LLM response
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            llm_response = response.content
            
            # Parse entities from response
            entities = self._parse_llm_response(
                llm_response, text, document_id, chunk_id
            )
            
            # Apply confidence filtering and deduplication
            filtered_entities = self._filter_and_deduplicate_entities(entities, text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Extracted {len(filtered_entities)} entities from text (original: {len(entities)})")
            
            return EntityExtractionResult(
                success=True,
                entities=filtered_entities,
                processing_time_seconds=processing_time,
                llm_response=llm_response,
                metadata={
                    "original_entity_count": len(entities),
                    "filtered_entity_count": len(filtered_entities),
                    "text_length": len(text),
                    "entity_types_requested": [et.value for et in entity_types]
                }
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error extracting entities: {str(e)}")
            
            return EntityExtractionResult(
                success=False,
                error_message=f"Entity extraction failed: {str(e)}",
                processing_time_seconds=processing_time
            )
    
    def _get_entity_type_description(self, entity_type: EntityType) -> str:
        """Get description for entity type to help LLM understand what to extract."""
        descriptions = {
            EntityType.PERSON: "Names of people, including full names, first names with context",
            EntityType.ORGANIZATION: "Companies, institutions, government bodies, NGOs, universities",
            EntityType.LOCATION: "Cities, countries, regions, landmarks, geographical locations",
            EntityType.CONCEPT: "Important ideas, theories, methodologies, abstract concepts",
            EntityType.EVENT: "Named events, conferences, historical events, meetings",
            EntityType.PRODUCT: "Products, services, software, tools, brands",
            EntityType.DATE: "Specific dates, time periods, years",
            EntityType.MONEY: "Monetary amounts, currencies, financial figures",
            EntityType.MISC: "Other important entities not covered by above categories"
        }
        return descriptions.get(entity_type, "Miscellaneous entities")
    
    def _parse_llm_response(
        self, 
        response: str, 
        original_text: str, 
        document_id: str, 
        chunk_id: str
    ) -> List[Entity]:
        """
        Parse LLM response to extract entities.
        
        Args:
            response: Raw LLM response
            original_text: Original text that was analyzed
            document_id: Source document ID
            chunk_id: Source chunk ID
            
        Returns:
            List of Entity objects
        """
        entities = []
        
        try:
            # Try to extract JSON from response
            logger.info(f"LLM response length: {len(response)} characters")
            logger.debug(f"Full LLM response for entity extraction: {response}")
            
            # First, try to parse the response directly as JSON (in case it's pure JSON)
            json_str = response.strip()
            
            # If that doesn't work, try to extract JSON from the response
            if not json_str.startswith('['):
                logger.debug("Response doesn't start with '[', trying regex patterns...")
                
                # Try multiple patterns to find JSON array
                json_patterns = [
                    r'(\[[\s\S]*\])',  # Greedy pattern to capture full array
                    r'```json\s*(\[[\s\S]*?\])\s*```',  # JSON in code blocks
                    r'```\s*(\[[\s\S]*?\])\s*```',  # Array in code blocks
                    r'\[[\s\S]*\]',  # Simple greedy pattern
                ]
                
                json_match = None
                for i, pattern in enumerate(json_patterns):
                    json_match = re.search(pattern, response, re.DOTALL)
                    if json_match:
                        logger.debug(f"JSON pattern {i} matched: {pattern}")
                        break
                    else:
                        logger.debug(f"JSON pattern {i} failed: {pattern}")
                
                if not json_match:
                    logger.warning(f"No JSON array found in LLM response. Response: {response[:200]}...")
                    # Try fallback extraction immediately
                    return self._fallback_entity_extraction(response, original_text, document_id, chunk_id)
                
                # Get the JSON string (use group 1 if it exists, otherwise group 0)
                json_str = json_match.group(1) if json_match.lastindex and json_match.lastindex >= 1 else json_match.group(0)
            
            logger.info(f"Attempting to parse JSON: {json_str[:200]}...")
            logger.debug(f"JSON string length: {len(json_str)}")
            logger.debug(f"JSON string repr: {repr(json_str[:100])}")
            entity_data = json.loads(json_str)
            
            if not isinstance(entity_data, list):
                logger.warning("LLM response is not a JSON array")
                return entities
            
            for i, entity_dict in enumerate(entity_data):
                try:
                    # Validate required fields
                    if not all(key in entity_dict for key in ['name', 'type']):
                        logger.warning(f"Entity {i} missing required fields")
                        continue
                    
                    # Parse entity type
                    try:
                        entity_type = EntityType(entity_dict['type'])
                    except ValueError:
                        logger.warning(f"Invalid entity type: {entity_dict['type']}")
                        continue
                    
                    # Find entity position in original text
                    entity_name = entity_dict['name'].strip()
                    start_char, end_char = self._find_entity_position(entity_name, original_text)
                    
                    # Create entity ID
                    entity_id = self._generate_entity_id(entity_name, entity_type, document_id)
                    
                    # Create entity object
                    entity = Entity(
                        id=entity_id,
                        name=entity_name,
                        entity_type=entity_type,
                        description=entity_dict.get('description', '').strip(),
                        confidence=float(entity_dict.get('confidence', 0.5)),
                        context=entity_dict.get('context', '').strip(),
                        source_document_id=document_id,
                        source_chunk_id=chunk_id,
                        start_char=start_char,
                        end_char=end_char,
                        metadata={
                            "extraction_method": "llm",
                            "llm_provider": self.config.llm.primary_llm
                        }
                    )
                    
                    entities.append(entity)
                    
                except Exception as e:
                    logger.warning(f"Error parsing entity {i}: {str(e)}")
                    continue
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from LLM response: {str(e)}")
            logger.debug(f"Attempted to parse: {json_str[:200]}...")
            
            # Try to fix truncated JSON by adding closing brackets
            if "Unterminated string" in str(e) or "Expecting" in str(e):
                logger.info("Attempting to fix truncated JSON...")
                fixed_json = self._fix_truncated_json(json_str)
                if fixed_json:
                    try:
                        entity_data = json.loads(fixed_json)
                        if isinstance(entity_data, list):
                            logger.info(f"Successfully parsed fixed JSON with {len(entity_data)} entities")
                            # Continue with the normal parsing logic
                            for i, entity_dict in enumerate(entity_data):
                                try:
                                    # Validate required fields
                                    if not all(key in entity_dict for key in ['name', 'type']):
                                        logger.warning(f"Entity {i} missing required fields")
                                        continue
                                    
                                    # Parse entity type
                                    try:
                                        entity_type = EntityType(entity_dict['type'])
                                    except ValueError:
                                        logger.warning(f"Invalid entity type: {entity_dict['type']}")
                                        continue
                                    
                                    # Create entity
                                    entity_name = entity_dict['name'].strip()
                                    start_char, end_char = self._find_entity_position(entity_name, original_text)
                                    
                                    # Create entity ID
                                    entity_id = self._generate_entity_id(entity_name, entity_type, document_id)
                                    
                                    # Create entity object
                                    entity = Entity(
                                        id=entity_id,
                                        name=entity_name,
                                        entity_type=entity_type,
                                        description=entity_dict.get('description', '').strip(),
                                        confidence=float(entity_dict.get('confidence', 0.5)),
                                        context=entity_dict.get('context', '').strip(),
                                        source_document_id=document_id,
                                        source_chunk_id=chunk_id,
                                        start_char=start_char,
                                        end_char=end_char,
                                        metadata={
                                            "extraction_method": "llm_fixed",
                                            "llm_provider": self.config.llm.primary_llm
                                        }
                                    )
                                    
                                    entities.append(entity)
                                    
                                except Exception as e:
                                    logger.warning(f"Error parsing entity {i}: {str(e)}")
                                    continue
                            
                            return entities
                    except json.JSONDecodeError:
                        logger.warning("Fixed JSON still invalid, trying fallback extraction")
            
            # Try fallback extraction
            return self._fallback_entity_extraction(response, original_text, document_id, chunk_id)
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
        
        return entities
    
    def _fix_truncated_json(self, json_str: str) -> str:
        """
        Attempt to fix truncated JSON by adding missing closing brackets.
        """
        try:
            # Count opening and closing brackets
            open_brackets = json_str.count('[')
            close_brackets = json_str.count(']')
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            
            # If we have unmatched brackets, try to fix
            fixed = json_str
            
            # If the string ends abruptly, try to close it properly
            if fixed.endswith(',') or fixed.endswith('"'):
                # Remove trailing comma if present
                if fixed.endswith(','):
                    fixed = fixed[:-1]
                
                # Add missing closing braces and brackets
                missing_braces = open_braces - close_braces
                missing_brackets = open_brackets - close_brackets
                
                for _ in range(missing_braces):
                    fixed += '}'
                for _ in range(missing_brackets):
                    fixed += ']'
            
            # Try to parse to validate
            json.loads(fixed)
            return fixed
            
        except Exception as e:
            logger.debug(f"Could not fix truncated JSON: {str(e)}")
            return None
    
    def _fallback_entity_extraction(self, response: str, text: str, document_id: str, chunk_id: str) -> List[Entity]:
        """
        Fallback entity extraction when JSON parsing fails.
        Tries to extract entities from structured text response.
        """
        entities = []
        logger.info("Attempting fallback entity extraction from text response")
        
        try:
            # Look for patterns like "Name: John Doe, Type: PERSON"
            lines = response.split('\n')
            current_entity = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Try to parse key-value pairs
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip().strip('"').strip("'")
                    
                    if key in ['name', 'entity', 'person', 'organization', 'location']:
                        current_entity['name'] = value
                    elif key in ['type', 'category']:
                        current_entity['type'] = value.upper()
                    elif key in ['description', 'desc']:
                        current_entity['description'] = value
                    elif key in ['confidence', 'score']:
                        try:
                            current_entity['confidence'] = float(value)
                        except:
                            current_entity['confidence'] = 0.7
                
                # If we have enough info, create an entity
                if 'name' in current_entity and 'type' in current_entity:
                    try:
                        entity_type = EntityType(current_entity['type'])
                        entity_name = current_entity['name']
                        
                        # Create entity
                        entity_id = self._generate_entity_id(entity_name, entity_type, document_id)
                        start_char, end_char = self._find_entity_position(entity_name, text)
                        
                        entity = Entity(
                            id=entity_id,
                            name=entity_name,
                            entity_type=entity_type,
                            description=current_entity.get('description', ''),
                            confidence=current_entity.get('confidence', 0.7),
                            context='',
                            source_document_id=document_id,
                            source_chunk_id=chunk_id,
                            start_char=start_char,
                            end_char=end_char,
                            metadata={
                                "extraction_method": "fallback",
                                "llm_provider": self.config.llm.primary_llm
                            }
                        )
                        
                        entities.append(entity)
                        logger.info(f"Fallback extracted entity: {entity_name} ({entity_type.value})")
                        
                    except ValueError:
                        logger.warning(f"Invalid entity type in fallback: {current_entity.get('type')}")
                    except Exception as e:
                        logger.warning(f"Error creating fallback entity: {str(e)}")
                    
                    # Reset for next entity
                    current_entity = {}
            
            if entities:
                logger.info(f"Fallback extraction found {len(entities)} entities")
            else:
                logger.warning("Fallback extraction found no entities - trying simple pattern matching")
                # Last resort: simple pattern matching for common entities
                entities = self._simple_pattern_extraction(text, document_id, chunk_id)
                
        except Exception as e:
            logger.error(f"Error in fallback entity extraction: {str(e)}")
        
        return entities
    
    def _simple_pattern_extraction(self, text: str, document_id: str, chunk_id: str) -> List[Entity]:
        """
        Simple pattern-based entity extraction as last resort.
        """
        entities = []
        
        try:
            # Look for capitalized words that might be names
            import re
            
            # Pattern for potential person names (2+ capitalized words)
            name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
            potential_names = re.findall(name_pattern, text)
            
            # Pattern for organizations (words with Corp, Inc, LLC, etc.)
            org_pattern = r'\b[A-Z][a-zA-Z\s]*(?:Corp|Inc|LLC|Ltd|Company|Organization|University|College|Institute)\b'
            potential_orgs = re.findall(org_pattern, text)
            
            # Pattern for locations (common location indicators)
            location_pattern = r'\b[A-Z][a-zA-Z\s]*(?:City|State|Country|Street|Avenue|Road|University|College)\b'
            potential_locations = re.findall(location_pattern, text)
            
            # Create entities from patterns
            for name in potential_names[:5]:  # Limit to avoid noise
                if len(name.split()) >= 2:  # At least first and last name
                    entity_id = self._generate_entity_id(name, EntityType.PERSON, document_id)
                    start_char, end_char = self._find_entity_position(name, text)
                    
                    entity = Entity(
                        id=entity_id,
                        name=name,
                        entity_type=EntityType.PERSON,
                        description=f"Person mentioned in document",
                        confidence=0.6,  # Lower confidence for pattern matching
                        context=f"Person mentioned in document: {name}",
                        source_document_id=document_id,
                        source_chunk_id=chunk_id,
                        start_char=start_char,
                        end_char=end_char,
                        metadata={
                            "extraction_method": "pattern_matching",
                            "llm_provider": self.config.llm.primary_llm
                        }
                    )
                    entities.append(entity)
            
            for org in potential_orgs[:3]:  # Limit organizations
                entity_id = self._generate_entity_id(org, EntityType.ORGANIZATION, document_id)
                start_char, end_char = self._find_entity_position(org, text)
                
                entity = Entity(
                    id=entity_id,
                    name=org,
                    entity_type=EntityType.ORGANIZATION,
                    description=f"Organization mentioned in document",
                    confidence=0.6,
                    context=f"Organization mentioned in document: {org}",
                    source_document_id=document_id,
                    source_chunk_id=chunk_id,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={
                        "extraction_method": "pattern_matching",
                        "llm_provider": self.config.llm.primary_llm
                    }
                )
                entities.append(entity)
            
            if entities:
                logger.info(f"Pattern matching found {len(entities)} entities")
            
        except Exception as e:
            logger.error(f"Error in pattern extraction: {str(e)}")
        
        return entities
    
    def _find_entity_position(self, entity_name: str, text: str) -> Tuple[int, int]:
        """
        Find the position of an entity in the original text.
        
        Args:
            entity_name: Name of the entity to find
            text: Original text to search in
            
        Returns:
            Tuple of (start_char, end_char) positions
        """
        # Try exact match first
        start_pos = text.find(entity_name)
        if start_pos != -1:
            return start_pos, start_pos + len(entity_name)
        
        # Try case-insensitive match
        start_pos = text.lower().find(entity_name.lower())
        if start_pos != -1:
            return start_pos, start_pos + len(entity_name)
        
        # Try partial matches (for cases where LLM extracted part of a longer phrase)
        words = entity_name.split()
        if len(words) > 1:
            for word in words:
                if len(word) > 3:  # Only search for meaningful words
                    start_pos = text.find(word)
                    if start_pos != -1:
                        return start_pos, start_pos + len(word)
        
        # Return -1 if not found
        return -1, -1
    
    def _generate_entity_id(self, name: str, entity_type: EntityType, document_id: str) -> str:
        """Generate a unique ID for an entity."""
        # Create a normalized name for ID generation
        normalized_name = re.sub(r'[^a-zA-Z0-9]', '_', name.lower())
        normalized_name = re.sub(r'_+', '_', normalized_name).strip('_')
        
        # Create base ID
        base_id = f"{entity_type.value.lower()}_{normalized_name}"
        
        # Add document prefix if provided
        if document_id:
            doc_prefix = document_id.split('_')[0] if '_' in document_id else document_id[:8]
            base_id = f"{doc_prefix}_{base_id}"
        
        return base_id
    
    def _filter_and_deduplicate_entities(self, entities: List[Entity], text: str) -> List[Entity]:
        """
        Filter entities by confidence and deduplicate similar entities.
        
        Args:
            entities: List of entities to filter
            text: Original text for context
            
        Returns:
            Filtered and deduplicated list of entities
        """
        if not entities:
            return entities
        
        # Filter by confidence (minimum 0.3)
        confident_entities = [e for e in entities if e.confidence >= 0.3]
        
        # Filter out very short entities (less than 2 characters)
        meaningful_entities = [e for e in confident_entities if len(e.name.strip()) >= 2]
        
        # Filter out common stop words that might be misclassified
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        filtered_entities = [e for e in meaningful_entities 
                           if e.name.lower().strip() not in stop_words]
        
        # Deduplicate similar entities
        deduplicated_entities = self._deduplicate_entities(filtered_entities)
        
        # Limit number of entities per chunk to prevent overwhelming
        if len(deduplicated_entities) > self.max_entities_per_chunk:
            # Sort by confidence and take top entities
            deduplicated_entities.sort(key=lambda x: x.confidence, reverse=True)
            deduplicated_entities = deduplicated_entities[:self.max_entities_per_chunk]
        
        return deduplicated_entities
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Deduplicate similar entities within the same extraction.
        
        Args:
            entities: List of entities to deduplicate
            
        Returns:
            Deduplicated list of entities
        """
        if len(entities) <= 1:
            return entities
        
        deduplicated = []
        
        for entity in entities:
            is_duplicate = False
            
            for existing in deduplicated:
                if self._are_entities_similar(entity, existing):
                    # Keep the entity with higher confidence
                    if entity.confidence > existing.confidence:
                        # Replace existing with current entity
                        deduplicated.remove(existing)
                        deduplicated.append(entity)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(entity)
        
        return deduplicated
    
    def _are_entities_similar(self, entity1: Entity, entity2: Entity) -> bool:
        """
        Check if two entities are similar enough to be considered duplicates.
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            True if entities are similar
        """
        # Must be same type
        if entity1.entity_type != entity2.entity_type:
            return False
        
        # Check name similarity
        name1 = entity1.name.lower().strip()
        name2 = entity2.name.lower().strip()
        
        # Exact match
        if name1 == name2:
            return True
        
        # One name contains the other
        if name1 in name2 or name2 in name1:
            return True
        
        # Calculate simple similarity score
        similarity = self._calculate_string_similarity(name1, name2)
        return similarity >= self.similarity_threshold
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings using simple character overlap.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not str1 or not str2:
            return 0.0
        
        # Convert to sets of characters
        set1 = set(str1.lower())
        set2 = set(str2.lower())
        
        # Calculate Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def deduplicate_across_documents(self, new_entities: List[Entity]) -> List[Entity]:
        """
        Deduplicate entities across all processed documents.
        
        Args:
            new_entities: New entities to check against registry
            
        Returns:
            List of entities after cross-document deduplication
        """
        deduplicated = []
        
        for entity in new_entities:
            # Check against existing entities in registry
            existing_entity = self._find_similar_entity_in_registry(entity)
            
            if existing_entity:
                # Update existing entity with new information
                self._merge_entity_information(existing_entity, entity)
                deduplicated.append(existing_entity)
            else:
                # Add new entity to registry
                self._entity_registry[entity.id] = entity
                deduplicated.append(entity)
        
        return deduplicated
    
    def _find_similar_entity_in_registry(self, entity: Entity) -> Optional[Entity]:
        """Find similar entity in the global registry."""
        for existing_entity in self._entity_registry.values():
            if self._are_entities_similar(entity, existing_entity):
                return existing_entity
        return None
    
    def _merge_entity_information(self, existing: Entity, new: Entity):
        """Merge information from new entity into existing entity."""
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
    
    def get_entity_registry(self) -> Dict[str, Entity]:
        """Get the current entity registry for inspection."""
        return self._entity_registry.copy()
    
    def clear_entity_registry(self):
        """Clear the entity registry (for testing or reset)."""
        self._entity_registry.clear()
        logger.info("Entity registry cleared")
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about entity extraction."""
        if not self._entity_registry:
            return {"total_entities": 0}
        
        entities = list(self._entity_registry.values())
        
        # Count by type
        type_counts = {}
        for entity_type in EntityType:
            type_counts[entity_type.value] = sum(1 for e in entities if e.entity_type == entity_type)
        
        # Calculate confidence statistics
        confidences = [e.confidence for e in entities]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "total_entities": len(entities),
            "type_counts": type_counts,
            "avg_confidence": avg_confidence,
            "min_confidence": min(confidences) if confidences else 0.0,
            "max_confidence": max(confidences) if confidences else 0.0,
            "llm_provider": self.config.llm.primary_llm
        }
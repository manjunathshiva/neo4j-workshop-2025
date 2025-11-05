"""
Text-to-Cypher generation using LLM for Graph RAG.
Converts natural language questions to Cypher queries.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

try:
    from ..config import get_config
    from ..graph.neo4j_schema import Neo4jSchema
    from ..graph.cypher_builder import build_text_to_cypher_prompt
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import get_config
    from graph.neo4j_schema import Neo4jSchema
    from graph.cypher_builder import build_text_to_cypher_prompt

logger = logging.getLogger(__name__)

@dataclass
class CypherGenerationResult:
    """Result of Cypher query generation."""
    success: bool
    cypher_query: str = ""
    parameters: Dict[str, Any] = None
    confidence: float = 0.0
    reasoning: str = ""
    error_message: str = ""
    processing_time_seconds: float = 0.0
    model_used: str = ""

class TextToCypherGenerator:
    """
    Generates Cypher queries from natural language using LLMs.
    Supports multiple LLM providers with schema-aware prompting.
    """
    
    def __init__(self):
        """Initialize Text-to-Cypher generator."""
        self.config = get_config()
        self.schema = Neo4jSchema()
        self._llm = None
        self._model_name = ""
        
        # Initialize LLM based on configuration
        self._initialize_llm()
        
        # Query examples for few-shot prompting
        self._examples = self._get_query_examples()
        
        logger.info(f"TextToCypherGenerator initialized with {self._model_name}")
    
    def _initialize_llm(self):
        """Initialize the LLM based on configuration."""
        try:
            primary_llm = self.config.llm.primary_llm.lower()
            
            if primary_llm == "groq" and self.config.llm.groq_api_key:
                self._llm = ChatGroq(
                    groq_api_key=self.config.llm.groq_api_key,
                    model_name=self.config.llm.text_to_cypher_model,
                    temperature=0.1,  # Low temperature for precise code generation
                    max_tokens=1000
                )
                self._model_name = f"Groq ({self.config.llm.text_to_cypher_model})"
                
            elif primary_llm == "openai" and self.config.llm.openai_api_key:
                self._llm = ChatOpenAI(
                    openai_api_key=self.config.llm.openai_api_key,
                    model_name="gpt-3.5-turbo",
                    temperature=0.1,
                    max_tokens=1000
                )
                self._model_name = "OpenAI (gpt-3.5-turbo)"
                
            elif primary_llm == "anthropic" and self.config.llm.anthropic_api_key:
                self._llm = ChatAnthropic(
                    anthropic_api_key=self.config.llm.anthropic_api_key,
                    model="claude-3-haiku-20240307",
                    temperature=0.1,
                    max_tokens=1000
                )
                self._model_name = "Anthropic (Claude-3-Haiku)"
                
            else:
                # Fallback to any available LLM
                if self.config.llm.groq_api_key:
                    self._llm = ChatGroq(
                        groq_api_key=self.config.llm.groq_api_key,
                        model_name="mixtral-8x7b-32768",
                        temperature=0.1,
                        max_tokens=1000
                    )
                    self._model_name = "Groq (mixtral-8x7b-32768)"
                elif self.config.llm.openai_api_key:
                    self._llm = ChatOpenAI(
                        openai_api_key=self.config.llm.openai_api_key,
                        model_name="gpt-3.5-turbo",
                        temperature=0.1,
                        max_tokens=1000
                    )
                    self._model_name = "OpenAI (gpt-3.5-turbo)"
                else:
                    raise ValueError("No LLM API key configured")
                    
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise ValueError(f"Failed to initialize LLM: {str(e)}")
    
    def _get_query_examples(self) -> List[Dict[str, str]]:
        """Get example question-Cypher pairs for few-shot prompting."""
        return [
            {
                "question": "What entities are mentioned in documents about AI?",
                "cypher": """MATCH (d:Document)-[:CONTAINS]->(e:Entity) 
WHERE d.content CONTAINS 'AI' OR d.title CONTAINS 'AI'
RETURN e.name, e.type, e.confidence
ORDER BY e.confidence DESC
LIMIT 20"""
            },
            {
                "question": "Find all people who work for organizations",
                "cypher": """MATCH (person:Entity {type: 'PERSON'})-[:WORKS_FOR]->(org:Entity {type: 'ORGANIZATION'})
RETURN person.name as person, org.name as organization
ORDER BY person.name
LIMIT 50"""
            },
            {
                "question": "Show me the most connected entities in the graph",
                "cypher": """MATCH (e:Entity)-[r]-()
RETURN e.name, e.type, count(r) as connections
ORDER BY connections DESC
LIMIT 10"""
            },
            {
                "question": "What documents contain information about a specific person?",
                "cypher": """MATCH (d:Document)-[:CONTAINS]->(e:Entity {type: 'PERSON'})
WHERE e.name CONTAINS $person_name
RETURN d.title, d.id, e.name as person
ORDER BY d.title
LIMIT 20"""
            },
            {
                "question": "Find relationships between two specific entities",
                "cypher": """MATCH path = (e1:Entity)-[*1..3]-(e2:Entity)
WHERE e1.name CONTAINS $entity1 AND e2.name CONTAINS $entity2
RETURN path, length(path) as path_length
ORDER BY path_length
LIMIT 10"""
            }
        ]
    
    def generate_cypher(
        self,
        question: str,
        include_examples: bool = True,
        max_retries: int = 2
    ) -> CypherGenerationResult:
        """
        Generate Cypher query from natural language question.
        
        Args:
            question: Natural language question
            include_examples: Whether to include examples in prompt
            max_retries: Maximum number of retry attempts
            
        Returns:
            CypherGenerationResult with generated query and metadata
        """
        start_time = datetime.now()
        
        if not self._llm:
            return CypherGenerationResult(
                success=False,
                error_message="LLM not initialized",
                processing_time_seconds=0.0
            )
        
        if not question.strip():
            return CypherGenerationResult(
                success=False,
                error_message="Empty question provided",
                processing_time_seconds=0.0
            )
        
        try:
            # Build schema information
            schema_info = {
                "node_labels": self.schema.get_all_node_labels(),
                "relationship_types": self.schema.get_all_relationship_types()
            }
            
            # Build prompt
            examples = self._examples if include_examples else []
            prompt_text = build_text_to_cypher_prompt(
                user_question=question,
                schema_info=schema_info,
                examples=examples
            )
            
            # Generate Cypher with retries
            for attempt in range(max_retries + 1):
                try:
                    logger.debug(f"Generating Cypher (attempt {attempt + 1}): {question}")
                    
                    # Create messages for chat model
                    messages = [
                        SystemMessage(content=prompt_text),
                        HumanMessage(content=f"Question: {question}")
                    ]
                    
                    # Generate response
                    response = self._llm.invoke(messages)
                    cypher_query = response.content.strip()
                    
                    # Clean up the response (remove markdown formatting if present)
                    cypher_query = self._clean_cypher_response(cypher_query)
                    
                    # Validate the generated Cypher
                    validation_result = self._validate_cypher(cypher_query)
                    
                    if validation_result["valid"]:
                        processing_time = (datetime.now() - start_time).total_seconds()
                        
                        logger.info(f"Cypher generated successfully for: {question}")
                        
                        return CypherGenerationResult(
                            success=True,
                            cypher_query=cypher_query,
                            parameters=validation_result.get("parameters", {}),
                            confidence=validation_result.get("confidence", 0.8),
                            reasoning=f"Generated using {self._model_name}",
                            processing_time_seconds=processing_time,
                            model_used=self._model_name
                        )
                    else:
                        logger.warning(f"Generated Cypher validation failed (attempt {attempt + 1}): {validation_result['error']}")
                        if attempt == max_retries:
                            # Return the query anyway with warning
                            processing_time = (datetime.now() - start_time).total_seconds()
                            return CypherGenerationResult(
                                success=True,
                                cypher_query=cypher_query,
                                parameters={},
                                confidence=0.5,
                                reasoning=f"Generated with validation warnings: {validation_result['error']}",
                                processing_time_seconds=processing_time,
                                model_used=self._model_name
                            )
                        
                except Exception as e:
                    logger.warning(f"Error in Cypher generation attempt {attempt + 1}: {str(e)}")
                    if attempt == max_retries:
                        raise e
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error generating Cypher for question '{question}': {str(e)}")
            
            return CypherGenerationResult(
                success=False,
                error_message=f"Cypher generation failed: {str(e)}",
                processing_time_seconds=processing_time,
                model_used=self._model_name
            )
    
    def _clean_cypher_response(self, response: str) -> str:
        """Clean up LLM response to extract Cypher query."""
        # Remove markdown code blocks
        if "```" in response:
            lines = response.split("\n")
            in_code_block = False
            cypher_lines = []
            
            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    cypher_lines.append(line)
            
            if cypher_lines:
                response = "\n".join(cypher_lines)
        
        # Remove common prefixes
        prefixes_to_remove = [
            "Cypher:",
            "Query:",
            "CYPHER:",
            "QUERY:",
            "Here's the Cypher query:",
            "The Cypher query is:"
        ]
        
        for prefix in prefixes_to_remove:
            if response.strip().startswith(prefix):
                response = response.strip()[len(prefix):].strip()
        
        return response.strip()
    
    def _validate_cypher(self, cypher_query: str) -> Dict[str, Any]:
        """
        Validate generated Cypher query for basic syntax and structure.
        
        Args:
            cypher_query: Cypher query to validate
            
        Returns:
            Dictionary with validation results
        """
        if not cypher_query:
            return {"valid": False, "error": "Empty query"}
        
        # Basic syntax checks
        cypher_upper = cypher_query.upper()
        
        # Check for required clauses
        has_match_or_create = any(clause in cypher_upper for clause in ["MATCH", "CREATE", "MERGE"])
        has_return = "RETURN" in cypher_upper
        
        if not has_match_or_create:
            return {"valid": False, "error": "Query must contain MATCH, CREATE, or MERGE clause"}
        
        if not has_return and "DELETE" not in cypher_upper and "SET" not in cypher_upper:
            return {"valid": False, "error": "Query should contain RETURN clause for retrieval"}
        
        # Check for dangerous operations (for safety)
        dangerous_operations = ["DELETE", "REMOVE", "DROP", "CREATE CONSTRAINT", "DROP CONSTRAINT"]
        for operation in dangerous_operations:
            if operation in cypher_upper:
                return {"valid": False, "error": f"Potentially dangerous operation detected: {operation}"}
        
        # Extract parameters (basic detection)
        parameters = {}
        import re
        param_matches = re.findall(r'\$(\w+)', cypher_query)
        for param in param_matches:
            parameters[param] = f"<{param}_value>"
        
        # Basic confidence scoring
        confidence = 0.7
        if "LIMIT" in cypher_upper:
            confidence += 0.1
        if "ORDER BY" in cypher_upper:
            confidence += 0.1
        if len(parameters) > 0:
            confidence += 0.1
        
        return {
            "valid": True,
            "parameters": parameters,
            "confidence": min(confidence, 1.0),
            "has_parameters": len(parameters) > 0
        }
    
    def generate_cypher_with_context(
        self,
        question: str,
        context_entities: List[str] = None,
        context_relationships: List[str] = None
    ) -> CypherGenerationResult:
        """
        Generate Cypher query with additional context about relevant entities/relationships.
        
        Args:
            question: Natural language question
            context_entities: List of relevant entity names for context
            context_relationships: List of relevant relationship types for context
            
        Returns:
            CypherGenerationResult with generated query
        """
        # Enhance the question with context
        enhanced_question = question
        
        if context_entities:
            entity_context = ", ".join(context_entities[:5])  # Limit to avoid prompt bloat
            enhanced_question += f"\n\nRelevant entities to consider: {entity_context}"
        
        if context_relationships:
            rel_context = ", ".join(context_relationships[:5])
            enhanced_question += f"\n\nRelevant relationships to consider: {rel_context}"
        
        return self.generate_cypher(enhanced_question)
    
    def explain_cypher(self, cypher_query: str) -> str:
        """
        Generate a natural language explanation of a Cypher query.
        
        Args:
            cypher_query: Cypher query to explain
            
        Returns:
            Natural language explanation
        """
        if not self._llm:
            return "LLM not available for explanation"
        
        try:
            explanation_prompt = f"""
You are a Neo4j expert. Explain the following Cypher query in simple, natural language.
Focus on what the query does, what data it retrieves, and how it works.

Cypher Query:
{cypher_query}

Provide a clear, concise explanation:
"""
            
            messages = [
                SystemMessage(content="You are a helpful Neo4j expert who explains Cypher queries clearly."),
                HumanMessage(content=explanation_prompt)
            ]
            
            response = self._llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error explaining Cypher query: {str(e)}")
            return f"Unable to generate explanation: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current LLM model."""
        return {
            "model_name": self._model_name,
            "provider": self.config.llm.primary_llm,
            "text_to_cypher_model": self.config.llm.text_to_cypher_model,
            "available": self._llm is not None,
            "examples_count": len(self._examples)
        }
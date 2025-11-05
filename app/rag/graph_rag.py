"""
Graph RAG engine for knowledge graph-based retrieval and generation.
Implements pure graph-based retrieval using Neo4j and Cypher queries.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

# Import handling for graph_rag module
import sys
from pathlib import Path

# Ensure proper path setup
current_dir = Path(__file__).parent
app_dir = current_dir.parent
project_root = app_dir.parent

if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import required modules
try:
    from rag.text_to_cypher import TextToCypherGenerator, CypherGenerationResult
except ImportError:
    TextToCypherGenerator = None
    CypherGenerationResult = None

try:
    from graph.graph_manager import GraphManager
except ImportError:
    GraphManager = None

try:
    from graph.cypher_builder import CommonQueries
except ImportError:
    CommonQueries = None

from config import get_config

logger = logging.getLogger(__name__)

@dataclass
class GraphContext:
    """Context retrieved from graph traversal."""
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    subgraphs: List[Dict[str, Any]] = field(default_factory=list)
    cypher_query: str = ""
    query_results: List[Dict[str, Any]] = field(default_factory=list)
    traversal_depth: int = 1
    total_nodes: int = 0
    total_relationships: int = 0

@dataclass
class GraphRAGResult:
    """Result of Graph RAG query processing."""
    success: bool
    query: str = ""
    answer: str = ""
    graph_context: Optional[GraphContext] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    cypher_generation: Optional[CypherGenerationResult] = None
    processing_time_seconds: float = 0.0
    error_message: str = ""
    confidence: float = 0.0
    reasoning_steps: List[str] = field(default_factory=list)

class GraphRetriever:
    """
    Retrieves relevant graph context using Cypher queries.
    Handles graph traversal and subgraph extraction.
    """
    
    def __init__(self, graph_manager: GraphManager):
        """Initialize graph retriever."""
        self.graph_manager = graph_manager
        self.cypher_generator = TextToCypherGenerator()
        
        logger.info("GraphRetriever initialized")
    
    async def retrieve_context(
        self,
        question: str,
        max_nodes: int = 50,
        max_depth: int = 2,
        use_text_to_cypher: bool = True
    ) -> GraphContext:
        """
        Retrieve graph context for a question.
        
        Args:
            question: Natural language question
            max_nodes: Maximum number of nodes to retrieve
            max_depth: Maximum traversal depth
            use_text_to_cypher: Whether to use Text-to-Cypher generation
            
        Returns:
            GraphContext with retrieved information
        """
        context = GraphContext()
        
        try:
            if use_text_to_cypher:
                # Generate Cypher query from natural language
                cypher_result = self.cypher_generator.generate_cypher(question)
                
                if cypher_result.success:
                    context.cypher_query = cypher_result.cypher_query
                    
                    # Execute the generated Cypher query
                    query_result = await self.graph_manager.query_graph(
                        cypher_result.cypher_query,
                        cypher_result.parameters
                    )
                    
                    if query_result["success"]:
                        context.query_results = query_result["records"]
                        
                        # Extract entities and relationships from results
                        await self._extract_context_from_results(context, query_result["records"])
                    else:
                        logger.warning(f"Cypher query execution failed: {query_result.get('error', 'Unknown error')}")
                        # Fallback to keyword-based retrieval
                        await self._fallback_keyword_retrieval(context, question, max_nodes)
                else:
                    logger.warning(f"Cypher generation failed: {cypher_result.error_message}")
                    # Fallback to keyword-based retrieval
                    await self._fallback_keyword_retrieval(context, question, max_nodes)
            else:
                # Use keyword-based retrieval directly
                await self._fallback_keyword_retrieval(context, question, max_nodes)
            
            # Enhance context with graph traversal if we have entities
            if context.entities:
                await self._enhance_with_traversal(context, max_depth, max_nodes)
            
            # Calculate totals
            context.total_nodes = len(context.entities)
            context.total_relationships = len(context.relationships)
            context.traversal_depth = max_depth
            
            logger.info(f"Retrieved graph context: {context.total_nodes} nodes, {context.total_relationships} relationships")
            
        except Exception as e:
            logger.error(f"Error retrieving graph context: {str(e)}")
            # Return empty context rather than failing
        
        return context
    
    async def _extract_context_from_results(
        self,
        context: GraphContext,
        query_results: List[Dict[str, Any]]
    ):
        """Extract entities and relationships from Cypher query results."""
        entities_seen = set()
        relationships_seen = set()
        
        for record in query_results:
            for key, value in record.items():
                if isinstance(value, dict):
                    # Check if this looks like a node (has id and labels/type)
                    if "id" in value and ("type" in value or "name" in value):
                        entity_id = value["id"]
                        if entity_id not in entities_seen:
                            context.entities.append(value)
                            entities_seen.add(entity_id)
                    
                    # Check if this looks like a relationship
                    elif "source" in value and "target" in value:
                        rel_id = f"{value.get('source', '')}_{value.get('target', '')}"
                        if rel_id not in relationships_seen:
                            context.relationships.append(value)
                            relationships_seen.add(rel_id)
    
    async def _fallback_keyword_retrieval(
        self,
        context: GraphContext,
        question: str,
        max_nodes: int
    ):
        """Fallback to keyword-based entity search when Cypher generation fails."""
        try:
            # Extract potential keywords from the question
            keywords = self._extract_keywords(question)
            
            # Search for entities using keywords
            for keyword in keywords[:3]:  # Limit to top 3 keywords
                search_result = await self.graph_manager.search_entities(keyword, limit=max_nodes // 3)
                
                if search_result["success"]:
                    for record in search_result["records"]:
                        if "entity" in record:
                            entity = record["entity"]
                            if entity not in context.entities:
                                context.entities.append(entity)
            
            # Set a simple query description
            context.cypher_query = f"Keyword search for: {', '.join(keywords)}"
            
        except Exception as e:
            logger.error(f"Error in fallback keyword retrieval: {str(e)}")
    
    def _extract_keywords(self, question: str) -> List[str]:
        """Extract potential keywords from a question."""
        # Simple keyword extraction (could be enhanced with NLP)
        import re
        
        # Remove common question words
        stop_words = {
            "what", "who", "where", "when", "why", "how", "is", "are", "was", "were",
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "about", "show", "find", "get", "tell", "me"
        }
        
        # Extract words (alphanumeric sequences)
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', question.lower())
        
        # Filter out stop words and short words
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:5]  # Return top 5 keywords
    
    async def _enhance_with_traversal(
        self,
        context: GraphContext,
        max_depth: int,
        max_nodes: int
    ):
        """Enhance context by traversing from found entities."""
        if not context.entities or len(context.entities) >= max_nodes:
            return
        
        try:
            # Get entity IDs for traversal
            entity_ids = [entity.get("id") for entity in context.entities if entity.get("id")]
            
            if not entity_ids:
                return
            
            # Traverse from each entity (limit to avoid explosion)
            for entity_id in entity_ids[:5]:  # Limit starting points
                traversal_result = await self.graph_manager.traverse_from_entity(
                    entity_id=entity_id,
                    max_depth=min(max_depth, 2),  # Limit depth to avoid large results
                    relationship_types=None
                )
                
                if traversal_result["success"]:
                    traversal_data = traversal_result["traversal_result"]
                    
                    # Add new nodes and relationships
                    if "all_nodes" in traversal_data:
                        for node in traversal_data["all_nodes"]:
                            if node not in context.entities and len(context.entities) < max_nodes:
                                context.entities.append(node)
                    
                    if "all_relationships" in traversal_data:
                        for rel in traversal_data["all_relationships"]:
                            if rel not in context.relationships:
                                context.relationships.append(rel)
                    
                    # Store subgraph information
                    if "paths" in traversal_data:
                        context.subgraphs.extend(traversal_data["paths"])
        
        except Exception as e:
            logger.error(f"Error enhancing context with traversal: {str(e)}")

class GraphContextBuilder:
    """
    Builds structured context from graph retrieval results.
    Formats graph data for LLM consumption.
    """
    
    def __init__(self):
        """Initialize context builder."""
        logger.info("GraphContextBuilder initialized")
    
    def build_context_text(
        self,
        graph_context: GraphContext,
        max_context_length: int = 4000
    ) -> str:
        """
        Build formatted context text from graph data.
        
        Args:
            graph_context: Graph context data
            max_context_length: Maximum length of context text
            
        Returns:
            Formatted context text for LLM
        """
        context_parts = []
        
        # Add query information
        if graph_context.cypher_query:
            context_parts.append(f"Graph Query: {graph_context.cypher_query}")
            context_parts.append("")
        
        # Add entities
        if graph_context.entities:
            context_parts.append("ENTITIES FOUND:")
            for i, entity in enumerate(graph_context.entities[:20]):  # Limit entities
                entity_text = self._format_entity(entity)
                context_parts.append(f"{i+1}. {entity_text}")
            
            if len(graph_context.entities) > 20:
                context_parts.append(f"... and {len(graph_context.entities) - 20} more entities")
            context_parts.append("")
        
        # Add relationships
        if graph_context.relationships:
            context_parts.append("RELATIONSHIPS FOUND:")
            for i, relationship in enumerate(graph_context.relationships[:15]):  # Limit relationships
                rel_text = self._format_relationship(relationship)
                context_parts.append(f"{i+1}. {rel_text}")
            
            if len(graph_context.relationships) > 15:
                context_parts.append(f"... and {len(graph_context.relationships) - 15} more relationships")
            context_parts.append("")
        
        # Add query results if available
        if graph_context.query_results:
            context_parts.append("QUERY RESULTS:")
            for i, result in enumerate(graph_context.query_results[:10]):  # Limit results
                result_text = self._format_query_result(result)
                context_parts.append(f"{i+1}. {result_text}")
            
            if len(graph_context.query_results) > 10:
                context_parts.append(f"... and {len(graph_context.query_results) - 10} more results")
            context_parts.append("")
        
        # Add subgraph information
        if graph_context.subgraphs:
            context_parts.append("GRAPH CONNECTIONS:")
            for i, subgraph in enumerate(graph_context.subgraphs[:5]):  # Limit subgraphs
                if "start_node" in subgraph and "end_node" in subgraph:
                    start_name = subgraph["start_node"].get("name", "Unknown")
                    end_name = subgraph["end_node"].get("name", "Unknown")
                    path_length = subgraph.get("path_length", 0)
                    context_parts.append(f"{i+1}. {start_name} connected to {end_name} (distance: {path_length})")
        
        # Join and truncate if necessary
        full_context = "\n".join(context_parts)
        
        if len(full_context) > max_context_length:
            # Truncate and add notice
            truncated_context = full_context[:max_context_length - 100]
            truncated_context += "\n\n[Context truncated due to length limits]"
            return truncated_context
        
        return full_context
    
    def _format_entity(self, entity: Dict[str, Any]) -> str:
        """Format entity for context display."""
        name = entity.get("name", "Unknown")
        entity_type = entity.get("type", "Unknown")
        description = entity.get("description", "")
        confidence = entity.get("confidence", 0)
        
        formatted = f"{name} ({entity_type})"
        
        if description:
            formatted += f" - {description[:100]}"
        
        if confidence > 0:
            formatted += f" [confidence: {confidence:.2f}]"
        
        return formatted
    
    def _format_relationship(self, relationship: Dict[str, Any]) -> str:
        """Format relationship for context display."""
        rel_type = relationship.get("relationship_type", "RELATED_TO")
        description = relationship.get("description", "")
        confidence = relationship.get("confidence", 0)
        
        # Try to get source and target names
        source = relationship.get("source", {})
        target = relationship.get("target", {})
        
        source_name = source.get("name", "Unknown") if isinstance(source, dict) else str(source)
        target_name = target.get("name", "Unknown") if isinstance(target, dict) else str(target)
        
        formatted = f"{source_name} --[{rel_type}]--> {target_name}"
        
        if description:
            formatted += f" ({description[:50]})"
        
        if confidence > 0:
            formatted += f" [confidence: {confidence:.2f}]"
        
        return formatted
    
    def _format_query_result(self, result: Dict[str, Any]) -> str:
        """Format query result for context display."""
        # Handle different types of query results
        formatted_parts = []
        
        for key, value in result.items():
            if isinstance(value, dict) and "name" in value:
                # This looks like an entity
                formatted_parts.append(f"{key}: {value['name']}")
            elif isinstance(value, (str, int, float)):
                formatted_parts.append(f"{key}: {value}")
            elif isinstance(value, list) and len(value) > 0:
                formatted_parts.append(f"{key}: {len(value)} items")
        
        return ", ".join(formatted_parts) if formatted_parts else str(result)
    
    def extract_sources(self, graph_context: GraphContext) -> List[Dict[str, Any]]:
        """Extract source information for attribution."""
        sources = []
        
        # Extract document sources from entities
        for entity in graph_context.entities:
            source_doc_id = entity.get("source_document_id")
            if source_doc_id:
                sources.append({
                    "type": "document",
                    "id": source_doc_id,
                    "entity": entity.get("name", "Unknown"),
                    "confidence": entity.get("confidence", 0)
                })
        
        # Extract from relationships
        for relationship in graph_context.relationships:
            source_doc_id = relationship.get("source_document_id")
            if source_doc_id:
                sources.append({
                    "type": "document",
                    "id": source_doc_id,
                    "relationship": relationship.get("relationship_type", "Unknown"),
                    "confidence": relationship.get("confidence", 0)
                })
        
        # Deduplicate sources by document ID
        unique_sources = {}
        for source in sources:
            doc_id = source["id"]
            if doc_id not in unique_sources or source["confidence"] > unique_sources[doc_id]["confidence"]:
                unique_sources[doc_id] = source
        
        return list(unique_sources.values())

class GraphRAGPipeline:
    """
    Complete Graph RAG pipeline that orchestrates retrieval and generation.
    Combines graph retrieval with LLM-based answer generation.
    """
    
    def __init__(self, graph_manager: GraphManager):
        """Initialize Graph RAG pipeline."""
        self.graph_manager = graph_manager
        self.retriever = GraphRetriever(graph_manager)
        self.context_builder = GraphContextBuilder()
        self.config = get_config()
        
        # Initialize LLM for answer generation (reuse from text-to-cypher)
        self.cypher_generator = TextToCypherGenerator()
        self._llm = self.cypher_generator._llm
        self._model_name = self.cypher_generator._model_name
        
        logger.info("GraphRAGPipeline initialized")
    
    async def query(
        self,
        question: str,
        max_context_nodes: int = 50,
        max_traversal_depth: int = 2,
        use_text_to_cypher: bool = True
    ) -> GraphRAGResult:
        """
        Process a question using Graph RAG.
        
        Args:
            question: Natural language question
            max_context_nodes: Maximum nodes to include in context
            max_traversal_depth: Maximum graph traversal depth
            use_text_to_cypher: Whether to use Text-to-Cypher generation
            
        Returns:
            GraphRAGResult with answer and context
        """
        start_time = datetime.now()
        reasoning_steps = []
        
        try:
            reasoning_steps.append("Starting Graph RAG query processing")
            
            # Step 1: Retrieve graph context
            reasoning_steps.append("Retrieving relevant graph context")
            graph_context = await self.retriever.retrieve_context(
                question=question,
                max_nodes=max_context_nodes,
                max_depth=max_traversal_depth,
                use_text_to_cypher=use_text_to_cypher
            )
            
            if not graph_context.entities and not graph_context.query_results:
                reasoning_steps.append("No relevant graph context found")
                return GraphRAGResult(
                    success=False,
                    query=question,
                    error_message="No relevant information found in the knowledge graph",
                    graph_context=graph_context,
                    processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                    reasoning_steps=reasoning_steps
                )
            
            reasoning_steps.append(f"Found {graph_context.total_nodes} entities and {graph_context.total_relationships} relationships")
            
            # Step 2: Build context for LLM
            reasoning_steps.append("Building context for answer generation")
            context_text = self.context_builder.build_context_text(graph_context)
            
            # Step 3: Generate answer using LLM
            reasoning_steps.append("Generating answer using LLM")
            answer = await self._generate_answer(question, context_text)
            
            # Step 4: Extract sources
            sources = self.context_builder.extract_sources(graph_context)
            
            # Calculate confidence based on context quality
            confidence = self._calculate_confidence(graph_context, len(answer))
            
            processing_time = (datetime.now() - start_time).total_seconds()
            reasoning_steps.append(f"Graph RAG processing completed in {processing_time:.2f} seconds")
            
            logger.info(f"Graph RAG query completed: {question}")
            
            return GraphRAGResult(
                success=True,
                query=question,
                answer=answer,
                graph_context=graph_context,
                sources=sources,
                processing_time_seconds=processing_time,
                confidence=confidence,
                reasoning_steps=reasoning_steps
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Graph RAG query failed: {str(e)}"
            reasoning_steps.append(error_msg)
            
            logger.error(f"Error in Graph RAG query '{question}': {str(e)}")
            
            return GraphRAGResult(
                success=False,
                query=question,
                error_message=error_msg,
                processing_time_seconds=processing_time,
                reasoning_steps=reasoning_steps
            )
    
    async def _generate_answer(self, question: str, context_text: str) -> str:
        """Generate answer using LLM with graph context."""
        if not self._llm:
            return "LLM not available for answer generation"
        
        try:
            # Build prompt for answer generation
            system_prompt = """You are a helpful assistant that answers questions based on information from a knowledge graph.

Use the provided graph context to answer the user's question. The context includes entities, relationships, and connections found in the knowledge graph.

Guidelines:
1. Base your answer primarily on the provided graph context
2. Be specific and cite relevant entities and relationships when possible
3. If the context doesn't contain enough information, say so clearly
4. Provide a comprehensive answer that explains the connections and relationships
5. Use natural language and make the answer easy to understand

Graph Context:
{context}

Answer the question based on this graph information."""
            
            user_prompt = f"Question: {question}\n\nPlease provide a detailed answer based on the graph context above."
            
            messages = [
                SystemMessage(content=system_prompt.format(context=context_text)),
                HumanMessage(content=user_prompt)
            ]
            
            response = self._llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Error generating answer: {str(e)}"
    
    def _calculate_confidence(self, graph_context: GraphContext, answer_length: int) -> float:
        """Calculate confidence score for the Graph RAG result."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on context richness
        if graph_context.total_nodes > 0:
            confidence += min(graph_context.total_nodes / 20, 0.2)  # Up to 0.2 boost
        
        if graph_context.total_relationships > 0:
            confidence += min(graph_context.total_relationships / 10, 0.1)  # Up to 0.1 boost
        
        # Boost if we have query results
        if graph_context.query_results:
            confidence += min(len(graph_context.query_results) / 10, 0.1)
        
        # Boost if we have a successful Cypher query
        if graph_context.cypher_query and "error" not in graph_context.cypher_query.lower():
            confidence += 0.1
        
        # Boost based on answer length (longer answers often indicate more context)
        if answer_length > 100:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the Graph RAG pipeline."""
        return {
            "model_name": self._model_name,
            "text_to_cypher_available": self.cypher_generator._llm is not None,
            "graph_manager_initialized": self.graph_manager is not None,
            "pipeline_components": [
                "GraphRetriever",
                "GraphContextBuilder", 
                "TextToCypherGenerator",
                "LLM Answer Generation"
            ]
        }
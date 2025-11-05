# Workshop Example Queries for Knowledge Graph RAG System

This document provides example queries that demonstrate the capabilities of both Graph RAG and Hybrid RAG approaches. These queries are designed to showcase different types of reasoning and retrieval patterns.

## Entity-Focused Queries

### Query 1: Person Identification
**Question**: "Who is Yann LeCun and what is he known for?"

**Expected Graph RAG Behavior**:
- Identifies "Yann LeCun" as a key entity
- Traverses relationships to find his affiliations (Meta AI, NYU)
- Discovers his research contributions (deep learning, CNNs)
- Shows connections to other researchers and institutions

**Expected Hybrid RAG Behavior**:
- Uses vector search to find semantically similar content about Yann LeCun
- Combines with graph traversal to show his network of collaborations
- Provides comprehensive context from multiple document sources

### Query 2: Organization Analysis
**Question**: "What companies are working on large language models?"

**Expected Graph RAG Behavior**:
- Identifies organizations mentioned in the knowledge graph
- Traverses "works_on" or "develops" relationships to LLM products
- Shows connections between companies and their LLM offerings

**Expected Hybrid RAG Behavior**:
- Finds documents discussing LLMs through vector similarity
- Extracts company names and their LLM products
- Ranks results by relevance and completeness

## Relationship-Focused Queries

### Query 3: Collaboration Networks
**Question**: "Which universities collaborate with MIT in AI research?"

**Expected Graph RAG Behavior**:
- Starts from MIT entity in the graph
- Traverses "collaborates_with" relationships
- Identifies partner universities and joint research areas
- Shows the network of academic partnerships

**Expected Hybrid RAG Behavior**:
- Searches for documents mentioning MIT collaborations
- Uses graph data to verify and expand collaboration networks
- Provides context about specific research projects and partnerships

### Query 4: Technology Connections
**Question**: "How are knowledge graphs related to RAG systems?"

**Expected Graph RAG Behavior**:
- Identifies both "knowledge graphs" and "RAG systems" as concepts
- Traverses relationships showing how they connect
- Finds intermediate concepts like "retrieval" and "structured knowledge"

**Expected Hybrid RAG Behavior**:
- Finds documents explaining both technologies
- Uses semantic similarity to identify related concepts
- Combines with graph structure to show technical relationships

## Multi-Hop Reasoning Queries

### Query 5: Indirect Connections
**Question**: "What is the connection between Stanford University and Neo4j?"

**Expected Graph RAG Behavior**:
- Multi-hop traversal from Stanford → researchers → companies → Neo4j
- Might find connections through alumni, partnerships, or research collaborations
- Shows the path of relationships between entities

**Expected Hybrid RAG Behavior**:
- Searches for documents mentioning both Stanford and Neo4j
- Uses graph traversal to find indirect connections
- Provides context about how academic research influences industry

### Query 6: Technology Evolution
**Question**: "How did transformer architectures lead to modern RAG systems?"

**Expected Graph RAG Behavior**:
- Traces the evolution from transformers → LLMs → RAG systems
- Shows key researchers and papers in this progression
- Identifies intermediate technologies and concepts

**Expected Hybrid RAG Behavior**:
- Finds documents describing the historical development
- Uses semantic search to identify related technological concepts
- Combines timeline information with technical relationships

## Comparative Analysis Queries

### Query 7: Technology Comparison
**Question**: "What are the differences between vector databases like Qdrant and graph databases like Neo4j?"

**Expected Graph RAG Behavior**:
- Identifies both database types as distinct entities
- Compares their properties and use cases through graph relationships
- Shows which companies use which technologies

**Expected Hybrid RAG Behavior**:
- Finds documents directly comparing these technologies
- Uses vector similarity to identify related database concepts
- Provides comprehensive technical comparisons

### Query 8: Research Institution Comparison
**Question**: "How do the AI research focuses of MIT and Stanford differ?"

**Expected Graph RAG Behavior**:
- Compares research areas associated with each institution
- Shows different faculty members and their specializations
- Identifies unique vs. overlapping research themes

**Expected Hybrid RAG Behavior**:
- Searches for documents describing each institution's research
- Uses semantic analysis to identify research themes
- Provides detailed comparison of research priorities

## Complex Analytical Queries

### Query 9: Industry Landscape Analysis
**Question**: "Which AI companies are founded by university researchers and what technologies do they focus on?"

**Expected Graph RAG Behavior**:
- Identifies researchers with "founded" relationships to companies
- Traces their university affiliations
- Shows the technologies these companies develop

**Expected Hybrid RAG Behavior**:
- Searches for information about AI company founders
- Cross-references with academic backgrounds
- Provides comprehensive industry-academia connection analysis

### Query 10: Research Impact Assessment
**Question**: "What are the most influential research areas in modern AI according to the documents?"

**Expected Graph RAG Behavior**:
- Analyzes centrality of research topics in the knowledge graph
- Identifies highly connected concepts and researchers
- Shows influence through relationship density

**Expected Hybrid RAG Behavior**:
- Uses semantic analysis to identify frequently discussed topics
- Combines with graph metrics to assess research impact
- Provides evidence-based ranking of research areas

## Performance Comparison Scenarios

### Scenario A: Factual Retrieval
**Query**: "What is the founding year of OpenAI?"

**Graph RAG Expected Performance**: Fast, precise answer from structured data
**Hybrid RAG Expected Performance**: Comprehensive answer with additional context

### Scenario B: Conceptual Understanding
**Query**: "Explain how retrieval-augmented generation works."

**Graph RAG Expected Performance**: Structured explanation following concept relationships
**Hybrid RAG Expected Performance**: Rich explanation combining multiple document sources

### Scenario C: Discovery and Exploration
**Query**: "What emerging trends in AI research should I know about?"

**Graph RAG Expected Performance**: Identifies trends through relationship analysis
**Hybrid RAG Expected Performance**: Discovers trends through semantic similarity and recency

## Workshop Demonstration Tips

### For Graph RAG Demonstrations:
1. Start with entity-focused queries to show graph traversal
2. Progress to relationship queries to demonstrate connection discovery
3. Use multi-hop queries to show complex reasoning capabilities
4. Highlight the Cypher query generation process

### For Hybrid RAG Demonstrations:
1. Begin with semantic similarity queries to show vector search
2. Demonstrate how graph and vector results are combined
3. Show deduplication and ranking processes
4. Compare fusion methods and their effectiveness

### For Comparative Analysis:
1. Use the same query on both systems
2. Compare response quality, speed, and completeness
3. Highlight when each approach excels
4. Discuss complementary strengths of hybrid approach

## Expected Performance Characteristics

### Graph RAG Strengths:
- Precise entity and relationship identification
- Excellent for factual queries with clear entities
- Fast traversal of known relationships
- Structured, logical reasoning paths

### Hybrid RAG Strengths:
- Better semantic understanding of complex queries
- More comprehensive context from multiple sources
- Effective for exploratory and analytical questions
- Robust handling of ambiguous or incomplete queries

### When to Use Each:
- **Graph RAG**: When you need precise, structured answers about known entities and relationships
- **Hybrid RAG**: When you need comprehensive analysis, semantic understanding, or exploration of complex topics

This query set provides a comprehensive foundation for demonstrating the capabilities and differences between Graph RAG and Hybrid RAG approaches in a workshop setting.
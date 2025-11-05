"""
RAG (Retrieval-Augmented Generation) engines for the knowledge graph system.
Provides Graph RAG, Vector Search, and Hybrid RAG implementations.
"""

# Import handling for rag module
import sys
from pathlib import Path

# Ensure proper path setup
current_dir = Path(__file__).parent
app_dir = current_dir.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

# Import with error handling
try:
    from .graph_rag import GraphRAGPipeline, GraphRAGResult, GraphContext
except ImportError:
    GraphRAGPipeline = None
    GraphRAGResult = None
    GraphContext = None

try:
    from .graph_rag_manager import GraphRAGManager
except ImportError:
    GraphRAGManager = None

try:
    from .vector_search import VectorSearchEngine, VectorSearchResult, VectorContext, create_vector_search_engine
except ImportError:
    VectorSearchEngine = None
    VectorSearchResult = None
    VectorContext = None
    create_vector_search_engine = None

try:
    from .hybrid_rag import (
        HybridRAGEngine, 
        HybridRAGResult, 
        HybridContext, 
        FusionMethod, 
        FusionWeights, 
        ResultFusion,
        create_hybrid_rag_engine
    )
except ImportError:
    HybridRAGEngine = None
    HybridRAGResult = None
    HybridContext = None
    FusionMethod = None
    FusionWeights = None
    ResultFusion = None
    create_hybrid_rag_engine = None

__all__ = [
    # Graph RAG
    "GraphRAGPipeline",
    "GraphRAGResult", 
    "GraphContext",
    "GraphRAGManager",
    
    # Vector Search
    "VectorSearchEngine",
    "VectorSearchResult",
    "VectorContext", 
    "create_vector_search_engine",
    
    # Hybrid RAG
    "HybridRAGEngine",
    "HybridRAGResult",
    "HybridContext",
    "FusionMethod",
    "FusionWeights",
    "ResultFusion",
    "create_hybrid_rag_engine"
]
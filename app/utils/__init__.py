# Utility functions for Knowledge Graph RAG System

try:
    from .workshop_demo import workshop_demo
    __all__ = ['workshop_demo']
except ImportError:
    # Workshop demo not available
    __all__ = []
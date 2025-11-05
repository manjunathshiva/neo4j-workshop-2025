"""
Streamlit Web Interface for Knowledge Graph RAG System.
Provides document upload, processing, and query interface for both Graph RAG and Hybrid RAG.
"""

import streamlit as st
import asyncio
import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import application components
import sys
import os
from pathlib import Path

# Set up Python path for proper imports
current_dir = Path(__file__).parent
project_root = current_dir.parent

# Add directories to Python path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

# Set PYTHONPATH environment variable as well
current_pythonpath = os.environ.get('PYTHONPATH', '')
new_paths = [str(project_root), str(current_dir)]
if current_pythonpath:
    os.environ['PYTHONPATH'] = os.pathsep.join(new_paths + [current_pythonpath])
else:
    os.environ['PYTHONPATH'] = os.pathsep.join(new_paths)

# Import with proper error handling
try:
    # Import configuration first
    sys.path.insert(0, str(current_dir))
    from app.config import get_config
    
    # Import other components
    from app.connections import get_database_manager
    from app.processors.document_processor import DocumentProcessor, ProcessingStatus
    
    # Import RAG components with fallback handling
    # Start with the most basic components first
    
    # Try GraphManager first (needed by others)
    try:
        from app.graph.graph_manager import GraphManager
        logger.info("✅ GraphManager imported successfully")
    except ImportError as e:
        logger.warning(f"Graph Manager import failed: {e}")
        GraphManager = None
    
    # Try EmbeddingPipeline
    try:
        from app.embeddings.embedding_pipeline import EmbeddingPipeline, EmbeddingType
        logger.info("✅ EmbeddingPipeline imported successfully")
    except ImportError as e:
        logger.warning(f"Embedding Pipeline import failed: {e}")
        EmbeddingPipeline = None
    
    # Try GraphRAGManager (depends on GraphManager)
    try:
        if GraphManager is not None:
            from app.rag.graph_rag_manager import GraphRAGManager
            logger.info("✅ GraphRAGManager imported successfully")
        else:
            raise ImportError("GraphManager not available")
    except ImportError as e:
        logger.warning(f"Graph RAG Manager import failed: {e}")
        GraphRAGManager = None
    
    # Try HybridRAGEngine (depends on both GraphManager and EmbeddingPipeline)
    try:
        if GraphManager is not None and EmbeddingPipeline is not None:
            from app.rag.hybrid_rag import HybridRAGEngine, FusionMethod
            logger.info("✅ HybridRAGEngine imported successfully")
        else:
            raise ImportError("Required dependencies not available")
    except ImportError as e:
        logger.warning(f"Hybrid RAG Engine import failed: {e}")
        HybridRAGEngine = None
        FusionMethod = None

except ImportError as e:
    st.error(f"❌ Critical import error: {e}")
    st.error("Please ensure all dependencies are installed and you're running from the correct directory.")
    st.code(f"Current directory: {Path.cwd()}")
    st.code(f"Python path: {sys.path}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Knowledge Graph RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .upload-area {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.db_manager = None
        st.session_state.document_processor = None
        st.session_state.graph_rag_manager = None
        st.session_state.hybrid_rag_engine = None
        st.session_state.processed_documents = []
        st.session_state.system_status = {}
        st.session_state.last_query = ""
        st.session_state.query_results = {}
        
    # Workshop demo session state (initialize separately to avoid overwriting)
    if 'demo_query_id' not in st.session_state:
        st.session_state.demo_query_id = None
    if 'demo_method' not in st.session_state:
        st.session_state.demo_method = None
    if 'demo_ready' not in st.session_state:
        st.session_state.demo_ready = False

def display_header():
    """Display the main application header."""
    st.markdown('<h1 class="main-header">🧠 Knowledge Graph RAG System</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p><strong>Workshop Demonstration:</strong> Graph RAG vs Hybrid RAG</p>
        <p>Upload documents, build knowledge graphs, and query with natural language</p>
    </div>
    """, unsafe_allow_html=True)

def display_system_status():
    """Display system status in the sidebar."""
    st.sidebar.header("🔧 System Status")
    
    if not st.session_state.initialized:
        st.sidebar.warning("⚠️ System not initialized")
        if st.sidebar.button("Initialize System"):
            with st.spinner("Initializing system components..."):
                initialize_system()
        return
    
    # Display connection status
    status = st.session_state.system_status
    
    # Database connections
    st.sidebar.subheader("Database Connections")
    if status.get('neo4j_connected', False):
        st.sidebar.markdown('<p class="status-success">✅ Neo4j Connected</p>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<p class="status-error">❌ Neo4j Disconnected</p>', unsafe_allow_html=True)
    
    if status.get('qdrant_connected', False):
        st.sidebar.markdown('<p class="status-success">✅ Qdrant Connected</p>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<p class="status-error">❌ Qdrant Disconnected</p>', unsafe_allow_html=True)
    
    # Component status
    st.sidebar.subheader("Components")
    components = ['document_processor', 'graph_rag_manager', 'hybrid_rag_engine']
    for component in components:
        if status.get(f'{component}_ready', False):
            st.sidebar.markdown(f'<p class="status-success">✅ {component.replace("_", " ").title()}</p>', unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f'<p class="status-error">❌ {component.replace("_", " ").title()}</p>', unsafe_allow_html=True)
    
    # System metrics
    if status.get('graph_stats'):
        st.sidebar.subheader("Graph Statistics")
        stats = status['graph_stats']
        
        # Extract neo4j statistics from nested structure
        neo4j_stats = stats.get('neo4j_statistics', {})
        node_count = neo4j_stats.get('total_nodes', 0)
        rel_count = neo4j_stats.get('total_relationships', 0)
        
        st.sidebar.metric("Nodes", node_count)
        st.sidebar.metric("Relationships", rel_count)
        st.sidebar.metric("Documents", len(st.session_state.processed_documents))

@st.cache_data
def get_supported_file_types():
    """Get list of supported file types."""
    return [".pdf", ".txt", ".docx", ".md"]

def initialize_system():
    """Initialize all system components."""
    try:
        # Initialize configuration
        config = get_config()
        
        # Initialize database manager
        db_manager = get_database_manager()
        
        # Run async initialization
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Initialize database connections
        db_status = loop.run_until_complete(db_manager.initialize())
        
        # Clean up existing data for fresh demo start
        async def cleanup_databases():
            """Clean up Neo4j and Qdrant databases."""
            # Clear Neo4j database
            if db_status.get('neo4j', {}).get('connected', False):
                try:
                    async with db_manager.get_neo4j_session() as session:
                        # Delete all nodes and relationships
                        session.run("MATCH (n) DETACH DELETE n")
                    return True, "Neo4j cleared"
                except Exception as e:
                    logger.warning(f"Failed to clear Neo4j: {e}")
                    return False, f"Neo4j cleanup failed: {e}"
            return False, "Neo4j not connected"
        
        if db_status['overall_success']:
            st.info("🧹 Cleaning up existing data for fresh start...")
            
            # Clear Neo4j
            neo4j_cleared, neo4j_msg = loop.run_until_complete(cleanup_databases())
            if neo4j_cleared:
                st.success(f"✅ {neo4j_msg}")
            
            # Clear Qdrant collections
            if db_status.get('qdrant', {}).get('connected', False):
                try:
                    from qdrant_client.models import Distance, VectorParams
                    
                    # Get the Qdrant client from database manager
                    qdrant_client = db_manager.qdrant.get_client()
                    collections_cleared = 0
                    
                    # Delete and recreate collections
                    for collection_name in ['documents', 'chunks', 'entities']:
                        try:
                            # Check if collection exists
                            collections = qdrant_client.get_collections().collections
                            collection_exists = any(c.name == collection_name for c in collections)
                            
                            if collection_exists:
                                # Delete existing collection
                                result = qdrant_client.delete_collection(collection_name)
                                logger.info(f"Deleted collection {collection_name}: {result}")
                            
                            # Recreate collection with proper configuration
                            qdrant_client.create_collection(
                                collection_name=collection_name,
                                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                            )
                            collections_cleared += 1
                            logger.info(f"Created fresh collection: {collection_name}")
                        except Exception as e:
                            logger.error(f"Failed to reset collection {collection_name}: {e}", exc_info=True)
                    
                    st.success(f"✅ Qdrant collections cleared ({collections_cleared}/3 collections reset)")
                except Exception as e:
                    logger.error(f"Failed to clear Qdrant: {e}", exc_info=True)
                    st.warning(f"⚠️ Qdrant cleanup had issues: {e}")
        
        # Initialize document processor
        document_processor = DocumentProcessor()
        
        # Initialize components based on available connections and imports
        graph_rag_manager = None
        hybrid_rag_engine = None
        
        if db_status['overall_success']:
            # Initialize graph components if available
            if GraphManager is not None:
                try:
                    graph_manager = GraphManager()
                    graph_init = loop.run_until_complete(graph_manager.initialize())
                    
                    if graph_init['success']:
                        # Initialize Graph RAG if available
                        if GraphRAGManager is not None:
                            try:
                                graph_rag_manager = GraphRAGManager()
                                graph_rag_init = loop.run_until_complete(graph_rag_manager.initialize())
                            except Exception as e:
                                logger.warning(f"Graph RAG Manager initialization failed: {e}")
                        
                        # Initialize Hybrid RAG if available
                        if HybridRAGEngine is not None and EmbeddingPipeline is not None:
                            try:
                                embedding_pipeline = EmbeddingPipeline()
                                hybrid_rag_engine = HybridRAGEngine(graph_manager, embedding_pipeline)
                                hybrid_init = loop.run_until_complete(hybrid_rag_engine.initialize())
                            except Exception as e:
                                logger.warning(f"Hybrid RAG Engine initialization failed: {e}")
                except Exception as e:
                    logger.warning(f"Graph Manager initialization failed: {e}")
            else:
                logger.warning("GraphManager not available - skipping graph-based components")
        
        # Get graph statistics
        graph_stats = {}
        if graph_rag_manager:
            try:
                stats_result = loop.run_until_complete(graph_rag_manager.get_graph_statistics())
                if stats_result['success']:
                    graph_stats = stats_result['statistics']
            except:
                pass
        
        # Update session state
        st.session_state.db_manager = db_manager
        st.session_state.document_processor = document_processor
        st.session_state.graph_rag_manager = graph_rag_manager
        st.session_state.hybrid_rag_engine = hybrid_rag_engine
        st.session_state.initialized = True
        
        # Update system status
        st.session_state.system_status = {
            'neo4j_connected': db_status.get('neo4j', {}).get('connected', False),
            'qdrant_connected': db_status.get('qdrant', {}).get('connected', False),
            'document_processor_ready': document_processor is not None,
            'graph_rag_manager_ready': graph_rag_manager is not None,
            'hybrid_rag_engine_ready': hybrid_rag_engine is not None,
            'graph_stats': graph_stats
        }
        
        loop.close()
        
        st.success("✅ System initialized successfully!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ System initialization failed: {str(e)}")
        logger.exception("System initialization error")

def document_upload_interface():
    """Document upload and processing interface."""
    st.header("📄 Document Upload & Processing")
    
    if not st.session_state.initialized:
        st.warning("⚠️ Please initialize the system first using the sidebar.")
        return
    
    # File upload section
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    st.subheader("Upload Documents")
    
    # Display supported file types
    supported_types = get_supported_file_types()
    st.info(f"**Supported file types:** {', '.join(supported_types)}")
    
    # File uploader with drag and drop
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['pdf', 'txt', 'docx', 'md'],
        accept_multiple_files=True,
        help="Drag and drop files here or click to browse"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process uploaded files
    if uploaded_files:
        st.subheader("📊 Processing Status")
        
        # Add a "Process All" button for convenience
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        with col_btn1:
            if st.button("🚀 Process All Files", type="primary"):
                for uploaded_file in uploaded_files:
                    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
                    if file_key not in st.session_state.get('processed_file_keys', set()):
                        st.write(f"Processing {uploaded_file.name}...")
                        success = process_file_with_progress(uploaded_file)
                        if success:
                            if 'processed_file_keys' not in st.session_state:
                                st.session_state.processed_file_keys = set()
                            st.session_state.processed_file_keys.add(file_key)
                st.rerun()
        
        with col_btn2:
            if st.button("🗑️ Clear All"):
                st.session_state.processed_file_keys = set()
                st.session_state.processed_documents = []
                st.rerun()
        
        # Create columns for processing display
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Process each file
            for uploaded_file in uploaded_files:
                process_uploaded_file(uploaded_file)
        
        with col2:
            # Display processing summary
            display_processing_summary()
    
    # Display processed documents
    display_processed_documents()

def process_uploaded_file(uploaded_file):
    """Process a single uploaded file with progress tracking."""
    file_name = uploaded_file.name
    file_size = uploaded_file.size
    
    # Check if this file has already been processed
    file_key = f"{file_name}_{file_size}"
    if file_key in st.session_state.get('processed_file_keys', set()):
        # File already processed, show summary
        st.success(f"✅ {file_name} - Already processed")
        return
    
    # Create a container for this file
    st.subheader(f"📄 {file_name}")
    
    # Validate file
    validation = st.session_state.document_processor.validate_file_upload(file_name, file_size)
    
    if not validation['valid']:
        st.error("❌ File validation failed:")
        for error in validation['errors']:
            st.error(f"• {error}")
        return
    
    if validation['warnings']:
        for warning in validation['warnings']:
            st.warning(f"⚠️ {warning}")
    
    # Display file info
    file_info = validation['file_info']
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("File Size", f"{file_info['file_size_mb']:.2f} MB")
    with col2:
        st.metric("File Type", file_info['file_type'])
    with col3:
        st.metric("Status", "Ready to Process")
    with col4:
        # Process button - make it prominent
        if st.button(f"🚀 Process {file_name}", key=f"process_{file_name}", type="primary"):
            if st.session_state.document_processor:
                # Initialize processed file keys if not exists
                if 'processed_file_keys' not in st.session_state:
                    st.session_state.processed_file_keys = set()
                
                # Process the file
                success = process_file_with_progress(uploaded_file)
                
                # Mark as processed if successful
                if success:
                    st.session_state.processed_file_keys.add(file_key)
                    st.rerun()
            else:
                st.error("❌ Document processor not available")
    
    st.divider()  # Add visual separation between files

def process_file_with_progress(uploaded_file):
    """Process file with real-time progress updates."""
    file_name = uploaded_file.name
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Reading file
        status_text.text("📖 Reading file content...")
        progress_bar.progress(20)
        time.sleep(0.5)  # Simulate processing time
        
        file_content = uploaded_file.read()
        
        # Step 2: Text extraction
        status_text.text("🔍 Extracting text content...")
        progress_bar.progress(40)
        time.sleep(0.5)
        
        # Step 3: Document processing
        status_text.text("⚙️ Processing document...")
        progress_bar.progress(60)
        
        # Process the document
        try:
            processed_doc = st.session_state.document_processor.process_uploaded_file(
                file_content=file_content,
                filename=file_name
            )
        except Exception as e:
            status_text.text("❌ Processing failed!")
            st.error(f"❌ Document processing error: {str(e)}")
            logger.exception(f"Document processing failed for {file_name}")
            return False
        
        # Step 4: Knowledge Graph Integration
        status_text.text("🧠 Building knowledge graph...")
        progress_bar.progress(70)
        
        # Process into knowledge graph if available
        if st.session_state.graph_rag_manager and processed_doc.processing_status == ProcessingStatus.COMPLETED:
            try:
                # Create async event loop for graph processing
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Process document into knowledge graph
                graph_manager = st.session_state.graph_rag_manager.graph_manager
                if graph_manager:
                    # Store document in Neo4j and extract entities/relationships
                    doc_result = loop.run_until_complete(
                        graph_manager.process_document_to_graph(
                            document_id=processed_doc.id,
                            title=processed_doc.title,
                            content=processed_doc.content,
                            file_type=processed_doc.file_type,
                            metadata=processed_doc.metadata
                        )
                    )
                    
                    if doc_result.get('success', False):
                        status_text.text("✅ Knowledge graph updated!")
                    else:
                        status_text.text("⚠️ Knowledge graph processing had issues")
                        logger.warning(f"Graph processing issues: {doc_result.get('error', 'Unknown')}")
                
                loop.close()
                
            except Exception as e:
                logger.warning(f"Knowledge graph processing failed: {str(e)}")
                status_text.text("⚠️ Knowledge graph processing failed, continuing...")
        
        # Step 5: Vector Embeddings
        status_text.text("🔢 Creating vector embeddings...")
        progress_bar.progress(85)
        
        # Process into vector database if available
        if st.session_state.hybrid_rag_engine and processed_doc.processing_status == ProcessingStatus.COMPLETED:
            try:
                # Create embeddings for document chunks
                embedding_pipeline = st.session_state.hybrid_rag_engine.embedding_pipeline
                if embedding_pipeline:
                    # Create document embeddings from chunks
                    chunk_texts = [chunk.text for chunk in processed_doc.chunks]
                    embeddings = embedding_pipeline.create_document_embeddings_from_chunks(
                        document_id=processed_doc.id,
                        document_title=processed_doc.title,
                        chunks=chunk_texts,
                        metadata=processed_doc.metadata
                    )
                    
                    if embeddings:
                        # Separate document and chunk embeddings
                        doc_embeddings = [e for e in embeddings if e.embedding_type.value == "document"]
                        chunk_embeddings = [e for e in embeddings if e.embedding_type.value == "chunk"]
                        
                        # Process document embeddings
                        if doc_embeddings:
                            doc_results = embedding_pipeline.process_documents(doc_embeddings, embedding_type=EmbeddingType.DOCUMENT)
                            logger.info(f"Document embeddings: success={doc_results.success}, count={doc_results.processed_count}")
                        
                        # Process chunk embeddings
                        if chunk_embeddings:
                            chunk_results = embedding_pipeline.process_documents(chunk_embeddings, embedding_type=EmbeddingType.CHUNK)
                            logger.info(f"Chunk embeddings: success={chunk_results.success}, count={chunk_results.processed_count}")
                        
                        status_text.text("✅ Vector embeddings created!")
                    else:
                        status_text.text("⚠️ No embeddings created")
                
            except Exception as e:
                logger.warning(f"Vector embedding processing failed: {str(e)}")
                status_text.text("⚠️ Vector embedding failed, continuing...")
        
        # Step 6: Complete
        progress_bar.progress(100)
        
        if processed_doc.processing_status == ProcessingStatus.COMPLETED:
            status_text.text("🎉 Document ready for querying!")
            
            # Add to processed documents
            if 'processed_documents' not in st.session_state:
                st.session_state.processed_documents = []
            st.session_state.processed_documents.append(processed_doc)
            
            # Update graph statistics after processing
            if st.session_state.graph_rag_manager:
                try:
                    stats_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(stats_loop)
                    stats_result = stats_loop.run_until_complete(st.session_state.graph_rag_manager.get_graph_statistics())
                    if stats_result.get('success'):
                        # The statistics are nested in the result
                        if 'statistics' in stats_result:
                            st.session_state.system_status['graph_stats'] = stats_result['statistics']
                        else:
                            # Fallback: use the result directly if it has the right structure
                            st.session_state.system_status['graph_stats'] = stats_result
                        logger.info(f"Updated graph stats after document processing")
                    stats_loop.close()
                except Exception as e:
                    logger.warning(f"Failed to update graph statistics: {e}")
            
            # Display processing results
            display_processing_results(processed_doc)
            
            st.success(f"✅ Successfully processed {file_name} - Ready for Graph RAG and Hybrid RAG queries!")
            st.info("💡 Switch to the 'Query Interface' tab to ask questions about this document.")
            return True
            
        else:
            status_text.text("❌ Processing failed!")
            st.error(f"❌ Processing failed: {processed_doc.error_message}")
            return False
    
    except Exception as e:
        status_text.text("❌ Processing error!")
        st.error(f"❌ Error processing {file_name}: {str(e)}")
        logger.exception(f"Error processing file {file_name}")
        return False

def display_processing_results(processed_doc):
    """Display detailed processing results for a document."""
    st.subheader("📊 Processing Results")
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Word Count", f"{processed_doc.word_count:,}")
    with col2:
        st.metric("Character Count", f"{processed_doc.character_count:,}")
    with col3:
        st.metric("Chunks Created", processed_doc.chunk_count)
    with col4:
        st.metric("Processing Time", f"{processed_doc.processing_time_seconds:.2f}s")
    
    # Display chunk information
    if processed_doc.chunks:
        st.subheader("📝 Document Chunks")
        
        # Create DataFrame for chunks
        chunk_data = []
        for chunk in processed_doc.chunks[:5]:  # Show first 5 chunks
            chunk_data.append({
                "Chunk ID": chunk.id,
                "Word Count": chunk.word_count,
                "Preview": chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
            })
        
        if chunk_data:
            df = pd.DataFrame(chunk_data)
            st.dataframe(df, width="stretch")
            
            if len(processed_doc.chunks) > 5:
                st.info(f"Showing first 5 chunks. Total chunks: {len(processed_doc.chunks)}")

def display_processing_summary():
    """Display summary of all document processing."""
    if not st.session_state.processed_documents:
        st.info("No documents processed yet")
        return
    
    st.subheader("📈 Processing Summary")
    
    # Calculate summary statistics
    total_docs = len(st.session_state.processed_documents)
    total_words = sum(doc.word_count for doc in st.session_state.processed_documents)
    total_chunks = sum(doc.chunk_count for doc in st.session_state.processed_documents)
    avg_processing_time = sum(doc.processing_time_seconds for doc in st.session_state.processed_documents) / total_docs
    
    # Display metrics
    st.metric("Total Documents", total_docs)
    st.metric("Total Words", f"{total_words:,}")
    st.metric("Total Chunks", total_chunks)
    st.metric("Avg Processing Time", f"{avg_processing_time:.2f}s")
    
    # Status breakdown
    status_counts = {}
    for doc in st.session_state.processed_documents:
        status = doc.processing_status.value
        status_counts[status] = status_counts.get(status, 0) + 1
    
    if status_counts:
        st.subheader("Status Breakdown")
        for status, count in status_counts.items():
            if status == "completed":
                st.success(f"✅ {status.title()}: {count}")
            elif status == "failed":
                st.error(f"❌ {status.title()}: {count}")
            else:
                st.info(f"⏳ {status.title()}: {count}")

def display_processed_documents():
    """Display list of all processed documents."""
    if not st.session_state.processed_documents:
        return
    
    st.header("📚 Processed Documents")
    
    # Create DataFrame for documents
    doc_data = []
    for doc in st.session_state.processed_documents:
        doc_data.append({
            "Title": doc.title,
            "Filename": doc.filename,
            "File Type": doc.file_type,
            "Status": doc.processing_status.value,
            "Word Count": f"{doc.word_count:,}",
            "Chunks": doc.chunk_count,
            "Processing Time": f"{doc.processing_time_seconds:.2f}s",
            "Created": doc.created_at.strftime("%Y-%m-%d %H:%M:%S")
        })
    
    if doc_data:
        df = pd.DataFrame(doc_data)
        st.dataframe(df, width="stretch")
        
        # Clear documents button
        if st.button("🗑️ Clear All Documents"):
            st.session_state.processed_documents.clear()
            st.session_state.document_processor.clear_processed_documents()
            st.success("✅ All documents cleared")
            st.rerun()

def query_interface():
    """Natural language query interface with RAG method selection."""
    st.header("🔍 Query Interface")
    
    if not st.session_state.initialized:
        st.warning("⚠️ Please initialize the system first using the sidebar.")
        return
    
    if not st.session_state.processed_documents:
        st.warning("⚠️ Please upload and process some documents first to enable querying.")
        return
    
    # Check for demo query from workshop
    demo_query_ready = (hasattr(st.session_state, 'demo_ready') and 
                       st.session_state.demo_ready and
                       hasattr(st.session_state, 'demo_query_id') and 
                       hasattr(st.session_state, 'demo_method') and 
                       st.session_state.demo_query_id and 
                       st.session_state.demo_method and
                       hasattr(st.session_state, 'last_query') and
                       st.session_state.last_query)
    
    if demo_query_ready:
        st.info(f"🎓 **Workshop Demo Query Ready**: {st.session_state.demo_method}")
        st.info(f"📝 **Query**: {st.session_state.last_query}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Execute Demo Query", type="primary"):
                if st.session_state.demo_method == "Graph RAG":
                    execute_query(st.session_state.last_query, "Graph RAG", 15, True, 0.7)
                elif st.session_state.demo_method == "Hybrid RAG":
                    execute_query(st.session_state.last_query, "Hybrid RAG", 15, True, 0.7)
                elif st.session_state.demo_method == "Compare":
                    compare_rag_methods(st.session_state.last_query, 15, True, 0.7)
                
                # Clear demo state
                st.session_state.demo_query_id = None
                st.session_state.demo_method = None
                st.session_state.demo_ready = False
                st.rerun()
        
        with col2:
            if st.button("❌ Clear Demo Query"):
                st.session_state.demo_query_id = None
                st.session_state.demo_method = None
                st.session_state.demo_ready = False
                st.rerun()
        
        st.divider()
    
    # Query input section
    st.subheader("💬 Ask a Question")
    
    # Create columns for query input and method selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Natural language query input
        query = st.text_area(
            "Enter your question:",
            value=st.session_state.last_query,
            height=100,
            placeholder="Ask anything about your uploaded documents...\n\nExample: 'What are the main topics discussed in the documents?'"
        )
        
        # Sample queries
        st.markdown("**💡 Sample Questions:**")
        sample_queries = [
            "What are the main topics discussed in the documents?",
            "Who are the key people mentioned?",
            "What organizations are referenced?",
            "Summarize the key findings",
            "What relationships exist between the entities?"
        ]
        
        cols = st.columns(len(sample_queries))
        for i, sample_query in enumerate(sample_queries):
            with cols[i]:
                if st.button(f"📝 {sample_query[:20]}...", key=f"sample_{i}", help=sample_query):
                    st.session_state.last_query = sample_query
                    st.rerun()
    
    with col2:
        # RAG method selection
        st.subheader("🎯 RAG Method")
        
        # Initialize selected_rag_method in session state if not present
        if 'selected_rag_method' not in st.session_state:
            st.session_state.selected_rag_method = "Hybrid RAG"
        
        # Update selected method based on demo selection
        if demo_query_ready and st.session_state.demo_method in ["Graph RAG", "Hybrid RAG"]:
            st.session_state.selected_rag_method = st.session_state.demo_method
        
        # Determine default index based on session state
        default_index = 0 if st.session_state.selected_rag_method == "Graph RAG" else 1
        
        rag_method = st.radio(
            "Choose approach:",
            options=["Graph RAG", "Hybrid RAG"],
            index=default_index,
            help="Graph RAG uses only knowledge graph traversal. Hybrid RAG combines graph and vector search."
        )
        
        # Update session state when user manually changes selection
        st.session_state.selected_rag_method = rag_method
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            max_results = st.slider("Max Results", 5, 50, 15)
            include_reasoning = st.checkbox("Include Reasoning Steps", value=True)
            similarity_threshold = st.slider("Similarity Threshold", 0.1, 1.0, 0.7, 0.1)
    
    # Query execution buttons
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("🚀 Ask Question", disabled=not query.strip()):
            if query.strip():
                st.session_state.last_query = query
                execute_query(query, rag_method, max_results, include_reasoning, similarity_threshold)
    
    with col_btn2:
        if st.button("⚖️ Compare Both Methods", disabled=not query.strip()):
            if query.strip():
                st.session_state.last_query = query
                compare_rag_methods(query, max_results, include_reasoning, similarity_threshold)
    
    # Display previous results if available
    if st.session_state.query_results:
        display_query_results()

def compare_rag_methods(query: str, max_results: int, include_reasoning: bool, similarity_threshold: float):
    """Compare Graph RAG and Hybrid RAG side by side."""
    
    st.subheader("⚖️ RAG Method Comparison")
    
    # Check if both methods are available
    graph_available = st.session_state.graph_rag_manager is not None
    hybrid_available = st.session_state.hybrid_rag_engine is not None
    
    if not graph_available and not hybrid_available:
        st.error("❌ Neither Graph RAG nor Hybrid RAG is available. Please check system initialization.")
        return
    
    # Create columns for side-by-side comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🕸️ Graph RAG")
        if graph_available:
            with st.spinner("Processing with Graph RAG..."):
                try:
                    start_time = time.time()
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    graph_result = execute_graph_rag_query(loop, query, max_results, include_reasoning)
                    loop.close()
                    processing_time = time.time() - start_time
                    
                    display_comparison_result(graph_result, "Graph RAG")
                    
                    # Record performance metrics
                    try:
                        from utils.workshop_demo import workshop_demo, PerformanceMetrics
                        
                        confidence = graph_result.get('confidence', 0.0) if graph_result.get('success', False) else 0.0
                        source_count = len(graph_result.get('sources', [])) if graph_result.get('success', False) else 0
                        context = graph_result.get('context', {})
                        entity_count = context.get('total_entities', 0)
                        relationship_count = context.get('total_relationships', 0)
                        
                        metrics = PerformanceMetrics(
                            method="Graph RAG",
                            query=query,
                            response_time=processing_time,
                            confidence=confidence,
                            source_count=source_count,
                            entity_count=entity_count,
                            relationship_count=relationship_count,
                            success=graph_result.get('success', False),
                            error_message=graph_result.get('error') if not graph_result.get('success', False) else None
                        )
                        
                        workshop_demo.record_performance(metrics)
                        
                    except ImportError:
                        pass
                    
                except Exception as e:
                    st.error(f"Graph RAG failed: {str(e)}")
        else:
            st.warning("⚠️ Graph RAG not available")
            st.info("Requires: Neo4j connection, Graph Manager, and Graph RAG Manager")
    
    with col2:
        st.markdown("### 🔄 Hybrid RAG")
        if hybrid_available:
            with st.spinner("Processing with Hybrid RAG..."):
                try:
                    start_time = time.time()
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    hybrid_result = execute_hybrid_rag_query(loop, query, max_results, include_reasoning, similarity_threshold)
                    loop.close()
                    processing_time = time.time() - start_time
                    
                    display_comparison_result(hybrid_result, "Hybrid RAG")
                    
                    # Record performance metrics
                    try:
                        from utils.workshop_demo import workshop_demo, PerformanceMetrics
                        
                        confidence = hybrid_result.get('confidence', 0.0) if hybrid_result.get('success', False) else 0.0
                        source_count = len(hybrid_result.get('sources', [])) if hybrid_result.get('success', False) else 0
                        context = hybrid_result.get('context', {})
                        
                        metrics = PerformanceMetrics(
                            method="Hybrid RAG",
                            query=query,
                            response_time=processing_time,
                            confidence=confidence,
                            source_count=source_count,
                            entity_count=0,  # Not applicable for Hybrid RAG
                            relationship_count=0,  # Not applicable for Hybrid RAG
                            success=hybrid_result.get('success', False),
                            error_message=hybrid_result.get('error') if not hybrid_result.get('success', False) else None
                        )
                        
                        workshop_demo.record_performance(metrics)
                        
                    except ImportError:
                        pass
                    
                except Exception as e:
                    st.error(f"Hybrid RAG failed: {str(e)}")
        else:
            st.warning("⚠️ Hybrid RAG not available")
            st.info("Requires: Database connections, Graph Manager, and Embedding Pipeline")

def display_comparison_result(result: Dict[str, Any], method_name: str):
    """Display a condensed result for comparison."""
    if not result.get('success', False):
        st.error(f"❌ {method_name} failed: {result.get('error', 'Unknown error')}")
        return
    
    # Confidence score
    confidence = result.get('confidence', 0.0)
    st.metric("Confidence", f"{confidence:.1%}")
    
    # Processing time
    processing_time = result.get('processing_time_seconds', 0.0)
    st.metric("Processing Time", f"{processing_time:.2f}s")
    
    # Answer preview
    answer = result.get('answer', 'No answer generated')
    if len(answer) > 200:
        answer = answer[:200] + "..."
    
    st.markdown("**Answer Preview:**")
    st.markdown(f"*{answer}*")
    
    # Source count
    sources = result.get('sources', [])
    st.metric("Sources Found", len(sources))
    
    # Method-specific metrics
    context = result.get('context', {})
    if method_name == "Graph RAG":
        entities = context.get('total_entities', 0)
        relationships = context.get('total_relationships', 0)
        st.metric("Graph Entities", entities)
        st.metric("Relationships", relationships)
    elif method_name == "Hybrid RAG":
        fusion_method = context.get('fusion_method', 'Unknown')
        total_sources = context.get('total_sources', 0)
        st.metric("Fusion Method", fusion_method)
        st.metric("Fused Sources", total_sources)

def execute_query(query: str, rag_method: str, max_results: int, include_reasoning: bool, similarity_threshold: float):
    """Execute a query using the selected RAG method."""
    
    with st.spinner(f"🔍 Processing query using {rag_method}..."):
        try:
            start_time = time.time()
            
            # Create async event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            if rag_method == "Graph RAG":
                result = execute_graph_rag_query(loop, query, max_results, include_reasoning)
            else:  # Hybrid RAG
                result = execute_hybrid_rag_query(loop, query, max_results, include_reasoning, similarity_threshold)
            
            loop.close()
            
            processing_time = time.time() - start_time
            
            # Store results in session state
            st.session_state.query_results = {
                'query': query,
                'method': rag_method,
                'result': result,
                'timestamp': datetime.now(),
                'processing_time': processing_time
            }
            
            # Record performance metrics for workshop demo
            try:
                from utils.workshop_demo import workshop_demo, PerformanceMetrics
                
                # Extract metrics from result
                confidence = result.get('confidence', 0.0) if result.get('success', False) else 0.0
                source_count = len(result.get('sources', [])) if result.get('success', False) else 0
                context = result.get('context', {})
                entity_count = context.get('total_entities', 0) if rag_method == "Graph RAG" else 0
                relationship_count = context.get('total_relationships', 0) if rag_method == "Graph RAG" else 0
                
                metrics = PerformanceMetrics(
                    method=rag_method,
                    query=query,
                    response_time=processing_time,
                    confidence=confidence,
                    source_count=source_count,
                    entity_count=entity_count,
                    relationship_count=relationship_count,
                    success=result.get('success', False),
                    error_message=result.get('error') if not result.get('success', False) else None
                )
                
                workshop_demo.record_performance(metrics)
                
            except ImportError:
                pass  # Workshop demo not available
            
            st.success(f"✅ Query processed in {processing_time:.2f} seconds")
            
        except Exception as e:
            st.error(f"❌ Query processing failed: {str(e)}")
            logger.exception(f"Query processing error: {query}")

def execute_graph_rag_query(loop, query: str, max_results: int, include_reasoning: bool) -> Dict[str, Any]:
    """Execute Graph RAG query."""
    if not st.session_state.graph_rag_manager:
        return {
            'success': False,
            'error': 'Graph RAG Manager not available. Please check system initialization.',
            'query': query,
            'method': 'graph_rag'
        }
    
    try:
        result = loop.run_until_complete(
            st.session_state.graph_rag_manager.query(
                question=query,
                max_context_nodes=max_results,
                max_traversal_depth=2,
                use_text_to_cypher=True,
                include_reasoning=include_reasoning
            )
        )
        return result
    except Exception as e:
        return {
            'success': False,
            'error': f'Graph RAG query failed: {str(e)}',
            'query': query,
            'method': 'graph_rag'
        }

def execute_hybrid_rag_query(loop, query: str, max_results: int, include_reasoning: bool, similarity_threshold: float) -> Dict[str, Any]:
    """Execute Hybrid RAG query."""
    if not st.session_state.hybrid_rag_engine:
        return {
            'success': False,
            'error': 'Hybrid RAG Engine not available. Please check system initialization.',
            'query': query,
            'method': 'hybrid_rag'
        }
    
    try:
        result = loop.run_until_complete(
            st.session_state.hybrid_rag_engine.query(
                question=query,
                enable_graph_search=True,
                enable_vector_search=True,
                max_results=max_results,
                similarity_threshold=similarity_threshold
            )
        )
        
        # Convert HybridRAGResult to dict for consistency
        if hasattr(result, '__dict__'):
            result_dict = {
                'success': result.success,
                'query': result.query,
                'answer': result.answer,
                'method': result.method,
                'confidence': result.confidence,
                'sources': result.sources,
                'processing_time_seconds': result.processing_time_seconds,
                'error': result.error_message if not result.success else None
            }
            
            if include_reasoning and hasattr(result, 'reasoning_steps'):
                result_dict['reasoning_steps'] = result.reasoning_steps
            
            if hasattr(result, 'hybrid_context') and result.hybrid_context:
                result_dict['context'] = {
                    'fusion_method': result.hybrid_context.fusion_method.value if hasattr(result.hybrid_context.fusion_method, 'value') else str(result.hybrid_context.fusion_method),
                    'total_sources': result.hybrid_context.total_unique_sources,
                    'deduplication_stats': result.hybrid_context.deduplication_stats
                }
            
            return result_dict
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Hybrid RAG query failed: {str(e)}',
            'query': query,
            'method': 'hybrid_rag'
        }

def display_query_results():
    """Display comprehensive query results with source attribution."""
    results = st.session_state.query_results
    
    st.header("📋 Query Results")
    
    # Display query info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Method Used", results['method'])
    with col2:
        st.metric("Processing Time", f"{results['processing_time']:.2f}s")
    with col3:
        timestamp = results['timestamp'].strftime("%H:%M:%S")
        st.metric("Query Time", timestamp)
    
    # Display the question
    st.subheader("❓ Question")
    st.info(f"**{results['query']}**")
    
    result = results['result']
    
    if not result.get('success', False):
        error_msg = result.get('error', 'Unknown error')
        st.error(f"❌ Query failed: {error_msg}")
        
        # Provide helpful suggestions based on the error and method
        if results['method'] == 'Graph RAG' and 'no relevant information found' in error_msg.lower():
            st.info("💡 **Suggestion**: Try using **Hybrid RAG** instead, which combines graph and vector search for better results.")
            st.info("🔧 **Tip**: Graph RAG works best when entities are properly extracted. The knowledge graph may need more documents or better entity extraction.")
        elif 'knowledge graph' in error_msg.lower():
            st.info("💡 **Suggestion**: The knowledge graph may be empty or entities weren't properly extracted. Try uploading more documents or using Hybrid RAG.")
        
        return
    
    # Display answer
    st.subheader("💡 Answer")
    answer = result.get('answer', 'No answer generated')
    st.markdown(f"""
    <div style="background-color: #f0f8ff; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;">
        <p style="margin: 0; font-size: 1.1rem; line-height: 1.6;">{answer}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display confidence score
    confidence = result.get('confidence', 0.0)
    st.subheader("🎯 Confidence Score")
    
    # Create confidence bar
    confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem;">
        <div style="display: flex; align-items: center;">
            <div style="flex-grow: 1; background-color: #e9ecef; height: 20px; border-radius: 10px; margin-right: 10px;">
                <div style="width: {confidence*100}%; height: 100%; background-color: {confidence_color}; border-radius: 10px;"></div>
            </div>
            <span style="font-weight: bold;">{confidence:.1%}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display sources with attribution
    sources = result.get('sources', [])
    if sources:
        display_sources_section(sources, results['method'])
    
    # Display context information
    context = result.get('context', {})
    if context:
        display_context_section(context, results['method'])
    
    # Display reasoning steps if available
    reasoning_steps = result.get('reasoning_steps', [])
    if reasoning_steps:
        display_reasoning_section(reasoning_steps)
    
    # Display graph traversal demonstration
    if results['method'] == "Graph RAG" and context:
        display_graph_traversal_demo(context)

def display_sources_section(sources: List[Dict[str, Any]], method: str):
    """Display source attribution with expandable details."""
    st.subheader("📚 Sources")
    
    if not sources:
        st.info("No sources found")
        return
    
    # Create tabs for different source types
    source_types = set()
    for source in sources:
        source_type = source.get('source_type', source.get('type', 'unknown'))
        source_types.add(source_type)
    
    if len(source_types) > 1:
        tabs = st.tabs([f"{stype.replace('_', ' ').title()}" for stype in sorted(source_types)])
        
        for i, source_type in enumerate(sorted(source_types)):
            with tabs[i]:
                display_sources_by_type(sources, source_type)
    else:
        display_sources_by_type(sources, list(source_types)[0] if source_types else 'all')

def display_sources_by_type(sources: List[Dict[str, Any]], filter_type: str):
    """Display sources filtered by type."""
    filtered_sources = sources
    if filter_type != 'all':
        filtered_sources = [s for s in sources if s.get('source_type', s.get('type', 'unknown')) == filter_type]
    
    for i, source in enumerate(filtered_sources[:10], 1):  # Limit to top 10 sources
        with st.expander(f"📄 Source {i}: {source.get('title', 'Unknown Title')}", expanded=i <= 3):
            
            # Source metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Score", f"{source.get('score', 0.0):.3f}")
            with col2:
                st.metric("Type", source.get('source_type', source.get('type', 'Unknown')))
            with col3:
                doc_id = source.get('document_id', source.get('id', 'Unknown'))
                st.metric("Document ID", doc_id[:12] + "..." if len(doc_id) > 12 else doc_id)
            
            # Source content
            content = source.get('content', 'No content available')
            if len(content) > 500:
                content = content[:500] + "..."
            
            st.markdown("**Content:**")
            st.markdown(f"```\n{content}\n```")
            
            # Additional metadata
            if source.get('metadata'):
                st.markdown("**Metadata:**")
                st.json(source['metadata'])

def display_context_section(context: Dict[str, Any], method: str):
    """Display context information based on RAG method."""
    st.subheader("🧠 Context Information")
    
    if method == "Graph RAG":
        display_graph_context(context)
    elif method == "Hybrid RAG":
        display_hybrid_context(context)

def display_graph_context(context: Dict[str, Any]):
    """Display Graph RAG specific context."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Entities Found", context.get('total_entities', 0))
    with col2:
        st.metric("Relationships", context.get('total_relationships', 0))
    with col3:
        st.metric("Traversal Depth", context.get('traversal_depth', 0))
    
    # Display Cypher query if available
    cypher_query = context.get('cypher_query')
    if cypher_query:
        st.subheader("🔍 Generated Cypher Query")
        st.code(cypher_query, language='cypher')

def display_hybrid_context(context: Dict[str, Any]):
    """Display Hybrid RAG specific context."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Fusion Method", context.get('fusion_method', 'Unknown'))
    with col2:
        st.metric("Total Sources", context.get('total_sources', 0))
    
    # Display deduplication stats
    dedup_stats = context.get('deduplication_stats', {})
    if dedup_stats:
        st.subheader("📊 Deduplication Statistics")
        dedup_col1, dedup_col2, dedup_col3 = st.columns(3)
        
        with dedup_col1:
            st.metric("Input Sources", dedup_stats.get('total_input', 0))
        with dedup_col2:
            st.metric("Duplicates Removed", dedup_stats.get('document_duplicates', 0) + dedup_stats.get('content_duplicates', 0))
        with dedup_col3:
            st.metric("Final Sources", dedup_stats.get('final_count', 0))

def display_reasoning_section(reasoning_steps: List[str]):
    """Display reasoning steps in an expandable section."""
    with st.expander("🧠 Reasoning Steps", expanded=False):
        for i, step in enumerate(reasoning_steps, 1):
            st.markdown(f"**Step {i}:** {step}")

def display_graph_traversal_demo(context: Dict[str, Any]):
    """Display graph traversal demonstration and explanation."""
    with st.expander("🕸️ Graph Traversal Demonstration", expanded=False):
        st.markdown("""
        **How Graph RAG Works:**
        
        1. **Query Analysis**: The natural language question is analyzed to identify key entities and concepts
        2. **Cypher Generation**: A Cypher query is generated to traverse the knowledge graph
        3. **Graph Traversal**: The system follows relationships between entities to gather relevant context
        4. **Context Assembly**: Related entities and their connections are assembled into a coherent context
        5. **Answer Generation**: The LLM uses the graph context to generate a comprehensive answer
        """)
        
        # Display entities and relationships if available
        entities = context.get('entities', [])
        relationships = context.get('relationships', [])
        
        if entities:
            st.subheader("🔗 Connected Entities")
            entity_names = [entity.get('name', 'Unknown') for entity in entities[:10]]
            st.write(", ".join(entity_names))
        
        if relationships:
            st.subheader("↔️ Relationships")
            rel_descriptions = [f"{rel.get('source', 'A')} → {rel.get('type', 'RELATED')} → {rel.get('target', 'B')}" 
                             for rel in relationships[:5]]
            for rel in rel_descriptions:
                st.write(f"• {rel}")

def workshop_demonstration_interface():
    """Workshop demonstration interface with sample data and examples."""
    st.header("🎓 Workshop Demonstration")
    
    # Import workshop demo utilities
    try:
        # Try different import paths
        try:
            from utils.workshop_demo import workshop_demo
        except ImportError:
            # Try absolute import
            import sys
            from pathlib import Path
            current_dir = Path(__file__).parent
            sys.path.insert(0, str(current_dir))
            from utils.workshop_demo import workshop_demo
        
        # Create sub-tabs for workshop features
        demo_tab1, demo_tab2, demo_tab3 = st.tabs(["📚 Sample Documents", "📝 Example Queries", "📊 Performance Dashboard"])
        
        with demo_tab1:
            workshop_demo.display_sample_documents_interface()
        
        with demo_tab2:
            workshop_demo.display_sample_queries_interface()
        
        with demo_tab3:
            workshop_demo.display_performance_dashboard()
            
    except ImportError as e:
        st.error("❌ Workshop demo utilities not available")
        st.error(f"Import error: {str(e)}")
        
        # Provide fallback content
        st.info("📚 **Sample Documents Available:**")
        st.write("- AI Research Paper (ai_research_paper.md)")
        st.write("- Tech Company Profiles (tech_company_profiles.txt)")
        st.write("- University Research Network (university_research_network.md)")
        st.write("- Workshop Example Queries (workshop_example_queries.md)")
        
        st.info("💡 **To enable full workshop features:**")
        st.code("pip install plotly", language="bash")
        
        return
    except Exception as e:
        st.error(f"❌ Error loading workshop demo: {str(e)}")
        return

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Display system status in sidebar
    display_system_status()
    
    # Main content area
    if st.session_state.initialized:
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["📄 Document Upload", "🔍 Query Interface", "🎓 Workshop Demo"])
        
        with tab1:
            document_upload_interface()
        
        with tab2:
            query_interface()
        
        with tab3:
            workshop_demonstration_interface()
    else:
        st.info("👈 Please initialize the system using the sidebar to get started.")

if __name__ == "__main__":
    main()
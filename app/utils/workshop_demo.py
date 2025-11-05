"""
Workshop demonstration utilities for Knowledge Graph RAG System.
Provides sample data, example queries, and performance comparison tools.
"""

import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Optional imports
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Optional imports for enhanced visualizations
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

@dataclass
class QueryExample:
    """Represents an example query for workshop demonstration."""
    id: str
    title: str
    question: str
    category: str
    description: str
    expected_graph_behavior: str
    expected_hybrid_behavior: str
    difficulty: str  # "beginner", "intermediate", "advanced"
    estimated_time: float  # seconds

@dataclass
class PerformanceMetrics:
    """Performance metrics for RAG method comparison."""
    method: str
    query: str
    response_time: float
    confidence: float
    source_count: int
    entity_count: int
    relationship_count: int
    success: bool
    error_message: Optional[str] = None

def requires_streamlit(func):
    """Decorator to check if streamlit is available."""
    def wrapper(*args, **kwargs):
        if not STREAMLIT_AVAILABLE:
            print(f"Streamlit not available for {func.__name__}")
            return None
        return func(*args, **kwargs)
    return wrapper

class WorkshopDemo:
    """Workshop demonstration utilities and sample data manager."""
    
    def __init__(self):
        self.sample_queries = self._load_sample_queries()
        self.performance_history = []
    
    def _load_sample_queries(self) -> List[QueryExample]:
        """Load predefined sample queries for workshop demonstration."""
        return [
            QueryExample(
                id="entity_person",
                title="Person Identification",
                question="What researchers are mentioned in the documents?",
                category="Entity-Focused",
                description="Demonstrates entity identification and relationship traversal",
                expected_graph_behavior="Identifies researcher entities, traverses to affiliations and research areas",
                expected_hybrid_behavior="Uses vector search for semantic content about researchers and their work",
                difficulty="beginner",
                estimated_time=3.0
            ),
            QueryExample(
                id="entity_organization",
                title="Organization Analysis",
                question="What organizations and companies are mentioned in the documents?",
                category="Entity-Focused",
                description="Shows how to find organizations by their activities",
                expected_graph_behavior="Traverses organization entities and their relationships",
                expected_hybrid_behavior="Semantic search for organization mentions and their activities",
                difficulty="beginner",
                estimated_time=4.0
            ),
            QueryExample(
                id="relationship_collaboration",
                title="Collaboration Networks",
                question="What collaborations and partnerships are mentioned in the documents?",
                category="Relationship-Focused",
                description="Demonstrates relationship traversal and network analysis",
                expected_graph_behavior="Traverses collaboration relationships between entities",
                expected_hybrid_behavior="Searches for collaboration and partnership mentions",
                difficulty="intermediate",
                estimated_time=5.0
            ),
            QueryExample(
                id="relationship_technology",
                title="Technology Connections",
                question="What are the main topics and technologies discussed in the documents?",
                category="Relationship-Focused",
                description="Shows conceptual relationships between technologies",
                expected_graph_behavior="Traverses concept relationships and technology connections",
                expected_hybrid_behavior="Semantic similarity for related concepts and technologies",
                difficulty="intermediate",
                estimated_time=4.5
            ),
            QueryExample(
                id="multihop_indirect",
                title="Indirect Connections",
                question="What is the connection between Stanford University and Neo4j?",
                category="Multi-Hop Reasoning",
                description="Demonstrates multi-hop traversal for indirect relationships",
                expected_graph_behavior="Multi-hop: Stanford → researchers → companies → Neo4j",
                expected_hybrid_behavior="Searches for documents mentioning both, finds indirect connections",
                difficulty="advanced",
                estimated_time=6.0
            ),
            QueryExample(
                id="multihop_evolution",
                title="Technology Evolution",
                question="How did transformer architectures lead to modern RAG systems?",
                category="Multi-Hop Reasoning",
                description="Shows technological evolution through relationship chains",
                expected_graph_behavior="Traces transformers → LLMs → RAG evolution path",
                expected_hybrid_behavior="Historical documents + timeline relationships",
                difficulty="advanced",
                estimated_time=7.0
            ),
            QueryExample(
                id="comparison_databases",
                title="Technology Comparison",
                question="What are the differences between vector databases like Qdrant and graph databases like Neo4j?",
                category="Comparative Analysis",
                description="Compares different technology types and their properties",
                expected_graph_behavior="Compares properties and use cases through graph relationships",
                expected_hybrid_behavior="Direct comparison documents + technical relationships",
                difficulty="intermediate",
                estimated_time=5.5
            ),
            QueryExample(
                id="comparison_institutions",
                title="Research Institution Comparison",
                question="How do the AI research focuses of MIT and Stanford differ?",
                category="Comparative Analysis",
                description="Analyzes and compares institutional research areas",
                expected_graph_behavior="Compares research areas and faculty specializations",
                expected_hybrid_behavior="Research description documents + semantic analysis",
                difficulty="intermediate",
                estimated_time=6.0
            ),
            QueryExample(
                id="analysis_industry",
                title="Industry Landscape Analysis",
                question="Which AI companies are founded by university researchers and what technologies do they focus on?",
                category="Complex Analysis",
                description="Complex query requiring multiple relationship types",
                expected_graph_behavior="Researchers → founded → companies → develops → technologies",
                expected_hybrid_behavior="Company founder information + technology focus analysis",
                difficulty="advanced",
                estimated_time=8.0
            ),
            QueryExample(
                id="analysis_trends",
                title="Research Impact Assessment",
                question="What are the most influential research areas in modern AI according to the documents?",
                category="Complex Analysis",
                description="Analytical query requiring aggregation and ranking",
                expected_graph_behavior="Analyzes concept centrality and relationship density",
                expected_hybrid_behavior="Semantic analysis + graph metrics for impact assessment",
                difficulty="advanced",
                estimated_time=7.5
            )
        ]
    
    def get_queries_by_category(self, category: str = None) -> List[QueryExample]:
        """Get sample queries filtered by category."""
        if category is None:
            return self.sample_queries
        return [q for q in self.sample_queries if q.category == category]
    
    def get_queries_by_difficulty(self, difficulty: str = None) -> List[QueryExample]:
        """Get sample queries filtered by difficulty."""
        if difficulty is None:
            return self.sample_queries
        return [q for q in self.sample_queries if q.difficulty == difficulty]
    
    def get_query_categories(self) -> List[str]:
        """Get list of all query categories."""
        return list(set(q.category for q in self.sample_queries))
    
    def get_query_by_id(self, query_id: str) -> Optional[QueryExample]:
        """Get a specific query by ID."""
        for query in self.sample_queries:
            if query.id == query_id:
                return query
        return None
    
    def record_performance(self, metrics: PerformanceMetrics):
        """Record performance metrics for comparison."""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'method': metrics.method,
            'query': metrics.query,
            'response_time': metrics.response_time,
            'confidence': metrics.confidence,
            'source_count': metrics.source_count,
            'entity_count': metrics.entity_count,
            'relationship_count': metrics.relationship_count,
            'success': metrics.success,
            'error_message': metrics.error_message
        })
    
    def get_performance_comparison(self, query: str) -> Dict[str, Any]:
        """Get performance comparison for a specific query."""
        query_results = [r for r in self.performance_history if r['query'] == query]
        
        if len(query_results) < 2:
            return {'error': 'Need results from both methods for comparison'}
        
        # Group by method
        graph_results = [r for r in query_results if r['method'] == 'Graph RAG']
        hybrid_results = [r for r in query_results if r['method'] == 'Hybrid RAG']
        
        comparison = {
            'query': query,
            'graph_rag': {
                'avg_response_time': sum(r['response_time'] for r in graph_results) / len(graph_results) if graph_results else 0,
                'avg_confidence': sum(r['confidence'] for r in graph_results) / len(graph_results) if graph_results else 0,
                'avg_sources': sum(r['source_count'] for r in graph_results) / len(graph_results) if graph_results else 0,
                'success_rate': sum(1 for r in graph_results if r['success']) / len(graph_results) if graph_results else 0,
                'runs': len(graph_results)
            },
            'hybrid_rag': {
                'avg_response_time': sum(r['response_time'] for r in hybrid_results) / len(hybrid_results) if hybrid_results else 0,
                'avg_confidence': sum(r['confidence'] for r in hybrid_results) / len(hybrid_results) if hybrid_results else 0,
                'avg_sources': sum(r['source_count'] for r in hybrid_results) / len(hybrid_results) if hybrid_results else 0,
                'success_rate': sum(1 for r in hybrid_results if r['success']) / len(hybrid_results) if hybrid_results else 0,
                'runs': len(hybrid_results)
            }
        }
        
        return comparison
    
    @requires_streamlit
    def display_sample_queries_interface(self):
        """Display interactive sample queries interface in Streamlit."""
        st.subheader("📝 Sample Queries for Workshop")
        
        # Check system status and provide guidance
        system_ready = (hasattr(st.session_state, 'initialized') and 
                       st.session_state.initialized and 
                       hasattr(st.session_state, 'processed_documents') and 
                       st.session_state.processed_documents)
        
        if not system_ready:
            st.warning("⚠️ **Setup Required**: To use these sample queries, you need to:")
            if not hasattr(st.session_state, 'initialized') or not st.session_state.initialized:
                st.write("1. 🔧 Initialize the system using the sidebar")
            else:
                st.write("1. ✅ System initialized")
            
            if not hasattr(st.session_state, 'processed_documents') or not st.session_state.processed_documents:
                st.write("2. 📄 Upload and process documents in the 'Document Upload' tab")
            else:
                st.write("2. ✅ Documents processed")
            
            st.info("💡 **Tip**: You can use the sample documents provided in the 'Sample Documents' tab above!")
        else:
            st.success("✅ **System Ready**: Click any button below to try sample queries!")
            
            # Show current document count
            doc_count = len(st.session_state.processed_documents)
            st.info(f"📚 **{doc_count} documents** available for querying")
            
            st.info("💡 **Tip**: Try **Hybrid RAG** first for best results, as it combines both graph and vector search!")
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            categories = ["All"] + self.get_query_categories()
            selected_category = st.selectbox("Category", categories)
        
        with col2:
            difficulties = ["All", "beginner", "intermediate", "advanced"]
            selected_difficulty = st.selectbox("Difficulty", difficulties)
        
        with col3:
            sort_by = st.selectbox("Sort by", ["Category", "Difficulty", "Estimated Time"])
        
        # Filter queries
        filtered_queries = self.sample_queries
        if selected_category != "All":
            filtered_queries = [q for q in filtered_queries if q.category == selected_category]
        if selected_difficulty != "All":
            filtered_queries = [q for q in filtered_queries if q.difficulty == selected_difficulty]
        
        # Sort queries
        if sort_by == "Category":
            filtered_queries.sort(key=lambda x: x.category)
        elif sort_by == "Difficulty":
            difficulty_order = {"beginner": 1, "intermediate": 2, "advanced": 3}
            filtered_queries.sort(key=lambda x: difficulty_order.get(x.difficulty, 4))
        elif sort_by == "Estimated Time":
            filtered_queries.sort(key=lambda x: x.estimated_time)
        
        # Display queries
        for query in filtered_queries:
            with st.expander(f"🔍 {query.title} ({query.difficulty.title()})", expanded=False):
                
                # Query info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Category", query.category)
                with col2:
                    st.metric("Difficulty", query.difficulty.title())
                with col3:
                    st.metric("Est. Time", f"{query.estimated_time:.1f}s")
                
                # Question
                st.markdown("**Question:**")
                st.info(query.question)
                
                # Description
                st.markdown("**Description:**")
                st.write(query.description)
                
                # Expected behaviors
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Graph RAG Expected:**")
                    st.write(query.expected_graph_behavior)
                
                with col2:
                    st.markdown("**Hybrid RAG Expected:**")
                    st.write(query.expected_hybrid_behavior)
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                
                # Check if system is ready for queries
                system_ready = (hasattr(st.session_state, 'initialized') and 
                              st.session_state.initialized and 
                              hasattr(st.session_state, 'processed_documents') and 
                              st.session_state.processed_documents)
                
                # Define callback functions to set session state
                # Use default arguments to capture current query value (avoid closure issue)
                def set_hybrid_demo(q=query):
                    st.session_state.last_query = q.question
                    st.session_state.demo_query_id = q.id
                    st.session_state.demo_method = "Hybrid RAG"
                    st.session_state.demo_ready = True
                
                def set_graph_demo(q=query):
                    st.session_state.last_query = q.question
                    st.session_state.demo_query_id = q.id
                    st.session_state.demo_method = "Graph RAG"
                    st.session_state.demo_ready = True
                
                def set_compare_demo(q=query):
                    st.session_state.last_query = q.question
                    st.session_state.demo_query_id = q.id
                    st.session_state.demo_method = "Compare"
                    st.session_state.demo_ready = True
                
                with col1:
                    button_clicked = st.button(
                        f"🔄 Try Hybrid RAG (Recommended)", 
                        key=f"hybrid_{query.id}", 
                        disabled=not system_ready, 
                        type="primary",
                        on_click=set_hybrid_demo if system_ready else None
                    )
                    if button_clicked and system_ready:
                        st.success(f"✅ Query loaded: '{query.question[:60]}...'")
                        st.info("👉 **Now click the 'Query Interface' tab above to see and execute your query!**")
                    elif button_clicked and not system_ready:
                        st.warning("⚠️ Please upload and process documents first in the 'Document Upload' tab.")
                
                with col2:
                    button_clicked = st.button(
                        f"🚀 Try Graph RAG", 
                        key=f"graph_{query.id}", 
                        disabled=not system_ready,
                        on_click=set_graph_demo if system_ready else None
                    )
                    if button_clicked and system_ready:
                        st.success(f"✅ Query loaded: '{query.question[:60]}...'")
                        st.info("👉 **Now click the 'Query Interface' tab above to see and execute your query!**")
                        st.warning("⚠️ Note: Graph RAG requires well-extracted entities. Try Hybrid RAG if this doesn't work.")
                    elif button_clicked and not system_ready:
                        st.warning("⚠️ Please upload and process documents first in the 'Document Upload' tab.")
                
                with col3:
                    button_clicked = st.button(
                        f"⚖️ Compare Both", 
                        key=f"compare_{query.id}", 
                        disabled=not system_ready,
                        on_click=set_compare_demo if system_ready else None
                    )
                    if button_clicked and system_ready:
                        st.success(f"✅ Query loaded: '{query.question[:60]}...'")
                        st.info("👉 **Now click the 'Query Interface' tab above to see and execute your query!**")
                    elif button_clicked and not system_ready:
                        st.warning("⚠️ Please upload and process documents first in the 'Document Upload' tab.")
                    if st.session_state.get('demo_method') == "Graph RAG":
                        st.warning("⚠️ Note: Graph RAG requires well-extracted entities. Try Hybrid RAG if this doesn't work.")
                
                # Show system status
                if not system_ready:
                    if not hasattr(st.session_state, 'initialized') or not st.session_state.initialized:
                        st.error("❌ System not initialized. Use the sidebar to initialize the system first.")
                    elif not hasattr(st.session_state, 'processed_documents') or not st.session_state.processed_documents:
                        st.warning("⚠️ No documents processed yet. Upload documents in the 'Document Upload' tab first.")
    
    @requires_streamlit
    def display_performance_dashboard(self):
        """Display performance comparison dashboard."""
        st.subheader("📊 Performance Dashboard")
        
        if not self.performance_history:
            st.info("No performance data available yet. Run some queries to see comparisons!")
            return
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.performance_history)
        
        # Overall statistics
        st.markdown("### Overall Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_queries = len(df)
            st.metric("Total Queries", total_queries)
        
        with col2:
            avg_response_time = df['response_time'].mean()
            st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
        
        with col3:
            avg_confidence = df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        with col4:
            success_rate = df['success'].mean()
            st.metric("Success Rate", f"{success_rate:.1%}")
        
        # Method comparison
        st.markdown("### Method Comparison")
        
        if len(df['method'].unique()) > 1:
            if PLOTLY_AVAILABLE:
                # Response time comparison
                fig_time = px.box(df, x='method', y='response_time', 
                                title='Response Time by Method',
                                labels={'response_time': 'Response Time (seconds)', 'method': 'RAG Method'})
                st.plotly_chart(fig_time, width='stretch')
                
                # Confidence comparison
                fig_conf = px.box(df, x='method', y='confidence',
                                title='Confidence Score by Method',
                                labels={'confidence': 'Confidence Score', 'method': 'RAG Method'})
                st.plotly_chart(fig_conf, width='stretch')
                
                # Source count comparison
                fig_sources = px.box(df, x='method', y='source_count',
                                   title='Source Count by Method',
                                   labels={'source_count': 'Number of Sources', 'method': 'RAG Method'})
                st.plotly_chart(fig_sources, width='stretch')
            else:
                st.info("📊 Install plotly for enhanced visualizations: `pip install plotly`")
                
                # Fallback to simple metrics
                for method in df['method'].unique():
                    method_data = df[df['method'] == method]
                    st.subheader(f"{method} Performance")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Response Time", f"{method_data['response_time'].mean():.2f}s")
                    with col2:
                        st.metric("Avg Confidence", f"{method_data['confidence'].mean():.1%}")
                    with col3:
                        st.metric("Avg Sources", f"{method_data['source_count'].mean():.1f}")
        
        # Query-specific analysis
        st.markdown("### Query-Specific Analysis")
        
        unique_queries = df['query'].unique()
        if len(unique_queries) > 0:
            selected_query = st.selectbox("Select Query for Detailed Analysis", unique_queries)
            
            query_df = df[df['query'] == selected_query]
            
            if len(query_df) > 1:
                # Create comparison table
                comparison_data = []
                for method in query_df['method'].unique():
                    method_data = query_df[query_df['method'] == method]
                    comparison_data.append({
                        'Method': method,
                        'Avg Response Time': f"{method_data['response_time'].mean():.2f}s",
                        'Avg Confidence': f"{method_data['confidence'].mean():.1%}",
                        'Avg Sources': f"{method_data['source_count'].mean():.1f}",
                        'Success Rate': f"{method_data['success'].mean():.1%}",
                        'Runs': len(method_data)
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, width='stretch')
        
        # Historical trends
        if len(df) > 5:
            st.markdown("### Historical Trends")
            
            # Add sequence number for time series
            df_sorted = df.sort_values('timestamp').reset_index(drop=True)
            df_sorted['sequence'] = range(len(df_sorted))
            
            if PLOTLY_AVAILABLE:
                # Response time trend
                fig_trend = px.line(df_sorted, x='sequence', y='response_time', color='method',
                                  title='Response Time Trend Over Queries',
                                  labels={'sequence': 'Query Sequence', 'response_time': 'Response Time (seconds)'})
                st.plotly_chart(fig_trend, width='stretch')
            else:
                # Fallback to simple line chart using Streamlit
                st.line_chart(df_sorted.set_index('sequence')[['response_time']])
    
    def export_performance_data(self) -> str:
        """Export performance data as JSON string."""
        return json.dumps(self.performance_history, default=str, indent=2)
    
    def import_performance_data(self, json_data: str):
        """Import performance data from JSON string."""
        try:
            imported_data = json.loads(json_data)
            self.performance_history.extend(imported_data)
            return True
        except json.JSONDecodeError:
            return False
    
    def clear_performance_data(self):
        """Clear all performance history."""
        self.performance_history.clear()
    
    def get_sample_documents_info(self) -> List[Dict[str, Any]]:
        """Get information about available sample documents."""
        sample_docs_path = Path("data/samples")
        
        if not sample_docs_path.exists():
            return []
        
        sample_docs = []
        for file_path in sample_docs_path.glob("*"):
            if file_path.is_file() and file_path.suffix in ['.md', '.txt']:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    word_count = len(content.split())
                    char_count = len(content)
                    
                    sample_docs.append({
                        'filename': file_path.name,
                        'path': str(file_path),
                        'size_kb': file_path.stat().st_size / 1024,
                        'word_count': word_count,
                        'char_count': char_count,
                        'type': file_path.suffix[1:].upper(),
                        'description': self._get_document_description(file_path.name)
                    })
                except Exception as e:
                    continue
        
        return sample_docs
    
    def _get_document_description(self, filename: str) -> str:
        """Get description for sample documents."""
        descriptions = {
            'sample_document.md': 'Introduction to Knowledge Graphs and RAG systems',
            'sample_text.txt': 'Natural Language Processing and Machine Learning overview',
            'ai_research_paper.md': 'Comprehensive paper on LLMs and Knowledge Graphs with researchers and organizations',
            'tech_company_profiles.txt': 'Detailed profiles of AI and database technology companies',
            'university_research_network.md': 'Global network of universities and their AI research collaborations',
            'workshop_example_queries.md': 'Example queries and expected behaviors for workshop demonstration'
        }
        return descriptions.get(filename, 'Sample document for workshop demonstration')
    
    @requires_streamlit
    def display_sample_documents_interface(self):
        """Display sample documents interface."""
        st.subheader("📚 Sample Documents")
        
        sample_docs = self.get_sample_documents_info()
        
        if not sample_docs:
            st.warning("No sample documents found in data/samples directory")
            return
        
        st.info(f"Found {len(sample_docs)} sample documents ready for upload and processing")
        
        # Create DataFrame for display
        df = pd.DataFrame(sample_docs)
        df['Size (KB)'] = df['size_kb'].round(2)
        df['Words'] = df['word_count']
        df['Type'] = df['type']
        df['Description'] = df['description']
        
        # Display table
        display_df = df[['filename', 'Type', 'Size (KB)', 'Words', 'Description']]
        st.dataframe(display_df, width='stretch')
        
        # Quick upload guidance
        st.markdown("### 🚀 Quick Start Guide")
        
        # Check if documents are already processed
        processed_count = 0
        if hasattr(st.session_state, 'processed_documents') and st.session_state.processed_documents:
            processed_count = len(st.session_state.processed_documents)
        
        if processed_count > 0:
            st.success(f"✅ **{processed_count} documents** already processed and ready for querying!")
            st.info("💡 You can now use the sample queries in the 'Example Queries' tab above.")
        else:
            st.info("📋 **To get started with the workshop:**")
            st.write("1. 🔧 **Initialize System**: Use the sidebar to initialize the system")
            st.write("2. 📄 **Upload Documents**: Go to the 'Document Upload' tab")
            st.write("3. 📤 **Process Files**: Upload any of the sample documents listed above")
            st.write("4. 🔍 **Try Queries**: Return here to use the 'Example Queries' tab")
            
            st.markdown("### 📁 Recommended Sample Documents")
            st.write("**For best workshop experience, upload these documents:**")
            st.write("• `ai_research_paper.md` - Rich content with entities and relationships")
            st.write("• `tech_company_profiles.txt` - Company and technology information")
            st.write("• `university_research_network.md` - Academic collaboration network")
        
        # Action buttons
        st.markdown("### Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Go to Document Upload"):
                st.info("💡 Switch to the 'Document Upload' tab to upload sample documents")
        
        with col2:
            if st.button("🔍 View Sample Queries"):
                st.info("💡 Switch to the 'Example Queries' tab above to see available queries")
        
        with col3:
            if st.button("📊 View Performance Dashboard"):
                st.info("💡 Performance data will appear here after running queries")

# Global instance for use in Streamlit app
workshop_demo = WorkshopDemo()
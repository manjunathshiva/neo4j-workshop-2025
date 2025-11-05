# Knowledge Graph RAG System

A workshop demonstration system that combines Neo4j graph database with Retrieval-Augmented Generation (RAG) capabilities, designed to run entirely in GitHub Codespaces.

## 🚀 Quick Start

> **Important**: This project uses a Python virtual environment to manage dependencies. Make sure to activate it before running any commands.

### 1. Setup Environment

**Option A: Automated Setup (Recommended)**
```bash
./setup.sh
```

**Option B: Manual Setup**
1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Copy environment template:**
   ```bash
   cp .env.template .env
   ```

4. **Configure your API keys in `.env`:**
   - Neo4j cloud credentials
   - Qdrant cloud credentials  
   - Groq API key (free tier available)

### 2. Validate Setup

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Validate configuration and connections
python startup.py
```

This will validate your configuration and test all database connections.

### 3. Launch Application

**Option A: Streamlit Web Interface (Recommended for workshops)**
```bash
# Make sure virtual environment is activated
source venv/bin/activate
streamlit run app/main.py --server.port 8501
```

**Option B: FastAPI Backend**
```bash
# Make sure virtual environment is activated
source venv/bin/activate
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

## 🏗️ System Architecture

- **Frontend**: Streamlit web application (port 8501)
- **Backend**: FastAPI REST API (port 8000)
- **Graph Database**: Neo4j Cloud (entities, relationships)
- **Vector Database**: Qdrant Cloud (document embeddings)
- **LLM Integration**: Groq/OpenAI/Anthropic via LangChain
- **Embeddings**: Local sentence-transformers (no API costs)

## 📋 Features

### Document Processing
- Upload PDF, DOCX, TXT, MD files
- Automatic text extraction and chunking
- Entity and relationship extraction
- Knowledge graph construction

### Dual RAG Approaches
- **Graph RAG**: Query Neo4j knowledge graph directly
- **Hybrid RAG**: Combine graph traversal with vector similarity search

### Workshop Features
- Interactive web interface
- Real-time processing status
- Graph visualization
- Source attribution
- Performance comparisons

## 🔧 Configuration

### Required Environment Variables

```bash
# Database Connections
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_PASSWORD=your-password
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key

# LLM API (choose one)
GROQ_API_KEY=your-groq-key          # Recommended: free tier
OPENAI_API_KEY=your-openai-key      # Alternative: premium
```

### Model Configuration

```bash
PRIMARY_LLM=groq                    # groq, openai, anthropic
TEXT_TO_CYPHER_MODEL=kimi-k2-instruct-0905
EMBEDDING_MODEL=local               # Uses sentence-transformers
```

## 🎯 Workshop Usage

1. **Upload Documents**: Drag and drop files into the web interface
2. **Process Documents**: Watch real-time entity extraction and graph building
3. **Query Knowledge**: Ask natural language questions
4. **Compare Methods**: Try both Graph RAG and Hybrid RAG
5. **Explore Results**: View graph connections and source attribution

## 📁 Project Structure

```
knowledge-graph-rag/
├── .devcontainer/          # Codespace configuration
├── app/                    # Main application code
│   ├── config.py          # Configuration management
│   ├── connections.py     # Database connections
│   ├── main.py           # Streamlit application
│   └── api/              # FastAPI backend
├── data/                  # Sample documents
├── venv/                  # Virtual environment (created by setup)
├── requirements.txt       # Python dependencies
├── .env.template         # Environment template
├── setup.sh              # Automated setup script
├── activate.sh           # Virtual environment activation
├── startup.py            # Validation script
└── README.md             # This file
```

## 🔍 Troubleshooting

### Virtual Environment Issues
- **"Module not found" errors**: Make sure virtual environment is activated
  ```bash
  source venv/bin/activate  # or ./activate.sh
  ```
- **Virtual environment not found**: Run the setup script
  ```bash
  ./setup.sh
  ```
- **Permission denied**: Make scripts executable
  ```bash
  chmod +x setup.sh activate.sh
  ```

### Connection Issues
- Ensure virtual environment is activated: `source venv/bin/activate`
- Verify your `.env` file has correct credentials
- Check that Neo4j and Qdrant cloud instances are running
- Run `python startup.py` to validate connections

### Port Access
- Codespace automatically forwards ports 8501 and 8000
- Access via the "Ports" tab in Codespace
- Make ports public if sharing with others

### Performance
- Codespace 4-core/8GB recommended for smooth operation
- Local embeddings optimize for CPU processing
- Document processing is sequential for demonstration

## 📚 Learning Objectives

This workshop demonstrates:
- Knowledge graph construction from unstructured text
- Entity and relationship extraction techniques
- Graph-based vs vector-based retrieval
- Hybrid RAG system architecture
- Production scalability considerations

## 🚀 Production Considerations

- Connection pooling for database clients
- Async processing for document pipelines
- Monitoring and observability
- Horizontal scaling strategies
- Cost optimization techniques

## 📄 License

MIT License - see LICENSE file for details.
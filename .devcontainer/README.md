# GitHub Codespaces Configuration

This directory contains the configuration for running the Knowledge Graph RAG System in GitHub Codespaces.

## Overview

The system is configured to run entirely in the cloud using:
- **GitHub Codespaces**: Cloud development environment
- **Neo4j Aura**: Cloud graph database
- **Qdrant Cloud**: Cloud vector database
- **Streamlit**: Web interface (port 8501)
- **FastAPI**: REST API (port 8000)

## Automatic Setup

When you create a Codespace, the following happens automatically:

1. **Container Creation**: Python 3.11 environment with Git and GitHub CLI
2. **Post-Create Script** (`post-create.sh`):
   - Upgrades pip
   - Installs Python dependencies from `requirements.txt`
   - Creates `.env` file from template
   - Makes scripts executable
   - Creates data directories

3. **Post-Start Script**:
   - Runs quick validation check on each Codespace start
   - Validates configuration and database connectivity

4. **Port Forwarding**:
   - Port 8501: Streamlit web interface
   - Port 8000: FastAPI backend

## Manual Setup Steps

### 1. Configure Environment Variables

Edit the `.env` file with your cloud database credentials:

```bash
# Neo4j Cloud (Required)
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

# Qdrant Cloud (Required)
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key

# LLM Configuration (Required - Groq is free)
GROQ_API_KEY=your-groq-key

# Optional Premium LLMs
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

### 2. Validate Setup

Run the validation script to check your configuration:

```bash
python validate_startup.py
```

This will:
- Validate all environment variables
- Test Neo4j connection
- Test Qdrant connection
- Display system status and recommendations

### 3. Start the Application

#### Option A: Use the convenience script (recommended)
```bash
bash .devcontainer/start-services.sh
```

This script:
- Runs validation
- Starts Streamlit if validation passes
- Provides helpful error messages if validation fails

#### Option B: Start services manually

**Streamlit (Web Interface):**
```bash
streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
```

**FastAPI (REST API):**
```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access the Application

1. Go to the **Ports** tab in Codespaces
2. Find port 8501 (Streamlit) or 8000 (FastAPI)
3. Click the globe icon to open in browser
4. Or use the forwarded URL provided by Codespaces

## Health Check Endpoints

When the FastAPI backend is running, you can check system health:

- **Basic Health**: `GET /health`
  - Returns 200 if API is running
  
- **Detailed Health**: `GET /health/detailed`
  - Returns comprehensive component status
  - Shows Neo4j, Qdrant, and configuration status
  
- **Readiness Check**: `GET /health/ready`
  - Returns 200 if system is ready to accept requests
  - Returns 503 if not ready
  
- **Liveness Check**: `GET /health/live`
  - Simple check that API process is alive
  
- **Startup Status**: `GET /health/startup`
  - Returns complete startup validation results
  
- **Trigger Validation**: `POST /health/validate`
  - Re-runs validation checks
  - Useful after configuration changes

- **Configuration Info**: `GET /config/info`
  - Returns non-sensitive configuration information

## Scripts

### `post-create.sh`
Runs once when Codespace is created. Sets up the environment.

### `start-services.sh`
Validates system and starts Streamlit application with error handling.

## Troubleshooting

### Configuration Errors

If you see configuration errors:
1. Check your `.env` file exists
2. Verify all required variables are set
3. Ensure no typos in variable names
4. Run `python validate_startup.py` for detailed diagnostics

### Database Connection Errors

**Neo4j Connection Failed:**
- Verify `NEO4J_URI` is correct (should start with `neo4j+s://`)
- Check `NEO4J_USERNAME` and `NEO4J_PASSWORD`
- Ensure your Neo4j Aura instance is running
- Check network connectivity

**Qdrant Connection Failed:**
- Verify `QDRANT_URL` is correct (should start with `https://`)
- Check `QDRANT_API_KEY` is valid
- Ensure your Qdrant cluster is running
- Check network connectivity

### Port Forwarding Issues

If you can't access the application:
1. Check the Ports tab in Codespaces
2. Ensure ports 8501 and 8000 are forwarded
3. Try making the port public if private forwarding doesn't work
4. Check that the application is actually running

### Graceful Degradation

The system supports graceful degradation:
- If only Neo4j is available: Graph RAG works, Hybrid RAG disabled
- If only Qdrant is available: Vector search works, Graph RAG disabled
- If both fail: System won't start

## Workshop Usage

For workshop participants:

1. **Fork the repository** to your GitHub account
2. **Create a Codespace** from your fork
3. **Wait for automatic setup** to complete
4. **Configure `.env`** with provided credentials
5. **Run validation**: `python validate_startup.py`
6. **Start the app**: `bash .devcontainer/start-services.sh`
7. **Access via Ports tab** in Codespaces

## Development

### Rebuilding the Container

If you modify `.devcontainer/devcontainer.json`:
1. Open Command Palette (Cmd/Ctrl + Shift + P)
2. Select "Codespaces: Rebuild Container"
3. Wait for rebuild to complete

### Adding Dependencies

1. Add package to `requirements.txt`
2. Run: `pip install -r requirements.txt`
3. Or rebuild the container

### Debugging

- **Python logs**: Check terminal output
- **Streamlit logs**: Visible in Streamlit terminal
- **FastAPI logs**: Visible in uvicorn terminal
- **Validation details**: Run `python validate_startup.py --json`

## Resources

- [GitHub Codespaces Documentation](https://docs.github.com/en/codespaces)
- [Neo4j Aura](https://neo4j.com/cloud/aura/)
- [Qdrant Cloud](https://qdrant.tech/documentation/cloud/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

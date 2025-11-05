#!/bin/bash

# Shell script to run Streamlit with the virtual environment
# Usage: ./run_streamlit.sh

echo "🚀 Starting Knowledge Graph RAG System with venv..."

# Check if virtual environment exists
if [ ! -f "venv/bin/python" ]; then
    echo "❌ Virtual environment not found at venv/bin/python"
    echo "💡 Please create a virtual environment first:"
    echo "   python -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Check if Streamlit is installed in venv
if ! venv/bin/python -c "import streamlit" 2>/dev/null; then
    echo "❌ Streamlit not found in virtual environment"
    echo "💡 Please install requirements:"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

echo "✅ Virtual environment found"
echo "🐍 Using: $(pwd)/venv/bin/python"

# Set up environment
export PYTHONPATH="$(pwd):$(pwd)/app:$PYTHONPATH"

echo "📁 Project root: $(pwd)"
echo "🐍 Python path: $PYTHONPATH"
echo "🌐 Starting Streamlit on http://localhost:8501"
echo "=" | tr ' ' '=' | head -c 60; echo

# Use the main Streamlit interface
echo "🚀 Using full Streamlit interface..."
STREAMLIT_APP="app/main.py"

# Run Streamlit with venv Python
exec ./venv/bin/python -m streamlit run $STREAMLIT_APP \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
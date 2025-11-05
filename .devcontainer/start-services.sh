#!/bin/bash
# Start services script for Knowledge Graph RAG System
# This script validates the system and starts the Streamlit application

set -e

echo "🚀 Starting Knowledge Graph RAG System..."
echo "============================================================"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🔌 Activating virtual environment..."
    source venv/bin/activate
fi

# Set PYTHONPATH to include app directory
export PYTHONPATH="${PWD}:${PWD}/app:${PYTHONPATH}"
echo "📍 PYTHONPATH set to include app directory"

# Run startup validation
echo "🔍 Validating system configuration and connections..."
python validate_startup.py

# Check validation exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Validation successful! Starting Streamlit application..."
    echo "============================================================"
    echo ""
    
    # Start Streamlit
    streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
    
elif [ $? -eq 2 ]; then
    echo ""
    echo "⚠️  System validation completed with warnings."
    echo "Some functionality may be limited."
    echo ""
    read -p "Do you want to start the application anyway? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Starting Streamlit application..."
        streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
    else
        echo "Startup cancelled. Please fix the warnings and try again."
        exit 1
    fi
    
else
    echo ""
    echo "❌ System validation failed!"
    echo "Please fix the errors above before starting the application."
    echo ""
    echo "Common issues:"
    echo "• Missing or incorrect .env configuration"
    echo "• Cloud database credentials not set"
    echo "• Network connectivity issues"
    echo ""
    echo "Run 'python validate_startup.py' for detailed diagnostics."
    exit 1
fi

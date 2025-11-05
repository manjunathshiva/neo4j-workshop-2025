#!/bin/bash
# Setup script for Knowledge Graph RAG System

set -e  # Exit on any error

echo "🚀 Setting up Knowledge Graph RAG System..."
echo "=" * 60

# Check Python version
echo "🐍 Checking Python version..."
python3 --version

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup completed successfully!"
echo "=" * 60
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Copy and configure environment file:"
echo "   cp .env.template .env"
echo "   # Edit .env with your API keys"
echo ""
echo "3. Validate your setup:"
echo "   python startup.py"
echo ""
echo "4. Launch the application:"
echo "   streamlit run app/main.py --server.port 8501"
echo ""
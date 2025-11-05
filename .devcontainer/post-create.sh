#!/bin/bash
# Post-create script for GitHub Codespaces
# This script runs once when the Codespace is created

set -e

echo "🚀 Setting up Knowledge Graph RAG Workshop Environment..."
echo "============================================================"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip in venv
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies in venv
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Copy environment template if .env doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.template .env
    echo "⚠️  Please configure your .env file with cloud database credentials"
else
    echo "✅ .env file already exists"
fi

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x startup.py validate_startup.py run_streamlit.sh activate.sh setup.sh

# Create data directories if they don't exist
echo "📁 Creating data directories..."
mkdir -p data/samples
mkdir -p data/uploads

echo ""
echo "✅ Setup complete!"
echo "============================================================"
echo "📝 Virtual environment created at: ./venv"
echo "🔌 VS Code will automatically use the venv Python interpreter"
echo ""
echo "Next steps:"
echo "1. Configure your .env file with cloud database credentials"
echo "2. Run: python validate_startup.py"
echo "3. Start the app: bash quick-start.sh"
echo ""
echo "💡 Tip: The venv is automatically activated in new terminals"
echo ""
echo "For workshop participants:"
echo "• The Streamlit app will be available on port 8501"
echo "• Access it via the Ports tab in Codespaces"
echo "============================================================"

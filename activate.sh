#!/bin/bash
# Convenience script to activate virtual environment

if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

echo "🔌 Activating virtual environment..."
source venv/bin/activate

echo "✅ Virtual environment activated!"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo ""
echo "You can now run:"
echo "  python startup.py          # Validate setup"
echo "  streamlit run app/main.py  # Launch web interface"
echo ""
echo "To deactivate later, run: deactivate"
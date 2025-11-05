#!/usr/bin/env python3
"""
Launch script for the Streamlit Knowledge Graph RAG System.
Validates system and starts the web interface.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit application."""
    print("🚀 Starting Knowledge Graph RAG System...")
    
    # Ensure we're in the right directory
    project_root = Path(__file__).parent.resolve()
    os.chdir(project_root)
    
    # Set up Python path environment variable for subprocess
    env = os.environ.copy()
    app_dir = project_root / "app"
    
    # Add both project root and app directory to PYTHONPATH
    current_pythonpath = env.get('PYTHONPATH', '')
    new_paths = [str(project_root), str(app_dir)]
    
    if current_pythonpath:
        new_pythonpath = os.pathsep.join(new_paths + [current_pythonpath])
    else:
        new_pythonpath = os.pathsep.join(new_paths)
    
    env['PYTHONPATH'] = new_pythonpath
    
    print("📋 Validating system configuration...")
    print(f"📁 Project root: {project_root}")
    print(f"📁 App directory: {app_dir}")
    print(f"🐍 Python path: {new_pythonpath}")
    
    # Launch Streamlit
    print("🌐 Launching Streamlit web interface...")
    print("📍 The application will be available at: http://localhost:8501")
    print("🔗 In Codespace, use the forwarded port URL")
    print("\n" + "="*60)
    
    try:
        # Use venv Python interpreter
        venv_python = project_root / "venv" / "bin" / "python"
        if not venv_python.exists():
            print(f"⚠️  Virtual environment not found at {venv_python}")
            print("Using system Python instead...")
            python_executable = sys.executable
        else:
            python_executable = str(venv_python)
            print(f"🐍 Using virtual environment: {python_executable}")
        
        # Launch Streamlit with the main app and proper environment
        cmd = [
            python_executable, "-m", "streamlit", "run", 
            "app/main.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        print(f"🚀 Running command: {' '.join(cmd)}")
        subprocess.run(cmd, env=env, check=True)
        
    except KeyboardInterrupt:
        print("\n⏹️  Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to start Streamlit: {e}")
        print("💡 Try running directly: streamlit run app/main.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
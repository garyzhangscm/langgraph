#!/usr/bin/env python3
"""Launch script for the Streamlit app with proper environment setup."""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit application."""
    
    # Set the working directory to the project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Set environment variables for the app
    os.environ.setdefault("PYTHONPATH", str(project_root / "src"))
    
    # Check if .env file exists
    env_file = project_root / ".env"
    if not env_file.exists():
        print("⚠️  Warning: .env file not found!")
        print("   Please copy .env.example to .env and configure your API keys.")
        print("   The app may not work properly without proper configuration.")
        print()
    
    # Launch Streamlit
    print("🚀 Starting LangGraph Knowledge Base Web Interface...")
    print("   Opening in your default browser...")
    print("   Press Ctrl+C to stop the server")
    print()
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.headless", "false",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
            "--theme.primaryColor", "#1f77b4",
            "--theme.backgroundColor", "#ffffff",
            "--theme.secondaryBackgroundColor", "#f0f2f6"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
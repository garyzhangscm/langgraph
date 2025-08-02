#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick setup script for LangGraph SOP Knowledge Base project.
Run this after cloning the repository to set up everything.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"Running: {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"SUCCESS: {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {description} - {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("Setting up LangGraph SOP Knowledge Base project...")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("ERROR: pyproject.toml not found. Are you in the project root directory?")
        sys.exit(1)
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("ERROR: Python 3.9 or higher is required")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    
    # Install dependencies
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -e \".[dev,test,docs]\"", "Installing all dependencies"),
        ("python -c \"import agent; print('Package import successful')\"", "Testing installation"),
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            print(f"\nSetup failed at: {desc}")
            print("Please check the error messages above and try again.")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("\nProject structure:")
    print("   static/data/sop/ - Place your SOP files here")
    print("   static/data/vector_store/ - Knowledge base storage")
    print("   .env - Configure your API keys and paths")
    print("\nNext steps:")
    print("   1. Copy .env.example to .env and add your API keys")
    print("   2. Place SOP files in static/data/sop/")
    print("   3. Run: langgraph dev")
    print("\nAvailable commands:")
    print("   make help          - Show all available commands")
    print("   make test          - Run tests")
    print("   make lint          - Check code quality")
    print("   langgraph dev      - Start the development server")


if __name__ == "__main__":
    main()
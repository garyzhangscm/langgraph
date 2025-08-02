@echo off
REM Quick setup script for Windows users
echo 🚀 Setting up LangGraph SOP Knowledge Base project...
echo ============================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "pyproject.toml" (
    echo ❌ Error: pyproject.toml not found
    echo Are you in the project root directory?
    pause
    exit /b 1
)

echo ✅ Python found, proceeding with setup...

REM Upgrade pip
echo 🔄 Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ❌ Failed to upgrade pip
    pause
    exit /b 1
)

REM Install dependencies
echo 🔄 Installing all dependencies...
pip install -e ".[dev,test,docs]"
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

REM Test installation
echo 🔄 Testing installation...
python -c "import agent; print('✅ Package import successful')"
if errorlevel 1 (
    echo ❌ Installation test failed
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 🎉 Setup completed successfully!
echo.
echo 📁 Project structure:
echo    📂 static/data/sop/ - Place your SOP files here
echo    📂 static/data/vector_store/ - Knowledge base storage
echo    📄 .env - Configure your API keys and paths
echo.
echo 🚀 Next steps:
echo    1. Copy .env.example to .env and add your API keys
echo    2. Place SOP files in static/data/sop/
echo    3. Run: langgraph dev
echo.
echo 📚 Available commands:
echo    make help          - Show all available commands
echo    make test          - Run tests
echo    make lint          - Check code quality
echo    langgraph dev      - Start the development server
echo.
pause
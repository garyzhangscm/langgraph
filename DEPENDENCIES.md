# Dependency Management Guide

This document outlines best practices for managing dependencies in the LangGraph SOP Knowledge Base project.

## **Current Dependency Management**

### **Primary Tool: pyproject.toml**
We use modern Python packaging with `pyproject.toml` (PEP 621) for dependency management.

### **Dependency Categories**

#### **Core Dependencies** (Production)
```toml
dependencies = [
    "langgraph>=0.2.6",           # Core LangGraph framework
    "python-dotenv>=1.0.1",       # Environment variable management
    "langchain_anthropic",        # Claude LLM integration
    "langchain-community>=0.0.29", # Community LangChain components
    "langchain-huggingface>=0.0.3", # HuggingFace embeddings (updated package)
    "sentence-transformers>=2.2.0", # Text embeddings
    "faiss-cpu>=1.7.0",          # Vector database
    "PyMuPDF>=1.23.0",           # PDF processing
    "python-docx>=0.8.11"        # Word document processing
]
```

#### **Development Dependencies**
```toml
[project.optional-dependencies]
dev = [
    "mypy>=1.11.1",              # Static type checking
    "ruff>=0.6.1",               # Code formatting and linting
    "pytest>=8.3.5",             # Testing framework
    "pytest-asyncio>=0.23.0",    # Async testing support
    "pytest-cov>=4.1.0"          # Test coverage
]
```

## **Installation Methods**

### **1. Production Installation**
```bash
# Install core dependencies only
pip install -e .

# Or with specific extras
pip install -e ".[test]"
```

### **2. Development Installation**
```bash
# Install with development dependencies
pip install -e ".[dev]"

# Or install everything
pip install -e ".[dev,test,docs]"
```

### **3. Legacy Requirements Files**
```bash
# Development dependencies (alternative method)
pip install -r requirements-dev.txt
```

## **Dependency Update Strategy**

### **1. Regular Updates**
```bash
# Check for outdated packages
pip list --outdated

# Update specific package
pip install --upgrade package_name
```

### **2. Security Updates**
```bash
# Check for security vulnerabilities
pip-audit

# Or use safety
safety check
```

### **3. Version Pinning Strategy**
- **Minimum versions** (`>=`) for core functionality
- **Compatible release** (`~=`) for stable APIs
- **Exact versions** (`==`) only for critical security fixes

## **Best Practices**

### **1. Dependency Categories**
- **Core**: Essential for application functionality
- **Dev**: Development, testing, and linting tools
- **Test**: Testing-specific dependencies
- **Docs**: Documentation generation tools

### **2. Version Management**
```toml
# Good: Allows patch updates
"package>=1.2.0,<2.0.0"

# Better: Compatible release
"package~=1.2.0"

# Use with caution: Exact pinning
"package==1.2.3"
```

### **3. Optional Dependencies**
```python
# Handle optional imports gracefully
try:
    import optional_package
    HAS_OPTIONAL = True
except ImportError:
    HAS_OPTIONAL = False
    print("Warning: optional_package not available")
```

## **Troubleshooting Common Issues**

### **1. PDF Processing**
```bash
# If PyMuPDF fails, try alternative
pip install pypdf

# For Windows users
pip install --upgrade PyMuPDF
```

### **2. Vector Database**
```bash
# CPU version (default)
pip install faiss-cpu

# GPU version (if CUDA available)
pip install faiss-gpu
```

### **3. LangChain Updates**
```bash
# LangChain ecosystem updates frequently
pip install --upgrade langchain-community langchain-core langchain-huggingface

# Note: HuggingFaceEmbeddings moved from langchain-community to langchain-huggingface
# in LangChain 0.2.2+. The old import is deprecated and will be removed in v1.0
```

## **Development Workflow**

### **1. New Dependency Addition**
1. Add to appropriate section in `pyproject.toml`
2. Test installation in clean environment
3. Update documentation
4. Add import handling if optional

### **2. Dependency Removal**
1. Remove from `pyproject.toml`
2. Remove imports from code
3. Test thoroughly
4. Update documentation

### **3. Version Updates**
1. Test in development environment
2. Check for breaking changes
3. Update version constraints
4. Run full test suite

## **Environment Management**

### **1. Virtual Environments**
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install project
pip install -e ".[dev]"
```

### **2. Container Deployment**
```dockerfile
# Dockerfile example
FROM python:3.11-slim

COPY pyproject.toml .
RUN pip install -e .

# Runtime dependencies only
```

## **Monitoring and Maintenance**

### **1. Regular Health Checks**
- Monthly dependency updates
- Security vulnerability scans
- Compatibility testing with new versions

### **2. Documentation Updates**
- Keep DEPENDENCIES.md current
- Update installation instructions
- Document known issues and workarounds

### **3. Performance Monitoring**
- Track dependency impact on startup time
- Monitor memory usage with different versions
- Benchmark critical path performance

## **Tools and Automation**

### **1. Dependency Scanning**
```bash
# Install security tools
pip install pip-audit safety

# Scan for vulnerabilities
pip-audit
safety check
```

### **2. Automated Updates**
Consider tools like:
- **Dependabot** (GitHub)
- **Renovate Bot**
- **PyUp**

### **3. Lock Files**
For reproducible builds, consider:
```bash
# Generate lock file
pip freeze > requirements.lock

# Or use pip-tools
pip-compile pyproject.toml
```

## **Migration Guide**

### **From requirements.txt to pyproject.toml**
1. Move dependencies to `[project.dependencies]`
2. Categorize dev dependencies in `[project.optional-dependencies]`
3. Test installation methods
4. Update CI/CD pipelines
5. Update documentation

This approach provides robust, maintainable dependency management that scales with your project's growth.
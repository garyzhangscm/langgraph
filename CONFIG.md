# Configuration Guide

This document describes the configuration options for the LangGraph SOP Knowledge Base Agent.

## Environment Variables

Configuration is managed through environment variables in the `.env` file. Copy `.env.example` to `.env` and customize as needed.

### API Keys

```env
# Required: Anthropic API key for Claude LLM
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: LangSmith for tracing and monitoring
LANGSMITH_PROJECT=new-agent
LANGSMITH_API_KEY=your_langsmith_api_key_here
```

### SOP Knowledge Base Configuration

```env
# SOP source folder path (relative to project root or absolute path)
SOP_SOURCE_FOLDER=static/data/sop

# Knowledge base storage path (where vector database is stored)
SOP_KNOWLEDGE_BASE_PATH=sop_knowledge_base
```

## Configuration Options

### `SOP_SOURCE_FOLDER`
- **Purpose**: Specifies where SOP files are located
- **Default**: `static/data/sop`
- **Examples**:
  - Relative path: `static/data/sop`
  - Absolute path: `/home/user/documents/sops`
  - Windows path: `C:\Documents\SOPs`

### `SOP_KNOWLEDGE_BASE_PATH`
- **Purpose**: Where the vector database and metadata are stored
- **Default**: `static/data/vector_store/sop_knowledge_base`
- **Examples**:
  - Default: `static/data/vector_store/sop_knowledge_base`
  - Shared storage: `/shared/knowledge_base`
  - Custom location: `/data/vector_db/sops`

## Supported File Types

The knowledge base automatically processes these file types:

- **Text files**: `.txt`
- **Markdown files**: `.md`
- **reStructuredText files**: `.rst`
- **Microsoft Word files**: `.doc`, `.docx`
- **PDF files**: `.pdf`

## Directory Structure

### Default Setup
```
langgraph/
├── static/
│   └── data/
│       ├── sop/              # SOP source files (SOP_SOURCE_FOLDER)
│       │   ├── onboarding.md
│       │   ├── security.pdf
│       │   └── procedures.txt
│       └── vector_store/
│           └── sop_knowledge_base/  # Vector database (SOP_KNOWLEDGE_BASE_PATH)
│               ├── metadata.json
│               └── vector_store/
└── .env                      # Configuration file
```

### Custom Configuration
```env
# Custom paths in .env
SOP_SOURCE_FOLDER=/company/procedures
SOP_KNOWLEDGE_BASE_PATH=static/data/vector_store/sop_knowledge_base
```

## Usage Examples

### Basic Setup
1. Copy `.env.example` to `.env`
2. Add your Anthropic API key
3. Place SOP files in `static/data/sop/`
4. Run `langgraph dev`

### Custom Folder Setup
1. Set `SOP_SOURCE_FOLDER` in `.env` to your desired path
2. Ensure the folder exists and contains SOP files
3. Run the application

### Programmatic Configuration
```python
from agent import get_sop_source_folder, get_sop_knowledge_base_path

# Get current configuration
sop_folder = get_sop_source_folder()
kb_path = get_sop_knowledge_base_path()

print(f"SOP files location: {sop_folder}")
print(f"Knowledge base location: {kb_path}")
```

## Automatic Initialization

The knowledge base is automatically built when the application starts:

1. **Checks** if the SOP source folder exists
2. **Scans** for supported file types
3. **Processes** each file to extract content
4. **Generates** metadata using AI
5. **Creates** vector embeddings for search
6. **Stores** everything in the knowledge base

## Manual Rebuild

To manually rebuild the knowledge base:

```python
from agent import build_sop_knowledge_base

# Rebuild from default location
results = build_sop_knowledge_base()

# Rebuild from custom location
results = build_sop_knowledge_base("/path/to/sop/files")

print(f"Processed: {results['processed']}")
print(f"Updated: {results['updated']}")
print(f"Errors: {results['errors']}")
```

## Troubleshooting

### Common Issues

1. **SOP folder not found**
   - Verify `SOP_SOURCE_FOLDER` path in `.env`
   - Ensure the folder exists and is accessible

2. **No files processed**
   - Check file extensions are supported
   - Verify files are not empty or corrupted

3. **PDF processing errors**
   - Ensure PyMuPDF is installed: `pip install PyMuPDF`
   - Check PDF files are not password-protected

### Debug Information

The application provides verbose logging during initialization:
```
Building SOP knowledge base from /path/to/sop/files...
Configuration: SOP_SOURCE_FOLDER=static/data/sop
SOP Knowledge base initialized successfully!
Processed: 5, Updated: 0, Skipped: 2, Errors: 0
```

## Security Considerations

- Keep your `.env` file private (it's in `.gitignore`)
- Use environment-specific configurations
- Store sensitive documents in secure locations
- Consider access controls for knowledge base storage
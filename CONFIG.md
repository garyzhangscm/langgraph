# Configuration Guide

This document describes the configuration options for the LangGraph SOP and SQL Schema Knowledge Base Agents.

## Environment Variables

Configuration is managed through environment variables in the `.env` file. Copy `.env.example` to `.env` and customize as needed.

### API Keys

```env
# LLM Provider API Keys
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Optional: LangSmith for tracing and monitoring
LANGSMITH_PROJECT=new-agent
LANGSMITH_API_KEY=your_langsmith_api_key_here
```

### LLM Configuration

```env
# LLM Provider: anthropic or openai
LLM_PROVIDER=anthropic

# Model selection based on provider
# Anthropic models: claude-3-5-sonnet-latest, claude-3-5-haiku-latest, claude-3-opus-latest
# OpenAI models: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
LLM_MODEL=claude-3-5-sonnet-latest
```

### SOP Knowledge Base Configuration

```env
# SOP source folder path (relative to project root or absolute path)
SOP_SOURCE_FOLDER=static/data/sop

# Knowledge base storage path (where vector database is stored)
SOP_KNOWLEDGE_BASE_PATH=static/data/vector_store/sop_knowledge_base
```

### SQL Schema Knowledge Base Configuration

```env
# SQL schema files folder path (relative to project root or absolute path)
SQL_SCHEMA_FOLDER=static/data/db_schema

# SQL knowledge base storage path (where vector database is stored)
SQL_KNOWLEDGE_BASE_PATH=static/data/vector_store/sql_knowledge_base
```

### Database Connection Configuration

```env
# Database type: mysql, sqlserver (or mssql), oracle
DB_TYPE=mysql

# Database connection details
DB_HOST=localhost
DB_PORT=3306
DB_INSTANCE=
DB_DATABASE=your_database_name
DB_USERNAME=your_username
DB_PASSWORD=your_password
```

## Configuration Options

### LLM Configuration

### `LLM_PROVIDER`
- **Purpose**: Specifies which LLM provider to use
- **Supported values**: `anthropic`, `openai`
- **Default**: `anthropic`
- **Examples**:
  - Anthropic: `LLM_PROVIDER=anthropic`
  - OpenAI: `LLM_PROVIDER=openai`

### `LLM_MODEL`
- **Purpose**: Specifies which model to use from the selected provider
- **Default**: `claude-3-5-sonnet-latest`
- **Anthropic Models**:
  - `claude-3-5-sonnet-latest` - Most capable, balanced performance
  - `claude-3-5-haiku-latest` - Fastest, most cost-effective
  - `claude-3-opus-latest` - Highest intelligence, most expensive
- **OpenAI Models**:
  - `gpt-4o` - Latest GPT-4 with vision capabilities
  - `gpt-4o-mini` - Faster, more cost-effective GPT-4
  - `gpt-4-turbo` - High performance GPT-4 variant
  - `gpt-3.5-turbo` - Fastest, most economical option

### API Key Requirements
- **Anthropic**: Requires `ANTHROPIC_API_KEY` when using `LLM_PROVIDER=anthropic`
- **OpenAI**: Requires `OPENAI_API_KEY` when using `LLM_PROVIDER=openai`
- **Fallback**: If initialization fails, system falls back to default Anthropic model

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

### `SQL_SCHEMA_FOLDER`
- **Purpose**: Specifies where database schema files are located
- **Default**: `static/data/db_schema`
- **Examples**:
  - Relative path: `static/data/db_schema`
  - Absolute path: `/home/user/database/schemas`
  - Windows path: `C:\Database\Schemas`

### `SQL_KNOWLEDGE_BASE_PATH`
- **Purpose**: Where the SQL schema vector database and metadata are stored
- **Default**: `static/data/vector_store/sql_knowledge_base`
- **Examples**:
  - Default: `static/data/vector_store/sql_knowledge_base`
  - Shared storage: `/shared/sql_knowledge_base`
  - Custom location: `/data/vector_db/schemas`

### Database Connection Settings

### `DB_TYPE`
- **Purpose**: Specifies the database type for SQL query execution
- **Supported values**: `mysql`, `sqlserver` (or `mssql`), `oracle`
- **Default**: `mysql`
- **Examples**:
  - MySQL: `DB_TYPE=mysql`
  - SQL Server: `DB_TYPE=sqlserver`
  - Oracle: `DB_TYPE=oracle`

### `DB_HOST`
- **Purpose**: Database server hostname or IP address
- **Default**: `localhost`
- **Examples**:
  - Local: `localhost`
  - Remote server: `192.168.1.100`
  - Named server: `database.company.com`

### `DB_PORT`
- **Purpose**: Database server port number
- **Default**: `3306` (MySQL default)
- **Common ports**:
  - MySQL: `3306`
  - SQL Server: `1433`
  - Oracle: `1521`

### `DB_INSTANCE`
- **Purpose**: Database instance name (primarily for SQL Server and Oracle)
- **Default**: (empty)
- **Examples**:
  - SQL Server: `SQLEXPRESS`
  - Oracle: `ORCL`
  - Leave empty for default instances

### `DB_DATABASE`
- **Purpose**: Name of the database/schema to connect to
- **Required**: Yes
- **Examples**:
  - `production_db`
  - `warehouse`
  - `analytics`

### `DB_USERNAME` and `DB_PASSWORD`
- **Purpose**: Database authentication credentials
- **Required**: Yes
- **Security**: Store sensitive credentials in `.env` file (not tracked in git)

## Supported File Types

### SOP Knowledge Base
The SOP knowledge base automatically processes these file types:

- **Text files**: `.txt`
- **Markdown files**: `.md`
- **reStructuredText files**: `.rst`
- **Microsoft Word files**: `.doc`, `.docx`
- **PDF files**: `.pdf`

### SQL Schema Knowledge Base
The SQL schema knowledge base processes:

- **Schema definition files**: `.txt` (structured database schema format)

## Database Driver Requirements

To connect and execute queries against databases, you need to install the appropriate database drivers:

### MySQL
```bash
pip install PyMySQL
```

### SQL Server
```bash
pip install pyodbc
```
**Note**: SQL Server also requires the ODBC Driver 17 for SQL Server to be installed on your system.

### Oracle
```bash
pip install cx_Oracle
```
**Note**: Oracle also requires the Oracle Instant Client to be installed on your system.

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
│       ├── db_schema/        # SQL schema files (SQL_SCHEMA_FOLDER)
│       │   └── BY_WMS_2021_1/
│       │       ├── client.txt
│       │       ├── adrmst.txt
│       │       └── ...
│       └── vector_store/
│           ├── sop_knowledge_base/    # SOP Vector database
│           │   ├── metadata.json
│           │   └── vector_store/
│           └── sql_knowledge_base/    # SQL Schema Vector database
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

## SQL Query Execution

The system now supports executing SQL queries generated from natural language input:

### Flow
1. **Natural Language Input**: User asks a data question
2. **SQL Generation**: LLM generates SQL query based on schema knowledge base
3. **SQL Execution**: System automatically connects to configured database and executes query
4. **Results Display**: Query results are formatted and displayed as a table

### Configuration Required
1. Set up database connection in `.env` file
2. Install appropriate database driver
3. Ensure database is accessible and credentials are correct

### Example Workflow
```
User: "Show me all customers from California"
↓
System generates: SELECT * FROM customers WHERE state = 'CA'
↓
System executes query against configured database
↓
Returns formatted table with results
```

## Usage Examples

### Basic Setup
1. Copy `.env.example` to `.env`
2. Add your Anthropic API key
3. Place SOP files in `static/data/sop/`
4. Configure database connection (optional, for SQL execution)
5. Run `langgraph dev`

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

The knowledge bases are automatically built when the application starts, but only if they don't already exist:

### Smart Initialization Process

1. **Checks** if knowledge base already exists by looking for:
   - `metadata.json` file in the knowledge base directory
   - `vector_store/` subdirectory with vector embeddings
2. **Skips** initialization if both files exist (saves time on subsequent runs)
3. **Rebuilds** only if knowledge base is missing or incomplete

### Build Process (when needed)

1. **Scans** source folders for supported file types
2. **Processes** each file to extract content
3. **Generates** metadata and relationships 
4. **Creates** vector embeddings for search
5. **Stores** everything in the knowledge base

### Knowledge Base Locations

- **SOP Knowledge Base**: `static/data/vector_store/sop_knowledge_base/`
- **SQL Schema Knowledge Base**: `static/data/vector_store/sql_knowledge_base/`

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
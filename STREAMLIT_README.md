# 🧠 LangGraph Knowledge Base Web Interface

A Streamlit-powered web interface for the LangGraph Knowledge Base application, providing easy access to both SOP and SQL schema knowledge bases.

## ✨ Features

### 🎯 **Multi-Modal Query Interface**
- **SQL Query Generation**: Ask natural language questions and get schema-aware SQL queries
- **SOP Document Search**: Find relevant Standard Operating Procedures and download them
- **General Q&A**: Get answers to technical questions and explanations

### 🔍 **Knowledge Base Exploration**
- Browse all available SOPs and database tables
- Search tables by column names
- View detailed schema information
- Real-time status of knowledge bases

### 📊 **Interactive Tools**
- Query history tracking
- Execution time monitoring
- Example queries for each category
- Advanced search and filtering options

### 🎨 **User-Friendly Interface**
- Clean, responsive design
- Syntax highlighting for SQL queries
- Expandable sections for detailed information
- Real-time status indicators

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install the application with Streamlit support
pip install -e .
```

### 2. Configure Environment
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# Required: ANTHROPIC_API_KEY
# Optional: LANGSMITH_API_KEY for tracing
```

### 3. Launch the Web Interface
```bash
# Option 1: Use the launch script (recommended)
python run_streamlit.py

# Option 2: Direct Streamlit command
streamlit run streamlit_app.py
```

### 4. Open in Browser
The application will automatically open in your default browser at `http://localhost:8501`

## 💡 Usage Examples

### SQL Queries
- "Show me all client information"
- "How many orders were placed last month?"
- "Find all inventory in warehouse A"
- "List users and their permissions"

### SOP Requests  
- "Show me the user onboarding procedure"
- "How do I handle customer complaints?"
- "What's the security incident response process?"
- "Give me the data backup SOP"

### General Questions
- "What is database normalization?"
- "How does REST API authentication work?"
- "Explain the difference between OLTP and OLAP"

## 🔧 Advanced Features

### Knowledge Base Management
- **Auto-Initialization**: Knowledge bases are built automatically on first run
- **Smart Caching**: Existing knowledge bases are reused to save time
- **Status Monitoring**: Real-time health checks for both SOP and SQL knowledge bases

### Query Routing
The application automatically routes queries to the appropriate handler:
- **SQL**: Data queries and database-related requests
- **SOP**: Procedure and process documentation requests
- **General**: Technical questions and explanations

### File Downloads
- SOP documents can be downloaded directly through the interface
- Support for multiple file formats (PDF, DOCX, TXT, MD)
- File information and metadata display

## 🎨 Interface Overview

### Main Query Interface
- Large text area for natural language input
- Submit and clear buttons
- Real-time query processing with progress indicators

### Sidebar Tools
- Knowledge base status indicators
- Example queries organized by category
- Quick access to browse functions

### Advanced Tools (Expandable)
- Browse all SOPs and database tables
- Search tables by column names
- View detailed schema information

### Query History
- Track recent queries and responses
- Execution time monitoring
- Easy re-execution of previous queries

## ⚙️ Configuration

### Streamlit Configuration
The app uses `.streamlit/config.toml` for Streamlit-specific settings:
- Theme colors and styling
- Server configuration
- Security settings

### Application Configuration
Uses the same `.env` configuration as the main application:
- `ANTHROPIC_API_KEY`: Required for LLM functionality
- `SQL_SCHEMA_FOLDER`: Path to database schema files
- `SOP_SOURCE_FOLDER`: Path to SOP documents
- Knowledge base storage paths

## 🛠️ Development

### File Structure
```
langgraph/
├── streamlit_app.py          # Main Streamlit application
├── run_streamlit.py          # Launch script with environment setup
├── .streamlit/
│   └── config.toml          # Streamlit configuration
└── src/agent/               # Core LangGraph application
    ├── graph.py            # Main workflow graph
    ├── sop_agent.py        # SOP knowledge base
    └── sql_agent.py        # SQL schema knowledge base
```

### Customization
- Modify `streamlit_app.py` to add new features
- Update `.streamlit/config.toml` for styling changes
- Add new example queries in the sidebar section

### Debugging
- Enable debug mode by setting `developmentMode = true` in config.toml
- View logs in the terminal where you launched the app
- Use browser developer tools for frontend debugging

## 🔒 Security Considerations

- API keys are loaded from environment variables
- No sensitive data is logged or displayed in the interface
- File downloads are restricted to the configured SOP folder
- CORS and XSRF protection can be enabled for production

## 📊 Performance

- Knowledge bases are cached for fast subsequent queries
- Vector search provides quick document retrieval
- Streamlit's caching optimizes repeated operations
- Concurrent processing for multiple knowledge bases

## 🐛 Troubleshooting

### Common Issues

1. **Knowledge Base Not Found**
   - Ensure `.env` file is configured correctly
   - Check that source folders exist and contain files
   - Verify file permissions

2. **API Key Errors**
   - Confirm `ANTHROPIC_API_KEY` is set in `.env`
   - Check API key validity and rate limits

3. **Slow Initial Load**
   - First run builds knowledge bases (can take several minutes)
   - Subsequent runs are much faster due to caching

4. **Import Errors**
   - Ensure all dependencies are installed: `pip install -e .`
   - Check that `src` folder is in Python path

### Getting Help
- Check the main application logs in the terminal
- Review the knowledge base initialization output
- Use the sidebar status indicators to identify issues

## 🚀 Production Deployment

For production deployment:

1. Set `headless = true` in `.streamlit/config.toml`
2. Configure proper authentication if needed
3. Use a production WSGI server
4. Set up proper logging and monitoring
5. Enable CORS and XSRF protection

Example production command:
```bash
streamlit run streamlit_app.py --server.headless true --server.port 8501
```
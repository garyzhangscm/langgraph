#!/usr/bin/env python3
"""Streamlit web interface for LangGraph Knowledge Base Application.

This app provides a user-friendly interface to interact with both SOP and SQL knowledge bases.
"""

import streamlit as st
import sys
from pathlib import Path
import time
import json
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the graph and functions
from agent.graph import router_workflow
from agent.sop_agent import (
    get_sop_recommendations, 
    list_all_sops,
    get_sop_file_info,
    prepare_file_download
)
from agent.sql_agent import (
    query_sql_schema,
    list_database_tables, 
    get_table_info,
    search_tables_by_column
)


def init_session_state():
    """Initialize session state variables."""
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""


def display_query_result(result: Dict[str, Any]):
    """Display the query result in a formatted way."""
    if "output" in result:
        st.markdown("### 🤖 Response")
        
        # Check if it's an SQL response (contains SQL code)
        output = result["output"]
        if "```sql" in output.lower() or "select " in output.lower():
            # It's likely an SQL query, display with syntax highlighting
            st.markdown(output)
        else:
            # Regular response
            st.markdown(output)
        
        # Show additional info if available
        if "sop_recommendations" in result and result["sop_recommendations"]:
            st.markdown("### 📋 Related SOPs")
            for i, rec in enumerate(result["sop_recommendations"][:3], 1):
                with st.expander(f"{i}. {rec['title']}"):
                    st.write(f"**Filename:** {rec['filename']}")
                    st.write(f"**Summary:** {rec['summary']}")
                    st.write(f"**Keywords:** {', '.join(rec['keywords'])}")


def display_sidebar():
    """Display sidebar with navigation and tools."""
    st.sidebar.title("🔧 Tools & Info")
    
    # Knowledge Base Status
    st.sidebar.markdown("### 📊 Knowledge Base Status")
    
    # Check SOP knowledge base
    try:
        sop_list = list_all_sops()
        st.sidebar.success(f"✅ SOP KB: {len(sop_list)} documents")
    except Exception as e:
        st.sidebar.error(f"❌ SOP KB: Error - {str(e)[:50]}...")
    
    # Check SQL knowledge base
    try:
        sql_tables = list_database_tables()
        st.sidebar.success(f"✅ SQL KB: {len(sql_tables)} tables")
    except Exception as e:
        st.sidebar.error(f"❌ SQL KB: Error - {str(e)[:50]}...")
    
    # Query Examples
    st.sidebar.markdown("### 💡 Example Queries")
    
    example_queries = {
        "SQL Queries": [
            "Show me all client information",
            "How many orders were placed last month?",
            "Find all inventory in warehouse A",
            "List all users and their roles"
        ],
        "SOP Requests": [
            "Show me the user onboarding procedure",
            "How do I handle customer complaints?",
            "What's the process for returns?",
            "Give me the security incident response SOP"
        ],
        "General Questions": [
            "What is the difference between OLTP and OLAP?",
            "How does authentication work?",
            "Explain database normalization",
            "What are best practices for data backup?"
        ]
    }
    
    for category, queries in example_queries.items():
        with st.sidebar.expander(f"📝 {category}"):
            for query in queries:
                if st.button(query, key=f"example_{hash(query)}"):
                    st.session_state.current_query = query
                    st.rerun()


def display_advanced_tools():
    """Display advanced tools and information."""
    with st.expander("🔍 Advanced Tools"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Browse SOPs")
            if st.button("List All SOPs"):
                try:
                    sops = list_all_sops()
                    st.write(f"Found {len(sops)} SOPs:")
                    for sop in sops[:10]:  # Show first 10
                        st.write(f"• **{sop['title']}** - {sop['filename']}")
                    if len(sops) > 10:
                        st.write(f"... and {len(sops) - 10} more")
                except Exception as e:
                    st.error(f"Error listing SOPs: {e}")
        
        with col2:
            st.markdown("#### 🗄️ Browse Database Tables")
            if st.button("List All Tables"):
                try:
                    tables = list_database_tables()
                    st.write(f"Found {len(tables)} tables:")
                    for table in tables[:10]:  # Show first 10
                        st.write(f"• **{table['table_name']}** ({table['columns_count']} columns)")
                    if len(tables) > 10:
                        st.write(f"... and {len(tables) - 10} more")
                except Exception as e:
                    st.error(f"Error listing tables: {e}")
        
        # Column search
        st.markdown("#### 🔍 Search Tables by Column")
        column_search = st.text_input("Enter column name to search:", placeholder="e.g., client_id, user_name")
        if column_search and st.button("Search Columns"):
            try:
                results = search_tables_by_column(column_search)
                if results:
                    st.write(f"Found {len(results)} tables with columns matching '{column_search}':")
                    for result in results[:5]:
                        st.write(f"• **{result['table_name']}** - {len(result['matching_columns'])} matching columns")
                else:
                    st.write(f"No tables found with columns matching '{column_search}'")
            except Exception as e:
                st.error(f"Error searching columns: {e}")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="LangGraph Knowledge Base",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.title("🧠 LangGraph Knowledge Base Assistant")
    st.markdown("Ask questions about SOPs, generate SQL queries, or get general information!")
    
    # Display sidebar
    display_sidebar()
    
    # Main query interface
    st.markdown("### 💬 Ask Your Question")
    
    # Use session state for query input
    query_input = st.text_area(
        "Enter your question or request:",
        value=st.session_state.current_query,
        placeholder="e.g., 'Show me all clients' or 'How do I onboard a new user?' or 'Explain database indexing'",
        height=100,
        key="query_textarea"
    )
    
    # Update session state when input changes
    if query_input != st.session_state.current_query:
        st.session_state.current_query = query_input
    
    # Query execution
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        submit_button = st.button("🚀 Submit Query", type="primary")
    
    with col2:
        clear_button = st.button("🗑️ Clear")
        if clear_button:
            st.session_state.current_query = ""
            st.rerun()
    
    # Execute query
    if submit_button and query_input.strip():
        with st.spinner("🤔 Processing your question..."):
            try:
                # Create input for the graph
                graph_input = {"input": query_input.strip()}
                
                # Track execution time
                start_time = time.time()
                
                # Run the graph
                result = router_workflow.invoke(graph_input)
                
                execution_time = time.time() - start_time
                
                # Display results
                display_query_result(result)
                
                # Show execution info
                with st.expander("ℹ️ Execution Details"):
                    st.write(f"**Execution Time:** {execution_time:.2f} seconds")
                    st.write(f"**Route Decision:** {result.get('decision', 'Unknown')}")
                    st.write(f"**Query Type:** {type(result.get('output', '')).__name__}")
                
                # Add to history
                st.session_state.query_history.append({
                    "query": query_input.strip(),
                    "result": result,
                    "timestamp": time.time(),
                    "execution_time": execution_time
                })
                
                # Limit history to last 10 queries
                if len(st.session_state.query_history) > 10:
                    st.session_state.query_history = st.session_state.query_history[-10:]
                    
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                st.exception(e)
    
    elif submit_button:
        st.warning("⚠️ Please enter a question or request.")
    
    # Display advanced tools
    display_advanced_tools()
    
    # Query History
    if st.session_state.query_history:
        st.markdown("### 📝 Recent Queries")
        
        # Show history in tabs or expander
        with st.expander(f"View History ({len(st.session_state.query_history)} queries)"):
            for i, item in enumerate(reversed(st.session_state.query_history)):
                with st.container():
                    st.markdown(f"**Query {len(st.session_state.query_history) - i}:** {item['query']}")
                    st.markdown(f"*Executed in {item['execution_time']:.2f}s*")
                    
                    # Show result summary
                    if item['result'].get('output'):
                        output_preview = item['result']['output'][:200]
                        if len(item['result']['output']) > 200:
                            output_preview += "..."
                        st.markdown(f"**Response:** {output_preview}")
                    
                    st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🧠 Powered by LangGraph • 🤖 Claude AI • 🔍 Vector Search</p>
        </div>
        """, 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
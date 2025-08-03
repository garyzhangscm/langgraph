#!/usr/bin/env python3
"""Demo script showing how to use the LangGraph Knowledge Base programmatically."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.graph import router_workflow
from agent.sop_agent import list_all_sops, get_sop_recommendations
from agent.sql_agent import list_database_tables, query_sql_schema


def demo_sql_queries():
    """Demonstrate SQL query generation."""
    print("🔍 SQL Query Generation Demo")
    print("=" * 40)
    
    sql_queries = [
        "Show me all client information",
        "How many tables are related to inventory?",
        "Find all users and their roles"
    ]
    
    for query in sql_queries:
        print(f"\nQuery: '{query}'")
        try:
            result = router_workflow.invoke({"input": query})
            output = result.get("output", "No output")
            print(f"Response: {output[:200]}{'...' if len(output) > 200 else ''}")
        except Exception as e:
            print(f"Error: {e}")


def demo_sop_queries():
    """Demonstrate SOP document search."""
    print("\n📋 SOP Document Search Demo")
    print("=" * 40)
    
    sop_queries = [
        "How do I onboard a new user?",
        "What's the process for handling incidents?",
        "Show me security procedures"
    ]
    
    for query in sop_queries:
        print(f"\nQuery: '{query}'")
        try:
            result = router_workflow.invoke({"input": query})
            output = result.get("output", "No output")
            print(f"Response: {output[:200]}{'...' if len(output) > 200 else ''}")
        except Exception as e:
            print(f"Error: {e}")


def demo_knowledge_base_stats():
    """Show knowledge base statistics."""
    print("\n📊 Knowledge Base Statistics")
    print("=" * 40)
    
    try:
        # SOP stats
        sops = list_all_sops()
        print(f"📋 SOP Documents: {len(sops)}")
        if sops:
            print(f"   Sample SOPs:")
            for sop in sops[:3]:
                print(f"   - {sop['title']}")
    except Exception as e:
        print(f"Error getting SOP stats: {e}")
    
    try:
        # SQL stats
        tables = list_database_tables()
        print(f"🗄️  Database Tables: {len(tables)}")
        if tables:
            print(f"   Sample Tables:")
            for table in tables[:3]:
                print(f"   - {table['table_name']} ({table['columns_count']} columns)")
    except Exception as e:
        print(f"Error getting SQL stats: {e}")


def demo_general_queries():
    """Demonstrate general knowledge queries."""
    print("\n🧠 General Knowledge Demo")
    print("=" * 40)
    
    general_queries = [
        "What is database normalization?",
        "Explain REST API best practices",
        "How does authentication work?"
    ]
    
    for query in general_queries:
        print(f"\nQuery: '{query}'")
        try:
            result = router_workflow.invoke({"input": query})
            output = result.get("output", "No output")
            print(f"Response: {output[:200]}{'...' if len(output) > 200 else ''}")
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Run the complete demo."""
    print("🧠 LangGraph Knowledge Base Demo")
    print("="*50)
    print("This demo shows the three main capabilities:")
    print("1. SQL query generation from natural language")
    print("2. SOP document search and retrieval")
    print("3. General knowledge and explanations")
    print()
    
    # Show knowledge base stats first
    demo_knowledge_base_stats()
    
    # Demo each capability
    demo_sql_queries()
    demo_sop_queries()
    demo_general_queries()
    
    print("\n" + "="*50)
    print("🌐 For interactive usage, run: python run_streamlit.py")
    print("📖 For detailed docs, see: STREAMLIT_README.md")


if __name__ == "__main__":
    main()
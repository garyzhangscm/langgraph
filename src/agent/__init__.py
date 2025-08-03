"""New LangGraph Agent.

This module defines a custom graph with SOP and SQL Schema knowledge base functionality.
"""

from agent.graph import graph
from agent.sop_agent import (
    build_sop_knowledge_base, 
    query_sop_knowledge_base, 
    get_sop_recommendations,
    get_sop_source_folder,
    get_sop_knowledge_base_path,
    prepare_file_download,
    copy_sop_file_to_downloads,
    get_sop_file_info
)
from agent.sql_agent import (
    build_sql_knowledge_base,
    query_sql_schema,
    get_table_info,
    search_tables_by_column,
    list_database_tables,
    get_sql_schema_folder,
    get_sql_knowledge_base_path
)

__all__ = [
    "graph", 
    # SOP Agent
    "build_sop_knowledge_base", 
    "query_sop_knowledge_base", 
    "get_sop_recommendations",
    "get_sop_source_folder",
    "get_sop_knowledge_base_path",
    "prepare_file_download",
    "copy_sop_file_to_downloads",
    "get_sop_file_info",
    # SQL Agent
    "build_sql_knowledge_base",
    "query_sql_schema",
    "get_table_info",
    "search_tables_by_column",
    "list_database_tables",
    "get_sql_schema_folder",
    "get_sql_knowledge_base_path"
]

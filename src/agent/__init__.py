"""New LangGraph Agent.

This module defines a custom graph with SOP knowledge base functionality.
"""

from agent.graph import graph
from agent.sop_agent import (
    build_sop_knowledge_base, 
    query_sop_knowledge_base, 
    get_sop_recommendations,
    get_sop_source_folder,
    get_sop_knowledge_base_path
)

__all__ = [
    "graph", 
    "build_sop_knowledge_base", 
    "query_sop_knowledge_base", 
    "get_sop_recommendations",
    "get_sop_source_folder",
    "get_sop_knowledge_base_path"
]

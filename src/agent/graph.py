"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field
from typing_extensions import Literal
from langchain_core.messages import HumanMessage, SystemMessage

from IPython.display import Image, display
from typing_extensions import NotRequired
from typing import TypedDict, Optional, List, Dict, Any
import asyncio
import os
from pathlib import Path
import json

from agent.sop_agent import (
    query_sop_knowledge_base, 
    build_sop_knowledge_base, 
    get_sop_recommendations,
    get_sop_source_folder,
    prepare_file_download,
    copy_sop_file_to_downloads,
    get_sop_file_info
)
from agent.sql_agent import (
    build_sql_knowledge_base,
    get_sql_schema_folder
)


llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

# Debug helper function
def debug_state(state: State, node_name: str = "Unknown", variables: Dict[str, Any] = None):
    """Helper function to debug state and variables at any point in the graph."""
    print(f"\n=== DEBUG: {node_name} ===")
    print(f"State: {json.dumps(state, indent=2, default=str)}")
    
    if variables:
        print(f"Variables:")
        for key, value in variables.items():
            if isinstance(value, (list, dict)):
                print(f"  {key}: {json.dumps(value, indent=4, default=str)}")
            else:
                print(f"  {key}: {value}")
    print("=" * (20 + len(node_name)))

# Schema for structured output to use as routing logic
class Route(BaseModel):
    step: Literal["sql", "SOP", "answer"] = Field(
        None, description="The next step in the routing process"
    )
    
# Augment the LLM with schema for structured output
router = llm.with_structured_output(Route)

class Configuration(TypedDict):
    """Configurable parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    my_configurable_param: str
    sop_source_folder: NotRequired[Optional[str]]
    sop_knowledge_base_path: NotRequired[Optional[str]]

 
class State(TypedDict):
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """
    input: str
    decision: NotRequired[Optional[str]]
    output: NotRequired[Optional[str]]
    sop_recommendations: NotRequired[Optional[List[Dict[str, Any]]]]
    build_kb_folder: NotRequired[Optional[str]]
    sop_filename: NotRequired[Optional[str]]
    download_info: NotRequired[Optional[Dict[str, Any]]]
    download_requested: NotRequired[Optional[bool]]
    
# Node 1 - generate SQL query
def llm_call_generate_sql(state: State):
    """Generate SQL queries based on user input and database schema knowledge base."""
    from agent.sql_agent import query_sql_schema, get_table_info, list_database_tables

    user_input = state["input"]

    # Debug the state and input
    debug_state(state, "SQL_NODE_START", {"user_input": user_input})
    
    # Uncomment the next line to add a breakpoint for debugging
    # import pdb; pdb.set_trace()
    
    try:
        # Query the SQL schema knowledge base for relevant tables
        relevant_tables = query_sql_schema(user_input)
        
        # Debug the query results
        debug_state(state, "AFTER_SCHEMA_QUERY", {
            "query": user_input,
            "relevant_tables_count": len(relevant_tables),
            "relevant_tables": relevant_tables[:3] if relevant_tables else []  # Show first 2 tables
        })

        if relevant_tables:
            # Get detailed information for the most relevant tables
            schema_context = []
            for table in relevant_tables[:3]:  # Use top 3 most relevant tables
                table_details = get_table_info(table['table_name'])
                if table_details:
                    # Format table information for the prompt
                    columns_info = "\n".join([
                        f"  - {col['name']} ({col['data_type']}{',' if col['nullable'] else ', NOT NULL'}): {col['description']}"
                        # for col in table_details['columns'][:10]  # Limit to first 10 columns to avoid prompt overflow
                        for col in table_details['columns']
                    ])
                    
                    table_info = f"""
**Table: {table_details['table_name']}**
Description: {table_details['description']}
Columns:
{columns_info}
"""
                    schema_context.append(table_info)
            
            schema_text = "\n".join(schema_context)
            
            # Create system prompt with actual schema information
            system_prompt = f"""You are a SQL expert assistant with access to a database schema knowledge base. 
Based on the user's natural language query, generate appropriate SQL queries using the following relevant database schema information:

{schema_text}

Guidelines for SQL generation:
1. Use only the tables and columns shown in the schema above
2. Pay attention to data types and nullability constraints
3. Include proper JOIN conditions when querying multiple tables
4. Use descriptive aliases for tables when needed
5. Format the SQL query clearly with proper indentation
6. Include comments explaining complex parts of the query
7. If the user's request cannot be fulfilled with the available schema, explain what's missing

Provide clean, executable SQL queries with proper formatting and explanations."""

        else:
            # If no relevant tables found, provide general guidance
            try:
                # Try to get a list of available tables
                all_tables = list_database_tables()
                table_list = ", ".join([table['table_name'] for table in all_tables[:10]])
                
                system_prompt = f"""You are a SQL expert assistant. I couldn't find tables directly relevant to your query in the database schema knowledge base.

Available tables in the database include: {table_list}{'...' if len(all_tables) > 10 else ''}

Please rephrase your query to be more specific about which tables or data you're interested in. I can help you generate SQL queries once I understand which tables are relevant to your needs.

If you're looking for specific types of data, try using keywords like:
- "client" or "customer" for customer-related tables
- "order" for order-related tables  
- "inventory" for inventory-related tables
- "address" for address-related tables"""
                
            except Exception:
                system_prompt = """You are a SQL expert assistant. I'm having trouble accessing the database schema knowledge base right now. 

Please provide more specific information about:
1. Which tables you want to query
2. What specific data you're looking for
3. Any relationships between tables you want to explore

This will help me generate better SQL queries for you."""
    
    except Exception as e:
        # Fallback if SQL agent fails
        system_prompt = f"""You are a SQL expert assistant. I encountered an issue accessing the database schema knowledge base: {str(e)}

I'll provide general SQL guidance, but for more accurate queries specific to your database schema, please provide:
1. Table names you want to query
2. Column names you're interested in
3. Any specific relationships or constraints

Provide clean, executable SQL queries with proper formatting."""

    debug_state(state, "SQL_NODE_LLM_QUERY", {"prompt": system_prompt})
    # Generate SQL query response
    result = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ])
    
    return {"output": result.content}


# Node 2 - get SOP Document
def llm_call_get_sop_filename(state: State):
    """Find the right SOP using knowledge base"""
    
    user_query = state["input"]
    
    # Check if this is a request to build knowledge base
    if "build knowledge base" in user_query.lower() or "build kb" in user_query.lower():
        # Extract folder path from query if possible
        import re
        folder_match = re.search(r'from\s+["\'](.*?)["\']|folder\s+["\'](.*?)["\']', user_query, re.IGNORECASE)
        if folder_match:
            folder_path = folder_match.group(1) or folder_match.group(2)
            try:
                results = build_sop_knowledge_base(folder_path)
                return {
                    "output": f"Knowledge base built successfully from {folder_path}. "
                             f"Processed: {results['processed']}, Updated: {results['updated']}, "
                             f"Skipped: {results['skipped']}, Errors: {results['errors']}"
                }
            except Exception as e:
                return {"output": f"Error building knowledge base: {str(e)}"}
        else:
            return {
                "output": "To build knowledge base, please specify folder path: 'build knowledge base from \"/path/to/sop/folder\"'"
            }
    
    # Query the knowledge base for SOP filename
    try:
        best_filename = query_sop_knowledge_base(user_query)
        recommendations = get_sop_recommendations(user_query, top_k=3)
        
        # Check if user wants to download the file
        download_requested = any(keyword in user_query.lower() for keyword in ["download", "get file", "send me", "give me the file"])
        
        if best_filename:
            output = f"Best matching SOP: {best_filename}"
            if recommendations:
                output += "\n\nOther relevant SOPs:"
                for i, rec in enumerate(recommendations[:3], 1):
                    output += f"\n{i}. {rec['filename']} - {rec['title']}"
            
            if download_requested:
                output += f"\n\nPreparing '{best_filename}' for download..."
        else:
            output = "No matching SOP found in knowledge base. You may need to build or update the knowledge base first."
        
        return {
            "output": output,
            "sop_recommendations": recommendations,
            "sop_filename": best_filename,
            "download_requested": download_requested
        }
        
    except Exception as e:
        return {"output": f"Error querying SOP knowledge base: {str(e)}"}


# Node 3 - get SOP Document
def llm_call_get_answer(state: State):
    """Find the answer"""

    result = llm.invoke(state["input"])
    return {"output": result.content}

# Node 4 - Handle SOP file download
def handle_sop_download(state: State):
    """Handle downloading of SOP files"""
    
    sop_filename = state.get("sop_filename")
    
    if not sop_filename:
        return {
            "output": "No SOP file specified for download. Please search for an SOP first.",
            "download_info": {"success": False, "error": "No filename provided"}
        }
    
    try:
        # Prepare the file for download
        download_info = prepare_file_download(sop_filename)
        
        if download_info["success"]:
            if download_info["download_method"] == "base64_encoded":
                output = f"✅ File '{sop_filename}' is ready for download!\n\n"
                output += f"📄 **File Details:**\n"
                output += f"   • Title: {download_info['title']}\n"
                output += f"   • Size: {download_info['file_size_mb']} MB\n"
                output += f"   • Format: {download_info['file_extension']}\n\n"
                output += f"📋 **Download Instructions:**\n"
                output += f"   The file content is encoded and ready for download.\n"
                output += f"   You can save this content to a file with the same name and extension."
            else:
                output = f"📁 File '{sop_filename}' is available for download!\n\n"
                output += f"📄 **File Details:**\n"
                output += f"   • Title: {download_info['title']}\n"
                output += f"   • Size: {download_info['file_size_mb']} MB\n"
                output += f"   • Format: {download_info['file_extension']}\n"
                output += f"   • Location: {download_info['file_path']}\n\n"
                output += f"📋 **Access Instructions:**\n"
                output += f"   The file is located at the path above.\n"
                output += f"   You can copy it to your preferred location."
            
            # Also try to copy to downloads folder
            try:
                copy_result = copy_sop_file_to_downloads(sop_filename)
                if copy_result["success"]:
                    output += f"\n\n📥 **Auto-copied to downloads:**\n"
                    output += f"   File copied to: {copy_result['destination_path']}"
            except Exception:
                pass  # Silent fail for auto-copy
                
        else:
            output = f"❌ Error preparing '{sop_filename}' for download:\n"
            output += f"   {download_info['error']}\n\n"
            
            if "available_files" in download_info:
                output += f"📋 **Available SOP files:**\n"
                for i, filename in enumerate(download_info["available_files"][:5], 1):
                    output += f"   {i}. {filename}\n"
        
        return {
            "output": output,
            "download_info": download_info
        }
        
    except Exception as e:
        return {
            "output": f"❌ Unexpected error during download preparation: {str(e)}",
            "download_info": {"success": False, "error": str(e)}
        }
    
def router_node(state: State) -> State:
    user_input = state["input"]
    
    router_prompt = """
        LangGraph Router System Prompt
        You are an intelligent routing agent that analyzes user requests and determines the appropriate path for handling them. Based on the user's input, you must route to exactly ONE of three paths:
        Routing Options
        Return exactly one of these values:

        sql - For database queries and data retrieval
        sop - For Standard Operating Procedure requests
        answer - For knowledge base questions

        Routing Logic
        Route to "sql" when:

        User wants to retrieve, query, find, or get data from a database
        Requests involve specific data points, records, counts, or statistics
        Keywords indicating data operations: "show me", "find all", "count", "list", "get data", "query", "search records"
        Examples:

        "Show me all orders from last month"
        "How many users are active?"
        "Find customers in California"
        "Get the sales data for Q3"
        "List all products with low inventory"



        Route to "sop" when:

        User explicitly asks for an SOP, procedure, or process document
        Requests for step-by-step instructions or workflows
        Requests to build or update the knowledge base
        Keywords: "SOP", "procedure", "process", "how to", "steps", "workflow", "protocol", "guideline", "build knowledge base", "build kb"
        Examples:

        "Show me the SOP for user onboarding"
        "I need the procedure for handling returns"
        "What's the process for escalating issues?"
        "Give me the workflow for order processing"
        "Where is the SOP for data backup?"
        "Build knowledge base from '/path/to/sop/folder'"
        "Update the SOP knowledge base"



        Route to "answer" when:

        User asks general questions requiring knowledge-based responses
        Requests for explanations, definitions, or conceptual information
        Troubleshooting or advisory questions
        Keywords: "what is", "why", "how does", "explain", "tell me about", "help with"
        Examples:

        "What is the difference between X and Y?"
        "Why is my system running slowly?"
        "How does authentication work?"
        "Explain the benefits of this feature"
        "What should I do if I encounter error X?"



        Decision Framework

        Check for explicit SOP requests first - If user mentions procedures, workflows, or asks "how to" do something operationally
        Check for data/query needs second - If user wants specific data, numbers, records, or database information
        Default to knowledge base - If user asks general questions, needs explanations, or seeks advice

        Edge Cases

        "How to query the database?" → answer (asking about methodology, not requesting actual data)
        "Show me the SOP and also get the data" → sop (prioritize the explicit SOP request)
        "What's in our customer table?" → sql (requesting actual data content)
        "How do I write an SOP?" → answer (asking for guidance, not requesting a specific SOP)

        Output Format
        Respond with ONLY the routing decision. No explanations, no additional text.
        Valid responses: sql, sop, or answer
        Examples
        Input: "Show me all customers who placed orders last week"
        Output: sql
        Input: "I need the SOP for handling customer complaints"
        Output: sop
        Input: "What is the best practice for password security?"
        Output: answer
        Input: "How many active users do we have?"
        Output: sql
        Input: "Explain the difference between REST and GraphQL"
        Output: answer
        Input: "Where can I find the procedure for system maintenance?"
        Output: sop

        Now analyze the user's request and return the appropriate routing decision.
         
    """
    
    # Get routing decision from LLM
    decision = router.invoke(
        [
            SystemMessage(
                content=router_prompt
            ),
            HumanMessage(content=state["input"]),
        ]
    )

    return {"decision": decision.step}
     
def route_decision(state: State) -> str:
    route = state["decision"]
    
    if route == "sql":
        return "llm_call_generate_sql"
    elif route == "sop" or route == "SOP":
        return "llm_call_get_sop_filename"
    else:
        return "llm_call_get_answer"

def route_after_sop(state: State) -> str:
    """Route after SOP node - check if download is requested"""
    download_requested = state.get("download_requested", False)
    sop_filename = state.get("sop_filename")
    
    # If download is requested and we have a filename, go to download node
    if download_requested and sop_filename:
        return "handle_sop_download"
    else:
        return "END"
        
         

# Build workflow
router_builder = StateGraph(State)

# Add nodes
router_builder.add_node("llm_call_generate_sql", llm_call_generate_sql)
router_builder.add_node("llm_call_get_sop_filename", llm_call_get_sop_filename)
router_builder.add_node("llm_call_get_answer", llm_call_get_answer)
router_builder.add_node("handle_sop_download", handle_sop_download)
router_builder.add_node("router_node", router_node)

# Add edges to connect nodes
router_builder.add_edge(START, "router_node")

# Initial routing based on query type
router_builder.add_conditional_edges(
    "router_node",
    route_decision,
    {  # Name returned by route_decision : Name of next node to visit
        "llm_call_generate_sql": "llm_call_generate_sql",
        "llm_call_get_sop_filename": "llm_call_get_sop_filename",
        "llm_call_get_answer": "llm_call_get_answer",
    },
)

# Direct endings for SQL and answer nodes
router_builder.add_edge("llm_call_generate_sql", END)
router_builder.add_edge("llm_call_get_answer", END)

# Conditional routing after SOP node - either download or end
router_builder.add_conditional_edges(
    "llm_call_get_sop_filename",
    route_after_sop,
    {
        "handle_sop_download": "handle_sop_download",
        "END": END
    }
)

# Download node ends the flow
router_builder.add_edge("handle_sop_download", END)

# Asynchronous initialization functions
async def initialize_sop_knowledge_base():
    """Initialize the SOP knowledge base from configured folder only if it doesn't exist."""
    try:
        from agent.sop_agent import get_sop_knowledge_base_path
        
        # Check if knowledge base already exists
        sop_kb_config = get_sop_knowledge_base_path()
        
        # If it's a relative path, make it relative to project root
        if not os.path.isabs(sop_kb_config):
            project_root = Path(__file__).parent.parent.parent
            sop_kb_path = project_root / sop_kb_config
        else:
            sop_kb_path = Path(sop_kb_config)
        
        # Check if knowledge base files already exist
        metadata_file = sop_kb_path / "metadata.json"
        vector_store_path = sop_kb_path / "vector_store"
        
        if metadata_file.exists() and vector_store_path.exists():
            print(f"SOP knowledge base already exists at {sop_kb_path}")
            print("Skipping SOP knowledge base initialization.")
            return {"status": "skipped", "reason": "already_exists"}
        
        # Get configured SOP source folder
        sop_folder_config = get_sop_source_folder()
        
        # If it's a relative path, make it relative to project root
        if not os.path.isabs(sop_folder_config):
            project_root = Path(__file__).parent.parent.parent
            sop_folder = project_root / sop_folder_config
        else:
            sop_folder = Path(sop_folder_config)
        
        if sop_folder.exists():
            print(f"Building SOP knowledge base from {sop_folder}...")
            print(f"Configuration: SOP_SOURCE_FOLDER={sop_folder_config}")
            
            # Run knowledge base building in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                build_sop_knowledge_base, 
                str(sop_folder)
            )
            
            print(f"SOP Knowledge base initialized successfully!")
            print(f"Processed: {results['processed']}, Updated: {results['updated']}, "
                  f"Skipped: {results['skipped']}, Errors: {results['errors']}")
            
            return results
        else:
            print(f"SOP folder not found at {sop_folder}")
            print(f"Configuration: SOP_SOURCE_FOLDER={sop_folder_config}")
            print("Skipping SOP knowledge base initialization.")
            return None
            
    except Exception as e:
        print(f"Error initializing SOP knowledge base: {e}")
        return None

async def initialize_sql_knowledge_base():
    """Initialize the SQL schema knowledge base from configured folder only if it doesn't exist."""
    try:
        from agent.sql_agent import get_sql_knowledge_base_path
        
        # Check if knowledge base already exists
        sql_kb_config = get_sql_knowledge_base_path()
        
        # If it's a relative path, make it relative to project root
        if not os.path.isabs(sql_kb_config):
            project_root = Path(__file__).parent.parent.parent
            sql_kb_path = project_root / sql_kb_config
        else:
            sql_kb_path = Path(sql_kb_config)
        
        # Check if knowledge base files already exist
        metadata_file = sql_kb_path / "metadata.json"
        vector_store_path = sql_kb_path / "vector_store"
        
        if metadata_file.exists() and vector_store_path.exists():
            print(f"SQL knowledge base already exists at {sql_kb_path}")
            print("Skipping SQL knowledge base initialization.")
            return {"status": "skipped", "reason": "already_exists"}
        
        # Get configured SQL schema source folder
        sql_folder_config = get_sql_schema_folder()
        
        # If it's a relative path, make it relative to project root
        if not os.path.isabs(sql_folder_config):
            project_root = Path(__file__).parent.parent.parent
            sql_folder = project_root / sql_folder_config
        else:
            sql_folder = Path(sql_folder_config)
        
        if sql_folder.exists():
            print(f"Building SQL schema knowledge base from {sql_folder}...")
            print(f"Configuration: SQL_SCHEMA_FOLDER={sql_folder_config}")
            
            # Run knowledge base building in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                build_sql_knowledge_base, 
                str(sql_folder)
            )
            
            print(f"SQL Schema knowledge base initialized successfully!")
            print(f"Processed: {results['processed']}, Updated: {results['updated']}, "
                  f"Skipped: {results['skipped']}, Errors: {results['errors']}")
            
            return results
        else:
            print(f"SQL schema folder not found at {sql_folder}")
            print(f"Configuration: SQL_SCHEMA_FOLDER={sql_folder_config}")
            print("Skipping SQL knowledge base initialization.")
            return None
            
    except Exception as e:
        print(f"Error initializing SQL knowledge base: {e}")
        return None

async def initialize_all_knowledge_bases():
    """Initialize both SOP and SQL knowledge bases only if they don't already exist.
    
    This function checks for the existence of:
    - SOP KB: static/data/vector_store/sop_knowledge_base/metadata.json and vector_store/
    - SQL KB: static/data/vector_store/sql_knowledge_base/metadata.json and vector_store/
    
    If both files exist for a knowledge base, initialization is skipped to save time.
    If either file is missing, the knowledge base is rebuilt from source files.
    """
    print("=== Initializing Knowledge Bases ===")
    
    # Initialize both knowledge bases concurrently
    sop_task = asyncio.create_task(initialize_sop_knowledge_base())
    sql_task = asyncio.create_task(initialize_sql_knowledge_base())
    
    sop_results, sql_results = await asyncio.gather(sop_task, sql_task, return_exceptions=True)
    
    print("=== Knowledge Base Initialization Complete ===")
    
    return {
        "sop_results": sop_results if not isinstance(sop_results, Exception) else str(sop_results),
        "sql_results": sql_results if not isinstance(sql_results, Exception) else str(sql_results)
    }

# Initialize both knowledge bases on module load
try:
    # Only run if we're in an async context or can create one
    if asyncio.get_event_loop().is_running():
        # If already in an async context, schedule the task
        asyncio.create_task(initialize_all_knowledge_bases())
    else:
        # If not in async context, run it
        asyncio.run(initialize_all_knowledge_bases())
except RuntimeError:
    # If no event loop exists, create one and run
    try:
        asyncio.run(initialize_all_knowledge_bases())
    except Exception as e:
        print(f"Could not initialize knowledge bases: {e}")

# Compile workflow
router_workflow = router_builder.compile()

# Show the workflow
display(Image(router_workflow.get_graph().draw_mermaid_png()))


# Define the graph
# graph = (
#     StateGraph(State, config_schema=Configuration)
#    .add_node(call_model)
#    .add_edge("__start__", "call_model")
#    .compile(name="New Graph")
#)
# Return the graph variable for LangSmith
graph = router_builder

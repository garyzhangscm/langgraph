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

from agent.sop_agent import query_sop_knowledge_base, build_sop_knowledge_base, get_sop_recommendations


llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

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
    
# Node 1 - generate SQL query
def llm_call_generate_sql(state: State):
    """Generate SQL Query"""

    result = llm.invoke(state["input"])
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
        
        if best_filename:
            output = f"Best matching SOP: {best_filename}"
            if recommendations:
                output += "\n\nOther relevant SOPs:"
                for i, rec in enumerate(recommendations[:3], 1):
                    output += f"\n{i}. {rec['filename']} - {rec['title']}"
        else:
            output = "No matching SOP found in knowledge base. You may need to build or update the knowledge base first."
        
        return {
            "output": output,
            "sop_recommendations": recommendations
        }
        
    except Exception as e:
        return {"output": f"Error querying SOP knowledge base: {str(e)}"}


# Node 2 - get SOP Document
def llm_call_get_answer(state: State):
    """Find the answer"""

    result = llm.invoke(state["input"])
    return {"output": result.content}
    
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
        
         

# Build workflow
router_builder = StateGraph(State)

# Add nodes
router_builder.add_node("llm_call_generate_sql", llm_call_generate_sql)
router_builder.add_node("llm_call_get_sop_filename", llm_call_get_sop_filename)
router_builder.add_node("llm_call_get_answer", llm_call_get_answer)
router_builder.add_node("router_node", router_node)

# Add edges to connect nodes
router_builder.add_edge(START, "router_node")
router_builder.add_conditional_edges(
    "router_node",
    route_decision,
    {  # Name returned by route_decision : Name of next node to visit
        "llm_call_generate_sql": "llm_call_generate_sql",
        "llm_call_get_sop_filename": "llm_call_get_sop_filename",
        "llm_call_get_answer": "llm_call_get_answer",
    },
)
router_builder.add_edge("llm_call_generate_sql", END)
router_builder.add_edge("llm_call_get_sop_filename", END)
router_builder.add_edge("llm_call_get_answer", END)

# Asynchronous initialization function
async def initialize_sop_knowledge_base():
    """Initialize the SOP knowledge base from static/data/sop folder."""
    try:
        # Get the project root directory (where this file is located)
        project_root = Path(__file__).parent.parent.parent
        sop_folder = project_root / "static" / "data" / "sop"
        
        if sop_folder.exists():
            print(f"Building SOP knowledge base from {sop_folder}...")
            
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
            print(f"SOP folder not found at {sop_folder}. Skipping knowledge base initialization.")
            return None
            
    except Exception as e:
        print(f"Error initializing SOP knowledge base: {e}")
        return None

# Initialize the knowledge base on module load
try:
    # Only run if we're in an async context or can create one
    if asyncio.get_event_loop().is_running():
        # If already in an async context, schedule the task
        asyncio.create_task(initialize_sop_knowledge_base())
    else:
        # If not in async context, run it
        asyncio.run(initialize_sop_knowledge_base())
except RuntimeError:
    # If no event loop exists, create one and run
    try:
        asyncio.run(initialize_sop_knowledge_base())
    except Exception as e:
        print(f"Could not initialize SOP knowledge base: {e}")

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

"""Integration tests for SOP agent with the main graph."""

import pytest
from unittest.mock import patch

from agent import graph

pytestmark = pytest.mark.anyio


@pytest.mark.langsmith
async def test_sop_query_integration():
    """Test SOP query through the main graph."""
    # Mock the SOP knowledge base functions
    with patch('agent.sop_agent.query_sop_knowledge_base') as mock_query, \
         patch('agent.sop_agent.get_sop_recommendations') as mock_recommendations:
        
        mock_query.return_value = "user_onboarding.md"
        mock_recommendations.return_value = [
            {
                "filename": "user_onboarding.md",
                "title": "User Onboarding SOP",
                "summary": "Standard procedure for onboarding new users",
                "keywords": ["onboarding", "users", "procedure"]
            }
        ]
        
        inputs = {"input": "I need the SOP for user onboarding"}
        result = await graph.ainvoke(inputs)
        
        assert result is not None
        assert "output" in result
        assert "user_onboarding.md" in result["output"]


@pytest.mark.langsmith
async def test_build_knowledge_base_integration():
    """Test building knowledge base through the main graph."""
    with patch('agent.sop_agent.build_sop_knowledge_base') as mock_build:
        mock_build.return_value = {
            "processed": 5,
            "updated": 0,
            "skipped": 0,
            "errors": 0
        }
        
        inputs = {"input": "build knowledge base from \"/path/to/sop/files\""}
        result = await graph.ainvoke(inputs)
        
        assert result is not None
        assert "output" in result
        assert "Knowledge base built successfully" in result["output"]
        mock_build.assert_called_once_with("/path/to/sop/files")


@pytest.mark.langsmith
async def test_sql_route_still_works():
    """Test that SQL routing still works after SOP agent integration."""
    inputs = {"input": "Show me all customers from last month"}
    result = await graph.ainvoke(inputs)
    
    assert result is not None
    assert "output" in result


@pytest.mark.langsmith
async def test_answer_route_still_works():
    """Test that answer routing still works after SOP agent integration."""
    inputs = {"input": "What is the difference between REST and GraphQL?"}
    result = await graph.ainvoke(inputs)
    
    assert result is not None
    assert "output" in result
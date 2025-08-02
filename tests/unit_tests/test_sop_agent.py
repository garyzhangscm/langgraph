"""Tests for SOP Knowledge Base Agent."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from agent.sop_agent import (
    SOPKnowledgeBase, 
    SOPDocument,
    build_sop_knowledge_base,
    query_sop_knowledge_base,
    get_sop_recommendations
)


@pytest.fixture
def temp_sop_folder():
    """Create a temporary folder with sample SOP files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample SOP files
        sop1_content = """
        # User Onboarding SOP
        
        This document describes the standard operating procedure for user onboarding.
        
        ## Steps:
        1. Create user account
        2. Send welcome email
        3. Assign initial permissions
        4. Schedule orientation
        """
        
        sop2_content = """
        # Data Backup Procedure
        
        This SOP outlines the data backup process.
        
        ## Requirements:
        - Daily backups at 2 AM
        - Weekly full system backup
        - Monthly offsite backup verification
        """
        
        # Write files
        with open(Path(temp_dir) / "user_onboarding.md", "w") as f:
            f.write(sop1_content)
            
        with open(Path(temp_dir) / "data_backup.txt", "w") as f:
            f.write(sop2_content)
        
        # Create a simple test PDF content (mock PDF for testing)
        pdf_content = """
        # Password Security SOP
        
        This document outlines password security requirements.
        
        ## Requirements:
        - Minimum 12 characters
        - Mix of uppercase, lowercase, numbers, symbols
        - Change every 90 days
        - No reuse of last 12 passwords
        """
        
        with open(Path(temp_dir) / "password_security.txt", "w") as f:
            f.write(pdf_content)  # Using .txt for testing since we can't easily create real PDFs in tests
            
        yield temp_dir


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing."""
    mock_embeddings = Mock()
    mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3] for _ in range(2)]
    mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock_embeddings


@pytest.fixture
def temp_kb_path():
    """Create temporary knowledge base path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class TestSOPDocument:
    """Test SOPDocument dataclass."""
    
    def test_sop_document_creation(self):
        doc = SOPDocument(
            filename="test.md",
            title="Test SOP",
            content="Test content",
            summary="Test summary",
            keywords=["test", "sop"],
            file_path="/path/to/test.md",
            last_modified="2024-01-01T10:00:00",
            file_hash="abc123"
        )
        
        assert doc.filename == "test.md"
        assert doc.title == "Test SOP"
        assert len(doc.keywords) == 2


class TestSOPKnowledgeBase:
    """Test SOPKnowledgeBase class."""
    
    @patch('agent.sop_agent.ChatAnthropic')
    @patch('agent.sop_agent.HuggingFaceEmbeddings')
    def test_init(self, mock_embeddings_class, mock_llm_class, temp_kb_path):
        """Test knowledge base initialization."""
        mock_embeddings_class.return_value = Mock()
        mock_llm_class.return_value = Mock()
        
        kb = SOPKnowledgeBase(kb_path=temp_kb_path)
        
        assert kb.kb_path == Path(temp_kb_path)
        assert kb.documents == {}
        assert kb.vector_store is None
    
    @patch('agent.sop_agent.ChatAnthropic')
    def test_extract_file_content(self, mock_llm_class, temp_kb_path):
        """Test file content extraction."""
        mock_llm_class.return_value = Mock()
        
        kb = SOPKnowledgeBase(kb_path=temp_kb_path)
        
        # Create test file
        test_file = Path(temp_kb_path) / "test.txt"
        test_content = "This is test content"
        test_file.write_text(test_content)
        
        content = kb._extract_file_content(test_file)
        assert content == test_content
    
    @patch('agent.sop_agent.ChatAnthropic')
    def test_generate_file_hash(self, mock_llm_class, temp_kb_path):
        """Test file hash generation."""
        mock_llm_class.return_value = Mock()
        
        kb = SOPKnowledgeBase(kb_path=temp_kb_path)
        
        # Create test file
        test_file = Path(temp_kb_path) / "test.txt"
        test_file.write_text("test content")
        
        hash1 = kb._generate_file_hash(test_file)
        hash2 = kb._generate_file_hash(test_file)
        
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length
    
    @patch('agent.sop_agent.ChatAnthropic')
    @patch('agent.sop_agent.PDF_AVAILABLE', True)
    def test_extract_pdf_content_no_pdf_lib(self, mock_llm_class, temp_kb_path):
        """Test PDF content extraction when PDF libraries are not available."""
        mock_llm_class.return_value = Mock()
        
        kb = SOPKnowledgeBase(kb_path=temp_kb_path)
        
        # Create mock PDF file
        test_file = Path(temp_kb_path) / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 mock pdf content")
        
        # Test when PDF_AVAILABLE is False
        with patch('agent.sop_agent.PDF_AVAILABLE', False):
            content = kb._extract_pdf_content(test_file)
            assert content == ""
    
    @patch('agent.sop_agent.ChatAnthropic')
    def test_extract_file_content_pdf_extension(self, mock_llm_class, temp_kb_path):
        """Test that PDF files are routed to PDF extraction method."""
        mock_llm_class.return_value = Mock()
        
        kb = SOPKnowledgeBase(kb_path=temp_kb_path)
        
        # Create mock PDF file
        test_file = Path(temp_kb_path) / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 mock pdf content")
        
        # Mock the PDF extraction method
        with patch.object(kb, '_extract_pdf_content', return_value="Extracted PDF content") as mock_pdf_extract:
            content = kb._extract_file_content(test_file)
            mock_pdf_extract.assert_called_once_with(test_file)
            assert content == "Extracted PDF content"
    
    @patch('agent.sop_agent.ChatAnthropic')
    @patch('agent.sop_agent.HuggingFaceEmbeddings')
    def test_extract_metadata_with_llm(self, mock_embeddings_class, mock_llm_class, temp_kb_path):
        """Test metadata extraction with LLM."""
        mock_embeddings_class.return_value = Mock()
        
        # Mock LLM response
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = '''
        {
            "title": "Test SOP",
            "summary": "This is a test SOP document",
            "keywords": ["test", "sop", "document"]
        }
        '''
        mock_llm.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm
        
        kb = SOPKnowledgeBase(kb_path=temp_kb_path)
        
        metadata = kb._extract_metadata_with_llm("test content", "test.md")
        
        assert metadata["title"] == "Test SOP"
        assert metadata["summary"] == "This is a test SOP document"
        assert len(metadata["keywords"]) == 3
    
    @patch('agent.sop_agent.ChatAnthropic')
    @patch('agent.sop_agent.HuggingFaceEmbeddings')
    @patch('agent.sop_agent.FAISS')
    def test_build_knowledge_base(self, mock_faiss, mock_embeddings_class, mock_llm_class, temp_sop_folder, temp_kb_path):
        """Test building knowledge base from folder."""
        # Setup mocks
        mock_embeddings = Mock()
        mock_embeddings_class.return_value = mock_embeddings
        
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = '''
        {
            "title": "Test SOP",
            "summary": "Test summary",
            "keywords": ["test", "sop"]
        }
        '''
        mock_llm.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm
        
        mock_vector_store = Mock()
        mock_faiss.from_documents.return_value = mock_vector_store
        
        kb = SOPKnowledgeBase(kb_path=temp_kb_path)
        
        results = kb.build_knowledge_base(temp_sop_folder)
        
        assert results["processed"] >= 0
        assert results["errors"] >= 0
        assert "files" in results
    
    @patch('agent.sop_agent.ChatAnthropic')
    @patch('agent.sop_agent.HuggingFaceEmbeddings')
    def test_query_sop_no_vector_store(self, mock_embeddings_class, mock_llm_class, temp_kb_path):
        """Test querying SOP when no vector store exists."""
        mock_embeddings_class.return_value = Mock()
        mock_llm_class.return_value = Mock()
        
        kb = SOPKnowledgeBase(kb_path=temp_kb_path)
        
        results = kb.query_sop("test query")
        assert results == []
    
    @patch('agent.sop_agent.ChatAnthropic')
    @patch('agent.sop_agent.HuggingFaceEmbeddings')
    def test_get_stats(self, mock_embeddings_class, mock_llm_class, temp_kb_path):
        """Test getting knowledge base statistics."""
        mock_embeddings_class.return_value = Mock()
        mock_llm_class.return_value = Mock()
        
        kb = SOPKnowledgeBase(kb_path=temp_kb_path)
        
        stats = kb.get_stats()
        
        assert "total_documents" in stats
        assert "has_vector_store" in stats
        assert "kb_path" in stats
        assert "last_updated" in stats


class TestSOPAgentFunctions:
    """Test module-level functions."""
    
    @patch('agent.sop_agent.sop_kb')
    def test_build_sop_knowledge_base(self, mock_kb):
        """Test build_sop_knowledge_base function."""
        mock_results = {"processed": 2, "errors": 0}
        mock_kb.build_knowledge_base.return_value = mock_results
        
        results = build_sop_knowledge_base("/test/path")
        
        mock_kb.build_knowledge_base.assert_called_once_with("/test/path")
        assert results == mock_results
    
    @patch('agent.sop_agent.sop_kb')
    def test_query_sop_knowledge_base(self, mock_kb):
        """Test query_sop_knowledge_base function."""
        mock_kb.get_best_sop_filename.return_value = "test_sop.md"
        
        result = query_sop_knowledge_base("test query")
        
        mock_kb.get_best_sop_filename.assert_called_once_with("test query")
        assert result == "test_sop.md"
    
    @patch('agent.sop_agent.sop_kb')
    def test_get_sop_recommendations(self, mock_kb):
        """Test get_sop_recommendations function."""
        mock_recommendations = [
            {"filename": "sop1.md", "title": "SOP 1"},
            {"filename": "sop2.md", "title": "SOP 2"}
        ]
        mock_kb.query_sop.return_value = mock_recommendations
        
        results = get_sop_recommendations("test query", top_k=2)
        
        mock_kb.query_sop.assert_called_once_with("test query", 2)
        assert results == mock_recommendations


if __name__ == "__main__":
    pytest.main([__file__])
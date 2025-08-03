"""Tests for SOP file download functionality."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from agent.sop_agent import (
    get_sop_file_info,
    prepare_file_download,
    copy_sop_file_to_downloads
)


@pytest.fixture
def temp_sop_file():
    """Create a temporary SOP file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("# Test SOP\n\nThis is a test SOP document for download testing.\n")
        f.flush()
        yield Path(f.name)
        Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_downloads_dir():
    """Create a temporary downloads directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class TestFileDownloadFunctionality:
    """Test file download related functions."""
    
    @patch('agent.sop_agent.sop_kb')
    def test_get_sop_file_info_from_kb(self, mock_kb, temp_sop_file):
        """Test getting file info from knowledge base."""
        # Mock knowledge base document
        mock_doc = Mock()
        mock_doc.title = "Test SOP Document"
        mock_doc.file_path = str(temp_sop_file)
        mock_doc.last_modified = "2024-01-01T10:00:00"
        
        mock_kb.documents = {"test.md": mock_doc}
        
        file_info = get_sop_file_info("test.md")
        
        assert file_info is not None
        assert file_info["filename"] == "test.md"
        assert file_info["title"] == "Test SOP Document"
        assert file_info["available"] is True
        assert file_info["file_extension"] == temp_sop_file.suffix
    
    @patch('agent.sop_agent.sop_kb')
    @patch('agent.sop_agent.get_sop_source_folder')
    def test_get_sop_file_info_from_folder(self, mock_get_folder, mock_kb, temp_sop_file):
        """Test getting file info from SOP folder when not in KB."""
        mock_kb.documents = {}  # Empty knowledge base
        mock_get_folder.return_value = str(temp_sop_file.parent)
        
        file_info = get_sop_file_info(temp_sop_file.name)
        
        assert file_info is not None
        assert file_info["filename"] == temp_sop_file.name
        assert file_info["available"] is True
        assert file_info["file_extension"] == temp_sop_file.suffix
    
    def test_get_sop_file_info_not_found(self):
        """Test getting file info for non-existent file."""
        with patch('agent.sop_agent.sop_kb') as mock_kb, \
             patch('agent.sop_agent.get_sop_source_folder') as mock_get_folder:
            
            mock_kb.documents = {}
            mock_get_folder.return_value = "/nonexistent/path"
            
            file_info = get_sop_file_info("nonexistent.md")
            
            assert file_info is None
    
    @patch('agent.sop_agent.get_sop_file_info')
    def test_prepare_file_download_small_file(self, mock_get_info, temp_sop_file):
        """Test preparing a small file for download (base64 encoding)."""
        mock_get_info.return_value = {
            "filename": "test.md",
            "title": "Test SOP",
            "file_path": str(temp_sop_file),
            "file_size": 100,  # Small file
            "file_size_mb": 0.0001,
            "file_extension": ".md"
        }
        
        result = prepare_file_download("test.md")
        
        assert result["success"] is True
        assert result["download_method"] == "base64_encoded"
        assert "encoded_content" in result
        assert result["filename"] == "test.md"
    
    @patch('agent.sop_agent.get_sop_file_info')
    def test_prepare_file_download_large_file(self, mock_get_info, temp_sop_file):
        """Test preparing a large file for download (file path method)."""
        mock_get_info.return_value = {
            "filename": "large_test.pdf",
            "title": "Large Test SOP",
            "file_path": str(temp_sop_file),
            "file_size": 2 * 1024 * 1024,  # 2MB file
            "file_size_mb": 2.0,
            "file_extension": ".pdf"
        }
        
        result = prepare_file_download("large_test.pdf")
        
        assert result["success"] is True
        assert result["download_method"] == "file_path"
        assert "file_path" in result
        assert result["filename"] == "large_test.pdf"
    
    @patch('agent.sop_agent.get_sop_file_info')
    def test_prepare_file_download_not_found(self, mock_get_info):
        """Test preparing non-existent file for download."""
        mock_get_info.return_value = None
        
        result = prepare_file_download("nonexistent.md")
        
        assert result["success"] is False
        assert "error" in result
        assert "not found" in result["error"].lower()
    
    @patch('agent.sop_agent.get_sop_file_info')
    def test_copy_sop_file_to_downloads(self, mock_get_info, temp_sop_file, temp_downloads_dir):
        """Test copying SOP file to downloads folder."""
        mock_get_info.return_value = {
            "filename": "test.md",
            "file_path": str(temp_sop_file),
            "file_size_mb": 0.001
        }
        
        result = copy_sop_file_to_downloads("test.md", temp_downloads_dir)
        
        assert result["success"] is True
        assert result["filename"] == "test.md"
        assert "destination_path" in result
        
        # Verify file was actually copied
        dest_path = Path(result["destination_path"])
        assert dest_path.exists()
        assert dest_path.name == "test.md"
    
    @patch('agent.sop_agent.get_sop_file_info')
    def test_copy_sop_file_not_found(self, mock_get_info, temp_downloads_dir):
        """Test copying non-existent file."""
        mock_get_info.return_value = None
        
        result = copy_sop_file_to_downloads("nonexistent.md", temp_downloads_dir)
        
        assert result["success"] is False
        assert "error" in result
        assert "not found" in result["error"].lower()


if __name__ == "__main__":
    pytest.main([__file__])
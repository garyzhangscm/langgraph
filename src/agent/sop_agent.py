"""SOP Knowledge Base Agent.

This module provides functionality to build and query a knowledge base of SOP files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
import shutil
import base64
from urllib.parse import quote
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStore
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

# PDF processing imports
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    try:
        import pypdf
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False
        print("Warning: PDF support not available. Install PyMuPDF or pypdf: pip install PyMuPDF")


@dataclass
class SOPDocument:
    """Represents an SOP document in the knowledge base."""
    filename: str
    title: str
    content: str
    summary: str
    keywords: List[str]
    file_path: str
    last_modified: str
    file_hash: str


class SOPKnowledgeBase:
    """Knowledge base for SOP documents with embedding-based search."""
    
    def __init__(
        self, 
        kb_path: Optional[str] = None,
        embeddings: Optional[Embeddings] = None
    ):
        # Use environment variable or default
        if kb_path is None:
            kb_path = get_sop_knowledge_base_path()
        
        self.kb_path = Path(kb_path)
        self.kb_path.mkdir(exist_ok=True)
        
        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.vector_store: Optional[VectorStore] = None
        self.documents: Dict[str, SOPDocument] = {}
        self.llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
        
        self._load_knowledge_base()
    
    def _load_knowledge_base(self) -> None:
        """Load existing knowledge base from disk."""
        metadata_file = self.kb_path / "metadata.json"
        vector_store_path = self.kb_path / "vector_store"
        
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.documents = {
                    k: SOPDocument(**v) for k, v in metadata.items()
                }
        
        if vector_store_path.exists():
            try:
                self.vector_store = FAISS.load_local(
                    str(vector_store_path), 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Warning: Could not load vector store: {e}")
                self.vector_store = None
    
    def _save_knowledge_base(self) -> None:
        """Save knowledge base to disk."""
        metadata_file = self.kb_path / "metadata.json"
        vector_store_path = self.kb_path / "vector_store"
        
        # Save metadata
        metadata = {k: asdict(v) for k, v in self.documents.items()}
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Save vector store
        if self.vector_store:
            self.vector_store.save_local(str(vector_store_path))
    
    def _extract_file_content(self, file_path: Path) -> str:
        """Extract content from various file formats."""
        file_extension = file_path.suffix.lower()
        
        # Handle PDF files
        if file_extension == '.pdf':
            return self._extract_pdf_content(file_path)
        
        # Handle text-based files
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                return ""
    
    def _extract_pdf_content(self, file_path: Path) -> str:
        """Extract text content from PDF files."""
        if not PDF_AVAILABLE:
            print(f"PDF support not available. Skipping {file_path}")
            return ""
        
        try:
            # Try PyMuPDF first (more reliable)
            if 'fitz' in globals():
                return self._extract_pdf_with_pymupdf(file_path)
            else:
                return self._extract_pdf_with_pypdf(file_path)
        except Exception as e:
            print(f"Error extracting PDF content from {file_path}: {e}")
            return ""
    
    def _extract_pdf_with_pymupdf(self, file_path: Path) -> str:
        """Extract PDF content using PyMuPDF."""
        import fitz
        
        text_content = ""
        doc = fitz.open(str(file_path))
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_content += page.get_text()
            text_content += "\n\n"  # Add page breaks
        
        doc.close()
        return text_content.strip()
    
    def _extract_pdf_with_pypdf(self, file_path: Path) -> str:
        """Extract PDF content using pypdf."""
        import pypdf
        
        text_content = ""
        
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text()
                text_content += "\n\n"  # Add page breaks
        
        return text_content.strip()
    
    def _generate_file_hash(self, file_path: Path) -> str:
        """Generate hash for file to detect changes."""
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return ""
    
    def _extract_metadata_with_llm(self, content: str, filename: str) -> Dict[str, Any]:
        """Use LLM to extract title, summary, and keywords from SOP content."""
        prompt = f"""
        Analyze the following SOP document and extract metadata:
        
        Filename: {filename}
        Content: {content[:2000]}...
        
        Please provide:
        1. A clear, descriptive title for this SOP
        2. A concise summary (2-3 sentences)
        3. 5-10 relevant keywords/tags
        
        Respond in this exact JSON format:
        {{
            "title": "Title here",
            "summary": "Summary here",
            "keywords": ["keyword1", "keyword2", "keyword3"]
        }}
        """
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are an expert at analyzing SOP documents and extracting metadata. Always respond with valid JSON."),
                HumanMessage(content=prompt)
            ])
            
            # Extract JSON from response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            
            return json.loads(content)
        except Exception as e:
            print(f"Error extracting metadata with LLM: {e}")
            return {
                "title": filename,
                "summary": "No summary available",
                "keywords": ["sop", "procedure"]
            }
    
    def build_knowledge_base(self, sop_folder_path: str) -> Dict[str, Any]:
        """Build knowledge base from a folder of SOP files."""
        sop_folder = Path(sop_folder_path)
        if not sop_folder.exists():
            raise FileNotFoundError(f"SOP folder not found: {sop_folder_path}")
        
        results = {
            "processed": 0,
            "updated": 0,
            "skipped": 0,
            "errors": 0,
            "files": []
        }
        
        # Supported file extensions
        supported_extensions = {'.txt', '.md', '.rst', '.doc', '.docx', '.pdf'}
        
        # Find all SOP files
        sop_files = []
        for ext in supported_extensions:
            sop_files.extend(sop_folder.rglob(f"*{ext}"))
        
        documents_to_embed = []
        
        for file_path in sop_files:
            try:
                # Check if file needs processing
                file_hash = self._generate_file_hash(file_path)
                filename = file_path.name
                
                if (filename in self.documents and 
                    self.documents[filename].file_hash == file_hash):
                    results["skipped"] += 1
                    continue
                
                # Extract content
                content = self._extract_file_content(file_path)
                if not content.strip():
                    results["errors"] += 1
                    continue
                
                # Extract metadata using LLM
                metadata = self._extract_metadata_with_llm(content, filename)
                
                # Create SOP document
                sop_doc = SOPDocument(
                    filename=filename,
                    title=metadata["title"],
                    content=content,
                    summary=metadata["summary"],
                    keywords=metadata["keywords"],
                    file_path=str(file_path),
                    last_modified=datetime.fromtimestamp(
                        file_path.stat().st_mtime
                    ).isoformat(),
                    file_hash=file_hash
                )
                
                # Store document
                self.documents[filename] = sop_doc
                
                # Prepare for embedding
                doc_text = f"Title: {sop_doc.title}\nSummary: {sop_doc.summary}\nKeywords: {', '.join(sop_doc.keywords)}\nContent: {content}"
                documents_to_embed.append(
                    Document(
                        page_content=doc_text,
                        metadata={
                            "filename": filename,
                            "title": sop_doc.title,
                            "keywords": sop_doc.keywords
                        }
                    )
                )
                
                if filename in self.documents:
                    results["updated"] += 1
                else:
                    results["processed"] += 1
                    
                results["files"].append({
                    "filename": filename,
                    "title": sop_doc.title,
                    "status": "processed"
                })
                
            except Exception as e:
                results["errors"] += 1
                results["files"].append({
                    "filename": file_path.name,
                    "status": "error",
                    "error": str(e)
                })
                print(f"Error processing {file_path}: {e}")
        
        # Build vector store
        if documents_to_embed:
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(
                    documents_to_embed, self.embeddings
                )
            else:
                # Add new documents to existing vector store
                new_vector_store = FAISS.from_documents(
                    documents_to_embed, self.embeddings
                )
                self.vector_store.merge_from(new_vector_store)
        
        # Save knowledge base
        self._save_knowledge_base()
        
        return results
    
    def query_sop(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the knowledge base for relevant SOP files."""
        if not self.vector_store:
            return []
        
        try:
            # Search for similar documents
            docs = self.vector_store.similarity_search(query, k=top_k)
            
            results = []
            for doc in docs:
                filename = doc.metadata.get("filename", "unknown")
                if filename in self.documents:
                    sop_doc = self.documents[filename]
                    download_info = create_download_link(filename)
                    
                    results.append({
                        "filename": filename,
                        "title": sop_doc.title,
                        "summary": sop_doc.summary,
                        "keywords": sop_doc.keywords,
                        "file_path": sop_doc.file_path,
                        "relevance_score": "high",  # Could implement actual scoring
                        "download_link": download_info["download_link"],
                        "download_text": download_info["download_text"]
                    })
            
            return results
            
        except Exception as e:
            print(f"Error querying SOP knowledge base: {e}")
            return []
    
    def get_best_sop_filename(self, query: str) -> Optional[str]:
        """Get the most relevant SOP filename for a query."""
        results = self.query_sop(query, top_k=1)
        if results:
            return results[0]["filename"]
        return None
    
    def list_all_sops(self) -> List[Dict[str, Any]]:
        """List all SOPs in the knowledge base."""
        results = []
        for doc in self.documents.values():
            download_info = create_download_link(doc.filename)
            results.append({
                "filename": doc.filename,
                "title": doc.title,
                "summary": doc.summary,
                "keywords": doc.keywords,
                "last_modified": doc.last_modified,
                "download_link": download_info["download_link"],
                "download_text": download_info["download_text"]
            })
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            "total_documents": len(self.documents),
            "has_vector_store": self.vector_store is not None,
            "kb_path": str(self.kb_path),
            "last_updated": datetime.now().isoformat()
        }


# Configuration helper functions
def get_sop_source_folder() -> str:
    """Get the configured SOP source folder path."""
    return os.getenv("SOP_SOURCE_FOLDER", "static/data/sop")

def get_sop_knowledge_base_path() -> str:
    """Get the configured SOP knowledge base path."""
    return os.getenv("SOP_KNOWLEDGE_BASE_PATH", "static/data/vector_store/sop_knowledge_base")


def get_sop_download_endpoint() -> str:
    """Get the configured SOP download endpoint."""
    return os.getenv("SOP_DOWNLOAD_ENDPOINT", "http://localhost:8000/download")


def create_download_link(filename: str) -> Dict[str, str]:
    """Create a proper download link for a filename, handling spaces and special characters."""
    download_endpoint = get_sop_download_endpoint()
    # URL encode the filename to handle spaces and special characters
    encoded_filename = quote(filename)
    download_url = f"{download_endpoint}/{encoded_filename}"
    
    return {
        "download_link": download_url,
        "download_text": "📁 Download",
        "filename": filename
    }


# Global knowledge base instance
sop_kb = SOPKnowledgeBase()


def build_sop_knowledge_base(folder_path: Optional[str] = None) -> Dict[str, Any]:
    """Build the SOP knowledge base from a folder of files.
    
    Args:
        folder_path: Path to SOP files. If None, uses SOP_SOURCE_FOLDER env var.
    """
    if folder_path is None:
        folder_path = get_sop_source_folder()
    
    return sop_kb.build_knowledge_base(folder_path)


def query_sop_knowledge_base(query: str) -> Optional[str]:
    """Query the SOP knowledge base and return the best matching filename."""
    return sop_kb.get_best_sop_filename(query)


def get_sop_recommendations(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Get multiple SOP recommendations for a query."""
    return sop_kb.query_sop(query, top_k)


def list_all_sops() -> List[Dict[str, Any]]:
    """Get a list of all SOPs in the knowledge base."""
    return sop_kb.list_all_sops()


# File download functionality
def get_sop_file_info(filename: str) -> Optional[Dict[str, Any]]:
    """Get information about an SOP file for download."""
    if filename in sop_kb.documents:
        sop_doc = sop_kb.documents[filename]
        file_path = Path(sop_doc.file_path)
        
        if file_path.exists():
            file_size = file_path.stat().st_size
            download_info = create_download_link(filename)
            
            return {
                "filename": filename,
                "title": sop_doc.title,
                "file_path": str(file_path),
                "file_size": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "last_modified": sop_doc.last_modified,
                "file_extension": file_path.suffix,
                "available": True,
                "download_link": download_info["download_link"],
                "download_text": download_info["download_text"]
            }
    
    # If not in knowledge base, try to find in SOP folder
    sop_folder = Path(get_sop_source_folder())
    possible_files = list(sop_folder.glob(f"**/{filename}"))
    
    if possible_files:
        file_path = possible_files[0]
        file_size = file_path.stat().st_size
        return {
            "filename": filename,
            "title": filename,  # Use filename as title if not in KB
            "file_path": str(file_path),
            "file_size": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            "file_extension": file_path.suffix,
            "available": True
        }
    
    return None


def prepare_file_download(filename: str) -> Dict[str, Any]:
    """Prepare a file for download by encoding it or providing download info."""
    file_info = get_sop_file_info(filename)
    
    if not file_info:
        return {
            "success": False,
            "error": f"File '{filename}' not found in SOP collection",
            "available_files": list(sop_kb.documents.keys())[:10]  # Show first 10 files
        }
    
    file_path = Path(file_info["file_path"])
    
    try:
        # For small files, we can encode them directly
        if file_info["file_size"] < 1024 * 1024:  # Less than 1MB
            with open(file_path, 'rb') as f:
                file_content = f.read()
                encoded_content = base64.b64encode(file_content).decode('utf-8')
            
            return {
                "success": True,
                "filename": filename,
                "title": file_info["title"],
                "file_size": file_info["file_size"],
                "file_size_mb": file_info["file_size_mb"],
                "file_extension": file_info["file_extension"],
                "encoded_content": encoded_content,
                "download_method": "base64_encoded",
                "download_link": file_info.get("download_link", ""),
                "download_text": file_info.get("download_text", "📁 Download"),
                "instructions": f"File '{filename}' is ready for download. The content is base64 encoded."
            }
        else:
            # For larger files, provide file path and download instructions
            return {
                "success": True,
                "filename": filename,
                "title": file_info["title"],
                "file_size": file_info["file_size"],
                "file_size_mb": file_info["file_size_mb"],
                "file_extension": file_info["file_extension"],
                "file_path": str(file_path),
                "download_method": "file_path",
                "download_link": file_info.get("download_link", ""),
                "download_text": file_info.get("download_text", "📁 Download"),
                "instructions": f"File '{filename}' ({file_info['file_size_mb']} MB) is available at: {file_path}"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Error preparing file '{filename}' for download: {str(e)}",
            "file_info": file_info
        }


def copy_sop_file_to_downloads(filename: str, downloads_folder: str = "downloads") -> Dict[str, Any]:
    """Copy an SOP file to a downloads folder for easy access."""
    file_info = get_sop_file_info(filename)
    
    if not file_info:
        return {
            "success": False,
            "error": f"File '{filename}' not found"
        }
    
    try:
        # Create downloads folder if it doesn't exist
        downloads_path = Path(downloads_folder)
        downloads_path.mkdir(exist_ok=True)
        
        source_path = Path(file_info["file_path"])
        dest_path = downloads_path / filename
        
        # Copy the file
        shutil.copy2(source_path, dest_path)
        
        return {
            "success": True,
            "filename": filename,
            "source_path": str(source_path),
            "destination_path": str(dest_path),
            "file_size_mb": file_info["file_size_mb"],
            "download_link": file_info.get("download_link", ""),
            "download_text": file_info.get("download_text", "📁 Download"),
            "message": f"File '{filename}' copied to {dest_path}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error copying file '{filename}': {str(e)}"
        }
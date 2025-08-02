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

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
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
        kb_path: str = "sop_knowledge_base",
        embeddings: Optional[Embeddings] = None
    ):
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
                    results.append({
                        "filename": filename,
                        "title": sop_doc.title,
                        "summary": sop_doc.summary,
                        "keywords": sop_doc.keywords,
                        "file_path": sop_doc.file_path,
                        "relevance_score": "high"  # Could implement actual scoring
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
        return [
            {
                "filename": doc.filename,
                "title": doc.title,
                "summary": doc.summary,
                "keywords": doc.keywords,
                "last_modified": doc.last_modified
            }
            for doc in self.documents.values()
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            "total_documents": len(self.documents),
            "has_vector_store": self.vector_store is not None,
            "kb_path": str(self.kb_path),
            "last_updated": datetime.now().isoformat()
        }


# Global knowledge base instance
sop_kb = SOPKnowledgeBase()


def build_sop_knowledge_base(folder_path: str) -> Dict[str, Any]:
    """Build the SOP knowledge base from a folder of files."""
    return sop_kb.build_knowledge_base(folder_path)


def query_sop_knowledge_base(query: str) -> Optional[str]:
    """Query the SOP knowledge base and return the best matching filename."""
    return sop_kb.get_best_sop_filename(query)


def get_sop_recommendations(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Get multiple SOP recommendations for a query."""
    return sop_kb.query_sop(query, top_k)
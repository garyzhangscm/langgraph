"""SQL Schema Knowledge Base Agent.

This module provides functionality to build and query a knowledge base of database schema files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
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
from langchain_openai import OpenAIEmbeddings


@dataclass
class SchemaTable:
    """Represents a database table schema in the knowledge base."""
    table_name: str
    schema_name: str
    description: str
    columns: List[Dict[str, Any]]
    relationships: List[str]
    file_path: str
    last_modified: str
    file_hash: str


class SQLSchemaKnowledgeBase:
    """Knowledge base for database schema documents with embedding-based search."""
    
    def __init__(
        self, 
        kb_path: Optional[str] = None,
        embeddings: Optional[Embeddings] = None
    ):
        # Use environment variable or default
        if kb_path is None:
            kb_path = get_sql_knowledge_base_path()
        
        self.kb_path = Path(kb_path)
        self.kb_path.mkdir(exist_ok=True)
        
        # self.embeddings = embeddings or HuggingFaceEmbeddings(
        #    model_name="all-MiniLM-L6-v2"
        # )
        self.embeddings = embeddings or OpenAIEmbeddings(
                                            model="text-embedding-3-large",
                                            # With the `text-embedding-3` class
                                            # of models, you can specify the size
                                            # of the embeddings you want returned.
                                            # dimensions=1024
                                        )
        self.vector_store: Optional[VectorStore] = None
        self.schemas: Dict[str, SchemaTable] = {}
        self.llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
        
        self._load_knowledge_base()
    
    def _load_knowledge_base(self) -> None:
        """Load existing knowledge base from disk."""
        metadata_file = self.kb_path / "metadata.json"
        vector_store_path = self.kb_path / "vector_store"
        
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.schemas = {
                    k: SchemaTable(**v) for k, v in metadata.items()
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
        metadata = {k: asdict(v) for k, v in self.schemas.items()}
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Save vector store
        if self.vector_store:
            self.vector_store.save_local(str(vector_store_path))
    
    def _parse_schema_file(self, file_path: Path) -> Optional[SchemaTable]:
        """Parse a database schema file and extract table information."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                return None
            
            lines = content.split('\n')
            if not lines:
                return None
            
            # Parse table name and description from first line
            first_line = lines[0].strip()
            if not first_line.startswith('Table:'):
                return None
            
            # Extract table name and description
            table_info = first_line[6:].strip()  # Remove "Table:"
            if ',' in table_info:
                table_full_name, description = table_info.split(',', 1)
                description = description.strip()
            else:
                table_full_name = table_info
                description = ""
            
            # Extract schema and table name
            if '.' in table_full_name:
                schema_name, table_name = table_full_name.split('.', 1)
                schema_name = schema_name.strip()
                table_name = table_name.strip()
            else:
                schema_name = "dbo"
                table_name = table_full_name.strip()
            
            # Parse columns
            columns = []
            in_columns_section = False
            
            for line in lines[1:]:
                line = line.strip()
                if line.startswith('Columns:'):
                    in_columns_section = True
                    continue
                
                if in_columns_section and line.startswith('-'):
                    # Parse column definition
                    column_def = line[1:].strip()  # Remove '-'
                    
                    if ':' in column_def:
                        column_name, column_info = column_def.split(':', 1)
                        column_name = column_name.strip()
                        column_info = column_info.strip()
                        
                        # Extract data type and nullability
                        data_type = ""
                        nullable = True
                        description_text = column_info
                        
                        # Look for type info in parentheses
                        if '(' in column_info and ')' in column_info:
                            type_start = column_info.rfind('(')
                            type_end = column_info.rfind(')')
                            if type_start < type_end:
                                type_info = column_info[type_start+1:type_end]
                                description_text = column_info[:type_start].strip()
                                
                                # Parse type and nullability
                                type_parts = [part.strip() for part in type_info.split(',')]
                                if type_parts:
                                    data_type = type_parts[0]
                                if len(type_parts) > 1:
                                    nullable = "Not Nullable" not in type_parts[1]
                        
                        columns.append({
                            "name": column_name,
                            "data_type": data_type,
                            "nullable": nullable,
                            "description": description_text
                        })
            
            # Generate file hash
            file_hash = self._generate_file_hash(file_path)
            
            return SchemaTable(
                table_name=table_name,
                schema_name=schema_name,
                description=description,
                columns=columns,
                relationships=[],  # Could be enhanced to parse relationships
                file_path=str(file_path),
                last_modified=datetime.fromtimestamp(
                    file_path.stat().st_mtime
                ).isoformat(),
                file_hash=file_hash
            )
            
        except Exception as e:
            print(f"Error parsing schema file {file_path}: {e}")
            return None
    
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
    
    def _enhance_schema_with_llm(self, schema: SchemaTable, enable_llm_enhancement: bool = False) -> SchemaTable:
        """Use LLM to enhance schema information with relationships and insights."""
        if not enable_llm_enhancement:
            # Skip LLM enhancement to avoid API costs
            # Add basic relationship detection based on column names
            relationships = []
            for col in schema.columns:
                col_name = col['name'].lower()
                if col_name.endswith('_id') and col_name != 'id':
                    # Likely foreign key
                    related_table = col_name[:-3]  # Remove '_id'
                    relationships.append(f"{related_table} (via {col['name']})")
            
            schema.relationships = relationships
            return schema
        
        # Create a description of the table and columns
        columns_text = "\n".join([
            f"- {col['name']} ({col['data_type']}): {col['description']}"
            for col in schema.columns
        ])
        
        prompt = f"""
        Analyze this database table schema and provide insights:
        
        Table: {schema.schema_name}.{schema.table_name}
        Description: {schema.description}
        
        Columns:
        {columns_text}
        
        Please provide:
        1. Potential relationships with other tables (look for foreign key patterns like *_id columns)
        2. Business purpose and use cases for this table
        3. Important patterns or constraints you notice
        
        Respond in this exact JSON format:
        {{
            "relationships": ["table1 (via column_name)", "table2 (via other_column)"],
            "business_purpose": "Brief description of business purpose",
            "insights": ["insight1", "insight2", "insight3"]
        }}
        """
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a database expert analyzing table schemas. Always respond with valid JSON."),
                HumanMessage(content=prompt)
            ])
            
            # Extract JSON from response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            
            enhancement = json.loads(content)
            
            # Update schema with enhancements
            schema.relationships = enhancement.get("relationships", [])
            
            return schema
            
        except Exception as e:
            print(f"Error enhancing schema with LLM: {e}")
            return schema
    
    def build_knowledge_base(self, schema_folder_path: str) -> Dict[str, Any]:
        """Build knowledge base from a folder of database schema files."""
        schema_folder = Path(schema_folder_path)
        if not schema_folder.exists():
            raise FileNotFoundError(f"Schema folder not found: {schema_folder_path}")
        
        results = {
            "processed": 0,
            "updated": 0,
            "skipped": 0,
            "errors": 0,
            "tables": []
        }
        
        # Find all schema files (*.txt files)
        schema_files = list(schema_folder.rglob("*.txt"))
        
        documents_to_embed = []
        
        for file_path in schema_files:
            try:
                # Check if file needs processing
                file_hash = self._generate_file_hash(file_path)
                table_key = file_path.stem  # Use filename without extension as key
                
                if (table_key in self.schemas and 
                    self.schemas[table_key].file_hash == file_hash):
                    results["skipped"] += 1
                    continue
                
                # Parse schema file
                schema = self._parse_schema_file(file_path)
                if not schema:
                    results["errors"] += 1
                    continue
                
                # Enhance with basic relationship detection (LLM enhancement disabled by default to avoid API costs)
                schema = self._enhance_schema_with_llm(schema, enable_llm_enhancement=False)
                
                # Store schema
                self.schemas[table_key] = schema
                
                # Prepare for embedding
                columns_text = "\n".join([
                    f"{col['name']} ({col['data_type']}): {col['description']}"
                    for col in schema.columns
                ])
                
                relationships_text = ", ".join(schema.relationships) if schema.relationships else "None specified"
                
                doc_text = (
                    f"Table: {schema.schema_name}.{schema.table_name}\n"
                    f"Description: {schema.description}\n"
                    f"Columns:\n{columns_text}\n"
                    f"Relationships: {relationships_text}"
                )
                
                documents_to_embed.append(
                    Document(
                        page_content=doc_text,
                        metadata={
                            "table_name": schema.table_name,
                            "schema_name": schema.schema_name,
                            "file_name": table_key,
                            "full_table_name": f"{schema.schema_name}.{schema.table_name}"
                        }
                    )
                )
                
                if table_key in self.schemas:
                    results["updated"] += 1
                else:
                    results["processed"] += 1
                    
                results["tables"].append({
                    "table_name": f"{schema.schema_name}.{schema.table_name}",
                    "file_name": table_key,
                    "columns_count": len(schema.columns),
                    "status": "processed"
                })
                
            except Exception as e:
                results["errors"] += 1
                results["tables"].append({
                    "file_name": file_path.name,
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
    
    def query_schema(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Query the knowledge base for relevant database tables."""
        if not self.vector_store:
            return []
        
        try:
            # Search for similar documents
            docs = self.vector_store.similarity_search_with_score(query, k=top_k)
            
            results = []
            for doc, score in docs:
                file_name = doc.metadata.get("file_name", "unknown")
                if file_name in self.schemas:
                    schema = self.schemas[file_name]
                    results.append({
                        "table_name": f"{schema.schema_name}.{schema.table_name}",
                        "file_name": file_name,
                        "description": schema.description,
                        "columns_count": len(schema.columns),
                        # Return all columns instead of just first 5
                        # "columns": schema.columns[:5],  # First 5 columns
                        "columns": schema.columns,
                        "relationships": schema.relationships,
                        "file_path": schema.file_path,
                        "score": score
                    })
            
            return results
            
        except Exception as e:
            print(f"Error querying schema knowledge base: {e}")
            return []
    
    def get_table_details(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific table."""
        # Try to find by table name or file name
        schema = None
        for key, s in self.schemas.items():
            if (key == table_name or 
                s.table_name == table_name or 
                f"{s.schema_name}.{s.table_name}" == table_name):
                schema = s
                break
        
        if not schema:
            return None
        
        return {
            "table_name": f"{schema.schema_name}.{schema.table_name}",
            "schema_name": schema.schema_name,
            "description": schema.description,
            "columns": schema.columns,
            "relationships": schema.relationships,
            "file_path": schema.file_path,
            "last_modified": schema.last_modified,
            "columns_count": len(schema.columns)
        }
    
    def list_all_tables(self) -> List[Dict[str, Any]]:
        """List all tables in the knowledge base."""
        return [
            {
                "table_name": f"{schema.schema_name}.{schema.table_name}",
                "file_name": key,
                "description": schema.description,
                "columns_count": len(schema.columns),
                "last_modified": schema.last_modified
            }
            for key, schema in self.schemas.items()
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            "total_tables": len(self.schemas),
            "has_vector_store": self.vector_store is not None,
            "kb_path": str(self.kb_path),
            "last_updated": datetime.now().isoformat()
        }
    
    def search_columns(self, column_pattern: str) -> List[Dict[str, Any]]:
        """Search for tables containing columns matching a pattern."""
        results = []
        for key, schema in self.schemas.items():
            matching_columns = [
                col for col in schema.columns
                if column_pattern.lower() in col['name'].lower() or
                   column_pattern.lower() in col['description'].lower()
            ]
            
            if matching_columns:
                results.append({
                    "table_name": f"{schema.schema_name}.{schema.table_name}",
                    "file_name": key,
                    "description": schema.description,
                    "matching_columns": matching_columns,
                    "total_columns": len(schema.columns)
                })
        
        return results


# Configuration helper functions
def get_sql_schema_folder() -> str:
    """Get the configured database schema folder path."""
    return os.getenv("SQL_SCHEMA_FOLDER", "static/data/db_schema")

def get_sql_knowledge_base_path() -> str:
    """Get the configured SQL knowledge base path."""
    return os.getenv("SQL_KNOWLEDGE_BASE_PATH", "static/data/vector_store/sql_knowledge_base")


# Global knowledge base instance
sql_kb = SQLSchemaKnowledgeBase()


def build_sql_knowledge_base(folder_path: Optional[str] = None) -> Dict[str, Any]:
    """Build the SQL schema knowledge base from a folder of files.
    
    Args:
        folder_path: Path to schema files. If None, uses SQL_SCHEMA_FOLDER env var.
    """
    if folder_path is None:
        folder_path = get_sql_schema_folder()
    
    return sql_kb.build_knowledge_base(folder_path)


def query_sql_schema(query: str) -> List[Dict[str, Any]]:
    """Query the SQL schema knowledge base for relevant tables."""
    return sql_kb.query_schema(query)


def get_table_info(table_name: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific table."""
    return sql_kb.get_table_details(table_name)


def search_tables_by_column(column_pattern: str) -> List[Dict[str, Any]]:
    """Search for tables containing columns matching a pattern."""
    return sql_kb.search_columns(column_pattern)


def list_database_tables() -> List[Dict[str, Any]]:
    """List all tables in the database schema knowledge base."""
    return sql_kb.list_all_tables()
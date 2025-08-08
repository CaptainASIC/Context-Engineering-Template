"""
Document models for the RAG system.

This module defines the data models for documents, chunks, and metadata
used in the retrieval-augmented generation system.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import uuid4, UUID
from enum import Enum

from pydantic import BaseModel, Field, validator


class DocumentType(str, Enum):
    """Document type enumeration."""
    PDF = "pdf"
    TEXT = "txt"
    MARKDOWN = "md"
    DOCX = "docx"
    HTML = "html"
    JSON = "json"
    CSV = "csv"


class ProcessingStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    """Document metadata model."""
    title: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    file_size: Optional[int] = None
    page_count: Optional[int] = None
    language: Optional[str] = "en"
    tags: List[str] = Field(default_factory=list)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Document(BaseModel):
    """Main document model."""
    id: UUID = Field(default_factory=uuid4)
    content: str = Field(..., description="Document content")
    document_type: DocumentType = Field(..., description="Document type")
    source_path: Optional[str] = Field(None, description="Original file path")
    source_url: Optional[str] = Field(None, description="Source URL if web document")
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError("Document content cannot be empty")
        return v.strip()
    
    @validator('source_path', 'source_url')
    def validate_source(cls, v, values):
        # At least one source should be provided
        if not v and not values.get('source_url') and not values.get('source_path'):
            raise ValueError("Either source_path or source_url must be provided")
        return v
    
    def get_word_count(self) -> int:
        """Get word count of the document."""
        return len(self.content.split())
    
    def get_char_count(self) -> int:
        """Get character count of the document."""
        return len(self.content)
    
    def update_metadata(self, **kwargs) -> None:
        """Update document metadata."""
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
            else:
                self.metadata.custom_fields[key] = value
        self.updated_at = datetime.utcnow()
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class ChunkMetadata(BaseModel):
    """Chunk metadata model."""
    chunk_index: int = Field(..., description="Chunk index in document")
    start_char: int = Field(..., description="Start character position")
    end_char: int = Field(..., description="End character position")
    page_number: Optional[int] = Field(None, description="Page number if applicable")
    section_title: Optional[str] = Field(None, description="Section title if available")
    parent_chunk_id: Optional[UUID] = Field(None, description="Parent chunk ID for hierarchical chunks")
    overlap_with_previous: int = Field(default=0, description="Character overlap with previous chunk")
    overlap_with_next: int = Field(default=0, description="Character overlap with next chunk")
    
    class Config:
        json_encoders = {
            UUID: lambda v: str(v)
        }


class DocumentChunk(BaseModel):
    """Document chunk model for vector storage."""
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk content")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    chunk_metadata: ChunkMetadata = Field(..., description="Chunk-specific metadata")
    document_metadata: DocumentMetadata = Field(..., description="Parent document metadata")
    similarity_score: Optional[float] = Field(None, description="Similarity score for search results")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError("Chunk content cannot be empty")
        return v.strip()
    
    @validator('embedding')
    def validate_embedding(cls, v):
        if v is not None and len(v) == 0:
            raise ValueError("Embedding cannot be empty list")
        return v
    
    def get_word_count(self) -> int:
        """Get word count of the chunk."""
        return len(self.content.split())
    
    def get_char_count(self) -> int:
        """Get character count of the chunk."""
        return len(self.content)
    
    def get_context_window(self, window_size: int = 100) -> str:
        """
        Get content with context window around this chunk.
        
        Args:
            window_size: Number of characters to include before and after
            
        Returns:
            Content with context
        """
        # This would be implemented with access to the full document
        # For now, return the chunk content
        return self.content
    
    def to_search_result(self) -> Dict[str, Any]:
        """Convert to search result format."""
        return {
            "id": str(self.id),
            "document_id": str(self.document_id),
            "content": self.content,
            "similarity_score": self.similarity_score,
            "metadata": {
                "chunk_index": self.chunk_metadata.chunk_index,
                "page_number": self.chunk_metadata.page_number,
                "section_title": self.chunk_metadata.section_title,
                "document_title": self.document_metadata.title,
                "document_author": self.document_metadata.author,
                "word_count": self.get_word_count(),
                "char_count": self.get_char_count()
            }
        }
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class SearchQuery(BaseModel):
    """Search query model."""
    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to return")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    document_types: Optional[List[DocumentType]] = Field(None, description="Filter by document types")
    date_range: Optional[Dict[str, datetime]] = Field(None, description="Date range filter")
    metadata_filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    include_embeddings: bool = Field(default=False, description="Include embeddings in results")
    
    @validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Search query cannot be empty")
        return v.strip()
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SearchResult(BaseModel):
    """Search result model."""
    chunks: List[DocumentChunk] = Field(..., description="Retrieved chunks")
    total_results: int = Field(..., description="Total number of matching results")
    query_time_ms: float = Field(..., description="Query execution time in milliseconds")
    aggregations: Optional[Dict[str, Any]] = Field(None, description="Search aggregations")
    
    def get_unique_documents(self) -> List[UUID]:
        """Get list of unique document IDs in results."""
        return list(set(chunk.document_id for chunk in self.chunks))
    
    def group_by_document(self) -> Dict[UUID, List[DocumentChunk]]:
        """Group chunks by document ID."""
        grouped = {}
        for chunk in self.chunks:
            if chunk.document_id not in grouped:
                grouped[chunk.document_id] = []
            grouped[chunk.document_id].append(chunk)
        return grouped
    
    def get_top_documents(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top documents by average similarity score."""
        grouped = self.group_by_document()
        doc_scores = []
        
        for doc_id, chunks in grouped.items():
            avg_score = sum(chunk.similarity_score or 0 for chunk in chunks) / len(chunks)
            doc_scores.append({
                "document_id": str(doc_id),
                "chunk_count": len(chunks),
                "average_score": avg_score,
                "title": chunks[0].document_metadata.title,
                "author": chunks[0].document_metadata.author
            })
        
        doc_scores.sort(key=lambda x: x["average_score"], reverse=True)
        return doc_scores[:limit]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class DocumentStats(BaseModel):
    """Document statistics model."""
    total_documents: int = 0
    total_chunks: int = 0
    total_words: int = 0
    total_characters: int = 0
    document_types: Dict[str, int] = Field(default_factory=dict)
    processing_status: Dict[str, int] = Field(default_factory=dict)
    average_document_size: float = 0.0
    average_chunk_size: float = 0.0
    languages: Dict[str, int] = Field(default_factory=dict)
    date_range: Optional[Dict[str, datetime]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


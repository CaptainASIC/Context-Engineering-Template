"""
Database models for Vector Database with pgvector.

This module defines SQLAlchemy models for storing documents,
embeddings, and metadata in PostgreSQL with pgvector extension.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from uuid import uuid4, UUID

from sqlalchemy import (
    Column, String, Text, Integer, Float, DateTime, Boolean, 
    JSON, LargeBinary, Index, UniqueConstraint, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class TimestampMixin:
    """Mixin for timestamp fields."""
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)


class Document(Base, TimestampMixin):
    """Document model for storing text documents and metadata."""
    
    __tablename__ = "documents"
    
    # Primary key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Document content and metadata
    title = Column(String(500), nullable=False, index=True)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False, unique=True, index=True)
    
    # Document classification
    document_type = Column(String(50), nullable=False, index=True)
    source_type = Column(String(50), nullable=False, default="upload")  # upload, url, api, etc.
    source_url = Column(String(1000), nullable=True)
    
    # File information
    file_name = Column(String(255), nullable=True)
    file_path = Column(String(1000), nullable=True)
    file_size = Column(Integer, nullable=True)
    mime_type = Column(String(100), nullable=True)
    
    # Content metadata
    language = Column(String(10), nullable=False, default="en")
    word_count = Column(Integer, nullable=True)
    character_count = Column(Integer, nullable=True)
    
    # Processing status
    processing_status = Column(String(20), nullable=False, default="pending")  # pending, processing, completed, failed
    processing_error = Column(Text, nullable=True)
    
    # User and organization
    user_id = Column(String(100), nullable=True, index=True)
    organization_id = Column(String(100), nullable=True, index=True)
    
    # Tags and categories
    tags = Column(JSON, nullable=True, default=list)
    categories = Column(JSON, nullable=True, default=list)
    custom_metadata = Column(JSON, nullable=True, default=dict)
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    embeddings = relationship("DocumentEmbedding", back_populates="document", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_documents_type_status', 'document_type', 'processing_status'),
        Index('idx_documents_user_org', 'user_id', 'organization_id'),
        Index('idx_documents_created_at', 'created_at'),
        Index('idx_documents_tags', 'tags', postgresql_using='gin'),
        Index('idx_documents_custom_metadata', 'custom_metadata', postgresql_using='gin'),
    )
    
    def __repr__(self):
        return f"<Document(id={self.id}, title='{self.title[:50]}...', type='{self.document_type}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "title": self.title,
            "document_type": self.document_type,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "language": self.language,
            "word_count": self.word_count,
            "character_count": self.character_count,
            "processing_status": self.processing_status,
            "user_id": self.user_id,
            "organization_id": self.organization_id,
            "tags": self.tags or [],
            "categories": self.categories or [],
            "custom_metadata": self.custom_metadata or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class DocumentChunk(Base, TimestampMixin):
    """Document chunk model for storing text chunks with position information."""
    
    __tablename__ = "document_chunks"
    
    # Primary key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign key to document
    document_id = Column(PGUUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Chunk content and metadata
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False, index=True)
    
    # Position information
    chunk_index = Column(Integer, nullable=False)
    start_position = Column(Integer, nullable=False)
    end_position = Column(Integer, nullable=False)
    
    # Chunk metadata
    word_count = Column(Integer, nullable=True)
    character_count = Column(Integer, nullable=True)
    
    # Hierarchical information
    section_title = Column(String(500), nullable=True)
    page_number = Column(Integer, nullable=True)
    paragraph_index = Column(Integer, nullable=True)
    
    # Overlap information
    overlap_with_previous = Column(Integer, nullable=False, default=0)
    overlap_with_next = Column(Integer, nullable=False, default=0)
    
    # Processing metadata
    chunk_type = Column(String(50), nullable=False, default="text")  # text, header, table, image_caption, etc.
    extraction_confidence = Column(Float, nullable=True)
    
    # Custom metadata
    custom_metadata = Column(JSON, nullable=True, default=dict)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    embeddings = relationship("ChunkEmbedding", back_populates="chunk", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_chunks_document_index', 'document_id', 'chunk_index'),
        Index('idx_chunks_position', 'start_position', 'end_position'),
        Index('idx_chunks_type', 'chunk_type'),
        UniqueConstraint('document_id', 'chunk_index', name='uq_document_chunk_index'),
    )
    
    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, index={self.chunk_index})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "document_id": str(self.document_id),
            "content": self.content,
            "chunk_index": self.chunk_index,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "word_count": self.word_count,
            "character_count": self.character_count,
            "section_title": self.section_title,
            "page_number": self.page_number,
            "paragraph_index": self.paragraph_index,
            "chunk_type": self.chunk_type,
            "extraction_confidence": self.extraction_confidence,
            "custom_metadata": self.custom_metadata or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class EmbeddingModel(Base, TimestampMixin):
    """Embedding model information."""
    
    __tablename__ = "embedding_models"
    
    # Primary key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Model information
    name = Column(String(200), nullable=False, unique=True, index=True)
    provider = Column(String(100), nullable=False)
    version = Column(String(50), nullable=True)
    
    # Model specifications
    dimension = Column(Integer, nullable=False)
    max_sequence_length = Column(Integer, nullable=False)
    model_type = Column(String(50), nullable=False, default="sentence_transformer")
    
    # Model configuration
    configuration = Column(JSON, nullable=True, default=dict)
    
    # Status and metadata
    is_active = Column(Boolean, nullable=False, default=True)
    description = Column(Text, nullable=True)
    
    # Relationships
    document_embeddings = relationship("DocumentEmbedding", back_populates="model")
    chunk_embeddings = relationship("ChunkEmbedding", back_populates="model")
    
    def __repr__(self):
        return f"<EmbeddingModel(name='{self.name}', dimension={self.dimension})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "name": self.name,
            "provider": self.provider,
            "version": self.version,
            "dimension": self.dimension,
            "max_sequence_length": self.max_sequence_length,
            "model_type": self.model_type,
            "configuration": self.configuration or {},
            "is_active": self.is_active,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class DocumentEmbedding(Base, TimestampMixin):
    """Document-level embeddings."""
    
    __tablename__ = "document_embeddings"
    
    # Primary key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign keys
    document_id = Column(PGUUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    model_id = Column(PGUUID(as_uuid=True), ForeignKey("embedding_models.id"), nullable=False, index=True)
    
    # Embedding data
    embedding = Column(Vector, nullable=False)
    
    # Embedding metadata
    embedding_type = Column(String(50), nullable=False, default="document")  # document, summary, title
    source_text = Column(Text, nullable=True)  # Text used to generate embedding
    source_text_hash = Column(String(64), nullable=False, index=True)
    
    # Processing metadata
    processing_time_ms = Column(Float, nullable=True)
    token_count = Column(Integer, nullable=True)
    
    # Relationships
    document = relationship("Document", back_populates="embeddings")
    model = relationship("EmbeddingModel", back_populates="document_embeddings")
    
    # Indexes
    __table_args__ = (
        Index('idx_doc_embeddings_document_model', 'document_id', 'model_id'),
        Index('idx_doc_embeddings_type', 'embedding_type'),
        UniqueConstraint('document_id', 'model_id', 'embedding_type', name='uq_document_model_embedding_type'),
    )
    
    def __repr__(self):
        return f"<DocumentEmbedding(id={self.id}, document_id={self.document_id}, type='{self.embedding_type}')>"


class ChunkEmbedding(Base, TimestampMixin):
    """Chunk-level embeddings for semantic search."""
    
    __tablename__ = "chunk_embeddings"
    
    # Primary key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign keys
    chunk_id = Column(PGUUID(as_uuid=True), ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=False, index=True)
    model_id = Column(PGUUID(as_uuid=True), ForeignKey("embedding_models.id"), nullable=False, index=True)
    
    # Embedding data
    embedding = Column(Vector, nullable=False)
    
    # Embedding metadata
    source_text_hash = Column(String(64), nullable=False, index=True)
    
    # Processing metadata
    processing_time_ms = Column(Float, nullable=True)
    token_count = Column(Integer, nullable=True)
    
    # Search optimization
    norm = Column(Float, nullable=True)  # L2 norm for optimization
    
    # Relationships
    chunk = relationship("DocumentChunk", back_populates="embeddings")
    model = relationship("EmbeddingModel", back_populates="chunk_embeddings")
    
    # Indexes for vector similarity search
    __table_args__ = (
        Index('idx_chunk_embeddings_chunk_model', 'chunk_id', 'model_id'),
        UniqueConstraint('chunk_id', 'model_id', name='uq_chunk_model_embedding'),
        # Vector similarity indexes (created separately due to pgvector requirements)
    )
    
    def __repr__(self):
        return f"<ChunkEmbedding(id={self.id}, chunk_id={self.chunk_id})>"


class SearchQuery(Base, TimestampMixin):
    """Search query logging and analytics."""
    
    __tablename__ = "search_queries"
    
    # Primary key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Query information
    query_text = Column(Text, nullable=False)
    query_hash = Column(String(64), nullable=False, index=True)
    query_type = Column(String(50), nullable=False, default="semantic")  # semantic, keyword, hybrid
    
    # Search parameters
    model_id = Column(PGUUID(as_uuid=True), ForeignKey("embedding_models.id"), nullable=True, index=True)
    limit_requested = Column(Integer, nullable=False, default=10)
    similarity_threshold = Column(Float, nullable=True)
    distance_metric = Column(String(20), nullable=False, default="cosine")
    
    # Filters applied
    filters = Column(JSON, nullable=True, default=dict)
    
    # Results and performance
    results_count = Column(Integer, nullable=False, default=0)
    execution_time_ms = Column(Float, nullable=True)
    cache_hit = Column(Boolean, nullable=False, default=False)
    
    # User context
    user_id = Column(String(100), nullable=True, index=True)
    session_id = Column(String(100), nullable=True, index=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(String(500), nullable=True)
    
    # Relationships
    model = relationship("EmbeddingModel")
    
    # Indexes
    __table_args__ = (
        Index('idx_search_queries_user_session', 'user_id', 'session_id'),
        Index('idx_search_queries_created_at', 'created_at'),
        Index('idx_search_queries_type_model', 'query_type', 'model_id'),
    )
    
    def __repr__(self):
        return f"<SearchQuery(id={self.id}, query='{self.query_text[:50]}...', results={self.results_count})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "query_text": self.query_text,
            "query_type": self.query_type,
            "model_id": str(self.model_id) if self.model_id else None,
            "limit_requested": self.limit_requested,
            "similarity_threshold": self.similarity_threshold,
            "distance_metric": self.distance_metric,
            "filters": self.filters or {},
            "results_count": self.results_count,
            "execution_time_ms": self.execution_time_ms,
            "cache_hit": self.cache_hit,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class Collection(Base, TimestampMixin):
    """Collections for organizing documents."""
    
    __tablename__ = "collections"
    
    # Primary key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Collection information
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Collection metadata
    collection_type = Column(String(50), nullable=False, default="user")  # user, system, shared
    is_public = Column(Boolean, nullable=False, default=False)
    
    # Owner information
    owner_id = Column(String(100), nullable=False, index=True)
    organization_id = Column(String(100), nullable=True, index=True)
    
    # Collection settings
    settings = Column(JSON, nullable=True, default=dict)
    
    # Statistics
    document_count = Column(Integer, nullable=False, default=0)
    total_size_bytes = Column(Integer, nullable=False, default=0)
    
    # Indexes
    __table_args__ = (
        Index('idx_collections_owner_org', 'owner_id', 'organization_id'),
        Index('idx_collections_type_public', 'collection_type', 'is_public'),
        UniqueConstraint('name', 'owner_id', name='uq_collection_name_owner'),
    )
    
    def __repr__(self):
        return f"<Collection(id={self.id}, name='{self.name}', owner='{self.owner_id}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "collection_type": self.collection_type,
            "is_public": self.is_public,
            "owner_id": self.owner_id,
            "organization_id": self.organization_id,
            "settings": self.settings or {},
            "document_count": self.document_count,
            "total_size_bytes": self.total_size_bytes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class DocumentCollection(Base):
    """Many-to-many relationship between documents and collections."""
    
    __tablename__ = "document_collections"
    
    # Composite primary key
    document_id = Column(PGUUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True)
    collection_id = Column(PGUUID(as_uuid=True), ForeignKey("collections.id", ondelete="CASCADE"), primary_key=True)
    
    # Relationship metadata
    added_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    added_by = Column(String(100), nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_doc_collections_collection', 'collection_id'),
        Index('idx_doc_collections_added_at', 'added_at'),
    )


# Database utility functions
def create_vector_indexes(engine):
    """Create vector similarity indexes."""
    from sqlalchemy import text
    
    with engine.connect() as conn:
        # Create vector indexes for chunk embeddings
        # IVFFlat index for approximate nearest neighbor search
        conn.execute(text("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunk_embeddings_vector_ivfflat
            ON chunk_embeddings 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """))
        
        # HNSW index for high-dimensional vectors (if supported)
        try:
            conn.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunk_embeddings_vector_hnsw
                ON chunk_embeddings 
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """))
        except Exception:
            # HNSW might not be available in all pgvector versions
            pass
        
        conn.commit()


def create_database_functions(engine):
    """Create custom database functions for vector operations."""
    from sqlalchemy import text
    
    with engine.connect() as conn:
        # Function for cosine similarity
        conn.execute(text("""
            CREATE OR REPLACE FUNCTION cosine_similarity(a vector, b vector)
            RETURNS float AS $$
            BEGIN
                RETURN 1 - (a <=> b);
            END;
            $$ LANGUAGE plpgsql IMMUTABLE;
        """))
        
        # Function for L2 distance
        conn.execute(text("""
            CREATE OR REPLACE FUNCTION l2_distance(a vector, b vector)
            RETURNS float AS $$
            BEGIN
                RETURN a <-> b;
            END;
            $$ LANGUAGE plpgsql IMMUTABLE;
        """))
        
        # Function for inner product
        conn.execute(text("""
            CREATE OR REPLACE FUNCTION inner_product(a vector, b vector)
            RETURNS float AS $$
            BEGIN
                RETURN a <#> b;
            END;
            $$ LANGUAGE plpgsql IMMUTABLE;
        """))
        
        conn.commit()


def get_model_registry() -> Dict[str, type]:
    """Get registry of all database models."""
    return {
        "Document": Document,
        "DocumentChunk": DocumentChunk,
        "EmbeddingModel": EmbeddingModel,
        "DocumentEmbedding": DocumentEmbedding,
        "ChunkEmbedding": ChunkEmbedding,
        "SearchQuery": SearchQuery,
        "Collection": Collection,
        "DocumentCollection": DocumentCollection
    }


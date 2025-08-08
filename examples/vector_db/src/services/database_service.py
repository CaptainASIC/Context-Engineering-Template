"""
Database Service for Vector Database with pgvector.

This module provides database connection management, session handling,
and high-level database operations for the vector database system.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from contextlib import asynccontextmanager
import logging
from datetime import datetime, timezone

import asyncpg
from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from databases import Database

from config.settings import EnhancedVectorDBSettings, get_settings
from src.models.database_models import (
    Base, Document, DocumentChunk, EmbeddingModel, 
    DocumentEmbedding, ChunkEmbedding, SearchQuery, Collection,
    create_vector_indexes, create_database_functions
)

logger = logging.getLogger(__name__)


class DatabaseService:
    """
    Database service for managing connections and operations.
    
    Provides both sync and async database operations with connection pooling,
    transaction management, and pgvector extension support.
    """
    
    def __init__(self, settings: Optional[EnhancedVectorDBSettings] = None):
        """Initialize database service."""
        self.settings = settings or get_settings()
        
        # Connection URLs
        self.sync_url = self.settings.get_connection_url()
        self.async_url = self.settings.get_async_connection_url()
        
        # Engines and sessions
        self.sync_engine = None
        self.async_engine = None
        self.sync_session_factory = None
        self.async_session_factory = None
        self.database = None
        
        # Connection state
        self._initialized = False
        self._health_status = {"status": "unknown", "last_check": None}
        
        logger.info(f"Initialized DatabaseService for {self.settings.database.host}")
    
    async def initialize(self) -> bool:
        """
        Initialize database connections and setup.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Create engines
            await self._create_engines()
            
            # Setup session factories
            self._setup_session_factories()
            
            # Initialize database connection
            await self._initialize_database_connection()
            
            # Setup database schema
            await self._setup_database_schema()
            
            # Create vector indexes and functions
            await self._setup_vector_extensions()
            
            self._initialized = True
            logger.info("Database service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database service: {e}")
            return False
    
    async def _create_engines(self):
        """Create SQLAlchemy engines."""
        # Sync engine
        self.sync_engine = create_engine(
            self.sync_url,
            pool_size=self.settings.database.min_connections,
            max_overflow=self.settings.database.max_connections - self.settings.database.min_connections,
            pool_timeout=self.settings.database.connection_timeout,
            pool_recycle=3600,  # Recycle connections every hour
            echo=self.settings.debug
        )
        
        # Async engine
        self.async_engine = create_async_engine(
            self.async_url,
            pool_size=self.settings.database.min_connections,
            max_overflow=self.settings.database.max_connections - self.settings.database.min_connections,
            pool_timeout=self.settings.database.connection_timeout,
            pool_recycle=3600,
            echo=self.settings.debug
        )
        
        logger.info("Database engines created")
    
    def _setup_session_factories(self):
        """Setup session factories."""
        self.sync_session_factory = sessionmaker(
            bind=self.sync_engine,
            expire_on_commit=False
        )
        
        self.async_session_factory = async_sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        logger.info("Session factories configured")
    
    async def _initialize_database_connection(self):
        """Initialize database connection for raw queries."""
        self.database = Database(self.async_url)
        await self.database.connect()
        logger.info("Database connection established")
    
    async def _setup_database_schema(self):
        """Setup database schema and extensions."""
        try:
            # Create pgvector extension
            await self.database.execute("CREATE EXTENSION IF NOT EXISTS vector")
            logger.info("pgvector extension created")
            
            # Create all tables
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database schema created")
            
        except Exception as e:
            logger.error(f"Failed to setup database schema: {e}")
            raise
    
    async def _setup_vector_extensions(self):
        """Setup vector indexes and custom functions."""
        try:
            # Create vector indexes (using sync engine for DDL)
            create_vector_indexes(self.sync_engine)
            logger.info("Vector indexes created")
            
            # Create custom functions (using sync engine)
            create_database_functions(self.sync_engine)
            logger.info("Custom vector functions created")
            
        except Exception as e:
            logger.warning(f"Failed to setup vector extensions: {e}")
            # Non-critical, continue without vector optimizations
    
    async def disconnect(self):
        """Disconnect from database."""
        try:
            if self.database:
                await self.database.disconnect()
            
            if self.async_engine:
                await self.async_engine.dispose()
            
            if self.sync_engine:
                self.sync_engine.dispose()
            
            self._initialized = False
            logger.info("Database service disconnected")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Async context manager for database sessions.
        
        Yields:
            AsyncSession instance
        """
        if not self._initialized:
            await self.initialize()
        
        async with self.async_session_factory() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"Session error: {e}")
                raise
            finally:
                await session.close()
    
    def get_sync_session(self) -> Session:
        """
        Get synchronous database session.
        
        Returns:
            Session instance
        """
        if not self._initialized:
            raise RuntimeError("Database service not initialized")
        
        return self.sync_session_factory()
    
    async def execute_query(
        self,
        query: str,
        values: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute raw SQL query.
        
        Args:
            query: SQL query string
            values: Query parameters
            
        Returns:
            List of result rows as dictionaries
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            if values:
                result = await self.database.fetch_all(query, values)
            else:
                result = await self.database.fetch_all(query)
            
            return [dict(row) for row in result]
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Values: {values}")
            raise
    
    async def execute_command(
        self,
        query: str,
        values: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Execute SQL command (INSERT, UPDATE, DELETE).
        
        Args:
            query: SQL command string
            values: Query parameters
            
        Returns:
            Number of affected rows
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            if values:
                result = await self.database.execute(query, values)
            else:
                result = await self.database.execute(query)
            
            return result
            
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Values: {values}")
            raise
    
    async def bulk_insert(
        self,
        table_name: str,
        records: List[Dict[str, Any]],
        batch_size: Optional[int] = None
    ) -> int:
        """
        Bulk insert records into table.
        
        Args:
            table_name: Target table name
            records: List of record dictionaries
            batch_size: Batch size for processing
            
        Returns:
            Number of inserted records
        """
        if not records:
            return 0
        
        batch_size = batch_size or self.settings.batch_size
        total_inserted = 0
        
        # Get column names from first record
        columns = list(records[0].keys())
        placeholders = ", ".join([f":{col}" for col in columns])
        column_names = ", ".join(columns)
        
        query = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"
        
        # Process in batches
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            
            try:
                async with self.get_async_session() as session:
                    result = await session.execute(text(query), batch)
                    await session.commit()
                    total_inserted += len(batch)
                    
            except Exception as e:
                logger.error(f"Bulk insert failed for batch {i//batch_size + 1}: {e}")
                raise
        
        logger.info(f"Bulk inserted {total_inserted} records into {table_name}")
        return total_inserted
    
    async def create_embedding_model(
        self,
        name: str,
        provider: str,
        dimension: int,
        max_sequence_length: int,
        **kwargs
    ) -> EmbeddingModel:
        """
        Create or update embedding model record.
        
        Args:
            name: Model name
            provider: Model provider
            dimension: Embedding dimension
            max_sequence_length: Max sequence length
            **kwargs: Additional model parameters
            
        Returns:
            EmbeddingModel instance
        """
        async with self.get_async_session() as session:
            # Check if model exists
            result = await session.execute(
                text("SELECT * FROM embedding_models WHERE name = :name"),
                {"name": name}
            )
            existing = result.fetchone()
            
            if existing:
                # Update existing model
                update_data = {
                    "provider": provider,
                    "dimension": dimension,
                    "max_sequence_length": max_sequence_length,
                    "updated_at": datetime.now(timezone.utc),
                    **kwargs
                }
                
                await session.execute(
                    text("""
                        UPDATE embedding_models 
                        SET provider = :provider, dimension = :dimension,
                            max_sequence_length = :max_sequence_length,
                            updated_at = :updated_at
                        WHERE name = :name
                    """),
                    {"name": name, **update_data}
                )
                
                model_id = existing.id
            else:
                # Create new model
                model = EmbeddingModel(
                    name=name,
                    provider=provider,
                    dimension=dimension,
                    max_sequence_length=max_sequence_length,
                    **kwargs
                )
                
                session.add(model)
                await session.flush()
                model_id = model.id
            
            await session.commit()
            
            # Fetch and return the model
            result = await session.execute(
                text("SELECT * FROM embedding_models WHERE id = :id"),
                {"id": model_id}
            )
            model_data = result.fetchone()
            
            return EmbeddingModel(**dict(model_data))
    
    async def get_embedding_model(self, name: str) -> Optional[EmbeddingModel]:
        """
        Get embedding model by name.
        
        Args:
            name: Model name
            
        Returns:
            EmbeddingModel instance or None
        """
        async with self.get_async_session() as session:
            result = await session.execute(
                text("SELECT * FROM embedding_models WHERE name = :name AND is_active = true"),
                {"name": name}
            )
            model_data = result.fetchone()
            
            if model_data:
                return EmbeddingModel(**dict(model_data))
            return None
    
    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        model_id: str,
        limit: int = 10,
        threshold: float = 0.7,
        distance_metric: str = "cosine",
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar document chunks using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            model_id: Embedding model ID
            limit: Maximum number of results
            threshold: Similarity threshold
            distance_metric: Distance metric (cosine, l2, inner_product)
            filters: Additional filters
            
        Returns:
            List of similar chunks with metadata
        """
        # Convert embedding to pgvector format
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
        
        # Choose distance operator based on metric
        if distance_metric == "cosine":
            distance_op = "<=>"
            similarity_expr = f"1 - (ce.embedding {distance_op} '{embedding_str}'::vector)"
        elif distance_metric == "l2":
            distance_op = "<->"
            similarity_expr = f"1 / (1 + (ce.embedding {distance_op} '{embedding_str}'::vector))"
        elif distance_metric == "inner_product":
            distance_op = "<#>"
            similarity_expr = f"ce.embedding <#> '{embedding_str}'::vector"
        else:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")
        
        # Build base query
        query = f"""
        SELECT 
            ce.id as embedding_id,
            ce.chunk_id,
            dc.content,
            dc.chunk_index,
            dc.section_title,
            dc.page_number,
            d.id as document_id,
            d.title as document_title,
            d.document_type,
            d.source_url,
            d.tags,
            d.custom_metadata,
            {similarity_expr} as similarity,
            ce.embedding {distance_op} '{embedding_str}'::vector as distance
        FROM chunk_embeddings ce
        JOIN document_chunks dc ON ce.chunk_id = dc.id
        JOIN documents d ON dc.document_id = d.id
        WHERE ce.model_id = :model_id
        """
        
        # Add filters
        query_params = {"model_id": model_id}
        
        if filters:
            if "document_types" in filters:
                placeholders = ",".join([f":doc_type_{i}" for i in range(len(filters["document_types"]))])
                query += f" AND d.document_type IN ({placeholders})"
                for i, doc_type in enumerate(filters["document_types"]):
                    query_params[f"doc_type_{i}"] = doc_type
            
            if "user_id" in filters:
                query += " AND d.user_id = :user_id"
                query_params["user_id"] = filters["user_id"]
            
            if "tags" in filters:
                query += " AND d.tags && :tags"
                query_params["tags"] = filters["tags"]
        
        # Add similarity threshold and ordering
        if distance_metric == "cosine":
            query += f" AND (1 - (ce.embedding {distance_op} '{embedding_str}'::vector)) >= :threshold"
        else:
            query += f" AND (ce.embedding {distance_op} '{embedding_str}'::vector) <= :threshold"
        
        query_params["threshold"] = threshold
        
        query += f" ORDER BY ce.embedding {distance_op} '{embedding_str}'::vector"
        query += " LIMIT :limit"
        query_params["limit"] = limit
        
        # Execute query
        try:
            results = await self.execute_query(query, query_params)
            
            # Process results
            processed_results = []
            for row in results:
                processed_results.append({
                    "embedding_id": str(row["embedding_id"]),
                    "chunk_id": str(row["chunk_id"]),
                    "document_id": str(row["document_id"]),
                    "content": row["content"],
                    "chunk_index": row["chunk_index"],
                    "section_title": row["section_title"],
                    "page_number": row["page_number"],
                    "document_title": row["document_title"],
                    "document_type": row["document_type"],
                    "source_url": row["source_url"],
                    "tags": row["tags"] or [],
                    "custom_metadata": row["custom_metadata"] or {},
                    "similarity": float(row["similarity"]),
                    "distance": float(row["distance"])
                })
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise
    
    async def log_search_query(
        self,
        query_text: str,
        query_type: str,
        model_id: Optional[str],
        results_count: int,
        execution_time_ms: float,
        **kwargs
    ) -> str:
        """
        Log search query for analytics.
        
        Args:
            query_text: Search query text
            query_type: Type of search
            model_id: Embedding model ID
            results_count: Number of results returned
            execution_time_ms: Execution time in milliseconds
            **kwargs: Additional parameters
            
        Returns:
            Query log ID
        """
        import hashlib
        
        query_hash = hashlib.sha256(query_text.encode()).hexdigest()
        
        search_query = SearchQuery(
            query_text=query_text,
            query_hash=query_hash,
            query_type=query_type,
            model_id=model_id,
            results_count=results_count,
            execution_time_ms=execution_time_ms,
            **kwargs
        )
        
        async with self.get_async_session() as session:
            session.add(search_query)
            await session.commit()
            await session.refresh(search_query)
            
            return str(search_query.id)
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        stats = {}
        
        try:
            # Document statistics
            doc_stats = await self.execute_query("""
                SELECT 
                    COUNT(*) as total_documents,
                    COUNT(DISTINCT document_type) as unique_document_types,
                    SUM(file_size) as total_file_size,
                    AVG(word_count) as avg_word_count
                FROM documents
                WHERE processing_status = 'completed'
            """)
            
            if doc_stats:
                stats["documents"] = dict(doc_stats[0])
            
            # Chunk statistics
            chunk_stats = await self.execute_query("""
                SELECT 
                    COUNT(*) as total_chunks,
                    AVG(word_count) as avg_chunk_word_count,
                    AVG(character_count) as avg_chunk_char_count
                FROM document_chunks
            """)
            
            if chunk_stats:
                stats["chunks"] = dict(chunk_stats[0])
            
            # Embedding statistics
            embedding_stats = await self.execute_query("""
                SELECT 
                    em.name as model_name,
                    COUNT(ce.id) as embedding_count,
                    em.dimension
                FROM embedding_models em
                LEFT JOIN chunk_embeddings ce ON em.id = ce.model_id
                WHERE em.is_active = true
                GROUP BY em.id, em.name, em.dimension
            """)
            
            stats["embeddings"] = [dict(row) for row in embedding_stats]
            
            # Search statistics
            search_stats = await self.execute_query("""
                SELECT 
                    COUNT(*) as total_searches,
                    AVG(execution_time_ms) as avg_execution_time,
                    AVG(results_count) as avg_results_count,
                    COUNT(*) FILTER (WHERE cache_hit = true) as cache_hits
                FROM search_queries
                WHERE created_at >= NOW() - INTERVAL '24 hours'
            """)
            
            if search_stats:
                stats["searches_24h"] = dict(search_stats[0])
            
            # Collection statistics
            collection_stats = await self.execute_query("""
                SELECT 
                    COUNT(*) as total_collections,
                    SUM(document_count) as total_documents_in_collections,
                    AVG(document_count) as avg_documents_per_collection
                FROM collections
            """)
            
            if collection_stats:
                stats["collections"] = dict(collection_stats[0])
            
            stats["timestamp"] = datetime.now(timezone.utc).isoformat()
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            stats["error"] = str(e)
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform database health check.
        
        Returns:
            Health check results
        """
        health = {
            "status": "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": {}
        }
        
        try:
            # Test basic connectivity
            if not self._initialized:
                await self.initialize()
            
            # Test query execution
            result = await self.execute_query("SELECT 1 as test")
            if result and result[0]["test"] == 1:
                health["status"] = "healthy"
                health["details"]["connectivity"] = "ok"
            else:
                health["status"] = "unhealthy"
                health["details"]["connectivity"] = "failed"
            
            # Test pgvector extension
            try:
                await self.execute_query("SELECT '[1,2,3]'::vector")
                health["details"]["pgvector"] = "ok"
            except Exception as e:
                health["details"]["pgvector"] = f"error: {str(e)}"
            
            # Get connection pool stats
            if self.async_engine:
                pool = self.async_engine.pool
                health["details"]["connection_pool"] = {
                    "size": pool.size(),
                    "checked_in": pool.checkedin(),
                    "checked_out": pool.checkedout(),
                    "overflow": pool.overflow(),
                    "invalid": pool.invalid()
                }
            
            # Get basic stats
            try:
                stats = await self.get_database_stats()
                health["details"]["stats"] = {
                    "documents": stats.get("documents", {}).get("total_documents", 0),
                    "chunks": stats.get("chunks", {}).get("total_chunks", 0),
                    "embeddings": sum(e.get("embedding_count", 0) for e in stats.get("embeddings", []))
                }
            except Exception as e:
                health["details"]["stats_error"] = str(e)
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["details"]["error"] = str(e)
            logger.error(f"Health check failed: {e}")
        
        self._health_status = health
        return health
    
    async def cleanup_old_data(self, days: int = 30) -> Dict[str, int]:
        """
        Cleanup old data from database.
        
        Args:
            days: Number of days to keep data
            
        Returns:
            Dictionary with cleanup statistics
        """
        cleanup_stats = {"deleted_queries": 0, "deleted_embeddings": 0}
        
        try:
            # Delete old search queries
            result = await self.execute_command("""
                DELETE FROM search_queries 
                WHERE created_at < NOW() - INTERVAL '%s days'
            """, {"days": days})
            cleanup_stats["deleted_queries"] = result
            
            # Delete embeddings for deleted documents
            result = await self.execute_command("""
                DELETE FROM chunk_embeddings 
                WHERE chunk_id NOT IN (SELECT id FROM document_chunks)
            """)
            cleanup_stats["deleted_embeddings"] = result
            
            logger.info(f"Cleanup completed: {cleanup_stats}")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            raise
        
        return cleanup_stats


# Global database service instance
_db_service = None


async def get_database_service() -> DatabaseService:
    """Get global database service instance."""
    global _db_service
    
    if _db_service is None:
        _db_service = DatabaseService()
        await _db_service.initialize()
    
    return _db_service


async def close_database_service():
    """Close global database service."""
    global _db_service
    
    if _db_service:
        await _db_service.disconnect()
        _db_service = None


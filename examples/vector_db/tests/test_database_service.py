"""
Tests for Database Service.

This module tests the database service including connection management,
CRUD operations, and vector similarity search functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np

from src.services.database_service import DatabaseService
from src.models.database_models import Document, DocumentChunk, EmbeddingModel, ChunkEmbedding
from config.settings import EnhancedVectorDBSettings, DatabaseConfig, EmbeddingConfig


@pytest.fixture
def mock_settings():
    """Mock database settings for testing."""
    settings = EnhancedVectorDBSettings()
    settings.database = DatabaseConfig(
        host="localhost",
        port=5432,
        database="test_vectordb",
        username="test_user",
        password="test_password"
    )
    settings.debug = True
    return settings


@pytest.fixture
def sample_document_data():
    """Sample document data for testing."""
    return {
        "title": "Test Document",
        "content": "This is a test document with some content for testing purposes.",
        "document_type": "text",
        "source_type": "upload",
        "language": "en",
        "word_count": 12,
        "character_count": 65,
        "processing_status": "completed",
        "user_id": "test_user_123",
        "tags": ["test", "document"],
        "custom_metadata": {"category": "testing"}
    }


@pytest.fixture
def sample_chunk_data():
    """Sample document chunk data for testing."""
    return {
        "content": "This is a test chunk of content.",
        "chunk_index": 0,
        "start_position": 0,
        "end_position": 32,
        "word_count": 7,
        "character_count": 32,
        "chunk_type": "text"
    }


@pytest.fixture
def sample_embedding_model_data():
    """Sample embedding model data for testing."""
    return {
        "name": "test-model",
        "provider": "sentence-transformers",
        "dimension": 384,
        "max_sequence_length": 512,
        "model_type": "sentence_transformer"
    }


@pytest.fixture
def sample_embedding_vector():
    """Sample embedding vector for testing."""
    return np.random.rand(384).tolist()


class TestDatabaseService:
    """Test database service functionality."""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, mock_settings):
        """Test database service initialization."""
        service = DatabaseService(mock_settings)
        
        assert service.settings == mock_settings
        assert service.sync_engine is None
        assert service.async_engine is None
        assert not service._initialized
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_settings):
        """Test successful database initialization."""
        with patch.multiple(
            'src.services.database_service.DatabaseService',
            _create_engines=AsyncMock(),
            _setup_session_factories=Mock(),
            _initialize_database_connection=AsyncMock(),
            _setup_database_schema=AsyncMock(),
            _setup_vector_extensions=AsyncMock()
        ):
            service = DatabaseService(mock_settings)
            result = await service.initialize()
            
            assert result is True
            assert service._initialized is True
    
    @pytest.mark.asyncio
    async def test_initialize_failure(self, mock_settings):
        """Test failed database initialization."""
        with patch.object(
            DatabaseService, '_create_engines', 
            side_effect=Exception("Connection failed")
        ):
            service = DatabaseService(mock_settings)
            result = await service.initialize()
            
            assert result is False
            assert service._initialized is False
    
    @pytest.mark.asyncio
    async def test_create_engines(self, mock_settings):
        """Test engine creation."""
        with patch('src.services.database_service.create_engine') as mock_sync_engine, \
             patch('src.services.database_service.create_async_engine') as mock_async_engine:
            
            service = DatabaseService(mock_settings)
            await service._create_engines()
            
            mock_sync_engine.assert_called_once()
            mock_async_engine.assert_called_once()
            assert service.sync_engine is not None
            assert service.async_engine is not None
    
    @pytest.mark.asyncio
    async def test_disconnect(self, mock_settings):
        """Test database disconnection."""
        service = DatabaseService(mock_settings)
        
        # Mock connections
        service.database = AsyncMock()
        service.async_engine = AsyncMock()
        service.sync_engine = Mock()
        service._initialized = True
        
        await service.disconnect()
        
        service.database.disconnect.assert_called_once()
        service.async_engine.dispose.assert_called_once()
        service.sync_engine.dispose.assert_called_once()
        assert service._initialized is False
    
    @pytest.mark.asyncio
    async def test_execute_query(self, mock_settings):
        """Test query execution."""
        service = DatabaseService(mock_settings)
        service._initialized = True
        
        # Mock database
        mock_result = [{"id": 1, "name": "test"}]
        service.database = AsyncMock()
        service.database.fetch_all.return_value = mock_result
        
        result = await service.execute_query("SELECT * FROM test", {"param": "value"})
        
        assert result == mock_result
        service.database.fetch_all.assert_called_once_with("SELECT * FROM test", {"param": "value"})
    
    @pytest.mark.asyncio
    async def test_execute_command(self, mock_settings):
        """Test command execution."""
        service = DatabaseService(mock_settings)
        service._initialized = True
        
        # Mock database
        service.database = AsyncMock()
        service.database.execute.return_value = 1
        
        result = await service.execute_command("INSERT INTO test VALUES (:value)", {"value": "test"})
        
        assert result == 1
        service.database.execute.assert_called_once_with("INSERT INTO test VALUES (:value)", {"value": "test"})
    
    @pytest.mark.asyncio
    async def test_bulk_insert(self, mock_settings):
        """Test bulk insert operation."""
        service = DatabaseService(mock_settings)
        service._initialized = True
        
        # Mock session
        mock_session = AsyncMock()
        mock_session.execute.return_value = Mock()
        
        with patch.object(service, 'get_async_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            records = [
                {"id": 1, "name": "test1"},
                {"id": 2, "name": "test2"}
            ]
            
            result = await service.bulk_insert("test_table", records)
            
            assert result == 2
            mock_session.execute.assert_called()
            mock_session.commit.assert_called()
    
    @pytest.mark.asyncio
    async def test_create_embedding_model(self, mock_settings, sample_embedding_model_data):
        """Test embedding model creation."""
        service = DatabaseService(mock_settings)
        
        # Mock session
        mock_session = AsyncMock()
        mock_session.execute.return_value.fetchone.return_value = None  # No existing model
        mock_session.flush = AsyncMock()
        mock_session.commit = AsyncMock()
        
        # Mock model creation
        mock_model = Mock()
        mock_model.id = uuid4()
        
        with patch.object(service, 'get_async_session') as mock_get_session, \
             patch('src.services.database_service.EmbeddingModel') as mock_model_class:
            
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            mock_model_class.return_value = mock_model
            
            # Mock final fetch
            mock_session.execute.return_value.fetchone.side_effect = [
                None,  # First call - no existing model
                sample_embedding_model_data  # Second call - return created model
            ]
            
            result = await service.create_embedding_model(**sample_embedding_model_data)
            
            assert isinstance(result, EmbeddingModel)
            mock_session.add.assert_called_once_with(mock_model)
            mock_session.commit.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_embedding_model(self, mock_settings, sample_embedding_model_data):
        """Test embedding model retrieval."""
        service = DatabaseService(mock_settings)
        
        # Mock session
        mock_session = AsyncMock()
        mock_session.execute.return_value.fetchone.return_value = sample_embedding_model_data
        
        with patch.object(service, 'get_async_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            result = await service.get_embedding_model("test-model")
            
            assert result is not None
            assert result.name == "test-model"
            mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_embedding_model_not_found(self, mock_settings):
        """Test embedding model retrieval when not found."""
        service = DatabaseService(mock_settings)
        
        # Mock session
        mock_session = AsyncMock()
        mock_session.execute.return_value.fetchone.return_value = None
        
        with patch.object(service, 'get_async_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            result = await service.get_embedding_model("nonexistent-model")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_search_similar_chunks(self, mock_settings, sample_embedding_vector):
        """Test vector similarity search."""
        service = DatabaseService(mock_settings)
        
        # Mock search results
        mock_results = [
            {
                "embedding_id": str(uuid4()),
                "chunk_id": str(uuid4()),
                "document_id": str(uuid4()),
                "content": "Test chunk content",
                "chunk_index": 0,
                "section_title": "Test Section",
                "page_number": 1,
                "document_title": "Test Document",
                "document_type": "text",
                "source_url": None,
                "tags": ["test"],
                "custom_metadata": {},
                "similarity": 0.85,
                "distance": 0.15
            }
        ]
        
        with patch.object(service, 'execute_query', return_value=mock_results):
            results = await service.search_similar_chunks(
                query_embedding=sample_embedding_vector,
                model_id=str(uuid4()),
                limit=10,
                threshold=0.7,
                distance_metric="cosine"
            )
            
            assert len(results) == 1
            assert results[0]["similarity"] == 0.85
            assert results[0]["content"] == "Test chunk content"
            assert "embedding_id" in results[0]
            assert "chunk_id" in results[0]
            assert "document_id" in results[0]
    
    @pytest.mark.asyncio
    async def test_search_similar_chunks_with_filters(self, mock_settings, sample_embedding_vector):
        """Test vector similarity search with filters."""
        service = DatabaseService(mock_settings)
        
        filters = {
            "document_types": ["text", "pdf"],
            "user_id": "test_user_123",
            "tags": ["important"]
        }
        
        with patch.object(service, 'execute_query', return_value=[]) as mock_execute:
            await service.search_similar_chunks(
                query_embedding=sample_embedding_vector,
                model_id=str(uuid4()),
                limit=5,
                threshold=0.8,
                distance_metric="cosine",
                filters=filters
            )
            
            # Verify query was called with filters
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args
            query = call_args[0][0]
            params = call_args[0][1]
            
            assert "d.document_type IN" in query
            assert "d.user_id = :user_id" in query
            assert "d.tags && :tags" in query
            assert params["user_id"] == "test_user_123"
            assert params["tags"] == ["important"]
    
    @pytest.mark.asyncio
    async def test_search_different_distance_metrics(self, mock_settings, sample_embedding_vector):
        """Test vector search with different distance metrics."""
        service = DatabaseService(mock_settings)
        
        with patch.object(service, 'execute_query', return_value=[]):
            # Test L2 distance
            await service.search_similar_chunks(
                query_embedding=sample_embedding_vector,
                model_id=str(uuid4()),
                distance_metric="l2"
            )
            
            # Test inner product
            await service.search_similar_chunks(
                query_embedding=sample_embedding_vector,
                model_id=str(uuid4()),
                distance_metric="inner_product"
            )
            
            # Test invalid metric
            with pytest.raises(ValueError):
                await service.search_similar_chunks(
                    query_embedding=sample_embedding_vector,
                    model_id=str(uuid4()),
                    distance_metric="invalid_metric"
                )
    
    @pytest.mark.asyncio
    async def test_log_search_query(self, mock_settings):
        """Test search query logging."""
        service = DatabaseService(mock_settings)
        
        # Mock session
        mock_session = AsyncMock()
        mock_query = Mock()
        mock_query.id = uuid4()
        
        with patch.object(service, 'get_async_session') as mock_get_session, \
             patch('src.services.database_service.SearchQuery', return_value=mock_query):
            
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            result = await service.log_search_query(
                query_text="test query",
                query_type="semantic",
                model_id=str(uuid4()),
                results_count=5,
                execution_time_ms=150.5
            )
            
            assert result == str(mock_query.id)
            mock_session.add.assert_called_once_with(mock_query)
            mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_database_stats(self, mock_settings):
        """Test database statistics retrieval."""
        service = DatabaseService(mock_settings)
        
        # Mock query results
        mock_doc_stats = [{"total_documents": 100, "unique_document_types": 5, "total_file_size": 1000000, "avg_word_count": 500}]
        mock_chunk_stats = [{"total_chunks": 500, "avg_chunk_word_count": 100, "avg_chunk_char_count": 600}]
        mock_embedding_stats = [{"model_name": "test-model", "embedding_count": 500, "dimension": 384}]
        mock_search_stats = [{"total_searches": 50, "avg_execution_time": 100.5, "avg_results_count": 8, "cache_hits": 10}]
        mock_collection_stats = [{"total_collections": 10, "total_documents_in_collections": 80, "avg_documents_per_collection": 8}]
        
        with patch.object(service, 'execute_query') as mock_execute:
            mock_execute.side_effect = [
                mock_doc_stats,
                mock_chunk_stats,
                mock_embedding_stats,
                mock_search_stats,
                mock_collection_stats
            ]
            
            stats = await service.get_database_stats()
            
            assert "documents" in stats
            assert "chunks" in stats
            assert "embeddings" in stats
            assert "searches_24h" in stats
            assert "collections" in stats
            assert "timestamp" in stats
            
            assert stats["documents"]["total_documents"] == 100
            assert stats["chunks"]["total_chunks"] == 500
            assert len(stats["embeddings"]) == 1
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_settings):
        """Test health check when system is healthy."""
        service = DatabaseService(mock_settings)
        service._initialized = True
        
        # Mock successful queries
        with patch.object(service, 'execute_query') as mock_execute, \
             patch.object(service, 'get_database_stats') as mock_stats:
            
            mock_execute.side_effect = [
                [{"test": 1}],  # Basic connectivity test
                []  # pgvector test
            ]
            mock_stats.return_value = {
                "documents": {"total_documents": 100},
                "chunks": {"total_chunks": 500},
                "embeddings": [{"embedding_count": 500}]
            }
            
            # Mock engine pool
            service.async_engine = Mock()
            service.async_engine.pool = Mock()
            service.async_engine.pool.size.return_value = 10
            service.async_engine.pool.checkedin.return_value = 8
            service.async_engine.pool.checkedout.return_value = 2
            service.async_engine.pool.overflow.return_value = 0
            service.async_engine.pool.invalid.return_value = 0
            
            health = await service.health_check()
            
            assert health["status"] == "healthy"
            assert health["details"]["connectivity"] == "ok"
            assert health["details"]["pgvector"] == "ok"
            assert "connection_pool" in health["details"]
            assert "stats" in health["details"]
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, mock_settings):
        """Test health check when system is unhealthy."""
        service = DatabaseService(mock_settings)
        
        with patch.object(service, 'execute_query', side_effect=Exception("Connection failed")):
            health = await service.health_check()
            
            assert health["status"] == "unhealthy"
            assert "error" in health["details"]
    
    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, mock_settings):
        """Test old data cleanup."""
        service = DatabaseService(mock_settings)
        
        with patch.object(service, 'execute_command') as mock_execute:
            mock_execute.side_effect = [10, 5]  # Deleted queries and embeddings
            
            result = await service.cleanup_old_data(days=30)
            
            assert result["deleted_queries"] == 10
            assert result["deleted_embeddings"] == 5
            assert mock_execute.call_count == 2


class TestDatabaseServiceIntegration:
    """Integration tests for database service."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, mock_settings):
        """Test complete workflow from model creation to search."""
        service = DatabaseService(mock_settings)
        
        # Mock all dependencies
        with patch.multiple(
            service,
            initialize=AsyncMock(return_value=True),
            create_embedding_model=AsyncMock(),
            search_similar_chunks=AsyncMock(return_value=[]),
            log_search_query=AsyncMock(return_value="query_id")
        ):
            # Initialize service
            await service.initialize()
            
            # Create embedding model
            await service.create_embedding_model(
                name="test-model",
                provider="sentence-transformers",
                dimension=384,
                max_sequence_length=512
            )
            
            # Perform search
            results = await service.search_similar_chunks(
                query_embedding=[0.1] * 384,
                model_id="model_id",
                limit=10
            )
            
            # Log query
            query_id = await service.log_search_query(
                query_text="test query",
                query_type="semantic",
                model_id="model_id",
                results_count=len(results),
                execution_time_ms=100.0
            )
            
            assert query_id == "query_id"


class TestErrorHandling:
    """Test error handling in database service."""
    
    @pytest.mark.asyncio
    async def test_query_execution_error(self, mock_settings):
        """Test handling of query execution errors."""
        service = DatabaseService(mock_settings)
        service._initialized = True
        service.database = AsyncMock()
        service.database.fetch_all.side_effect = Exception("Query failed")
        
        with pytest.raises(Exception) as exc_info:
            await service.execute_query("INVALID QUERY")
        
        assert "Query failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_bulk_insert_error(self, mock_settings):
        """Test handling of bulk insert errors."""
        service = DatabaseService(mock_settings)
        service._initialized = True
        
        # Mock session that raises exception
        mock_session = AsyncMock()
        mock_session.execute.side_effect = Exception("Insert failed")
        
        with patch.object(service, 'get_async_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            
            records = [{"id": 1, "name": "test"}]
            
            with pytest.raises(Exception) as exc_info:
                await service.bulk_insert("test_table", records)
            
            assert "Insert failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_vector_search_error(self, mock_settings):
        """Test handling of vector search errors."""
        service = DatabaseService(mock_settings)
        
        with patch.object(service, 'execute_query', side_effect=Exception("Search failed")):
            with pytest.raises(Exception) as exc_info:
                await service.search_similar_chunks(
                    query_embedding=[0.1] * 384,
                    model_id="model_id"
                )
            
            assert "Search failed" in str(exc_info.value)


class TestGlobalServiceManagement:
    """Test global service management functions."""
    
    @pytest.mark.asyncio
    async def test_get_database_service(self):
        """Test getting global database service."""
        from src.services.database_service import get_database_service, _db_service
        
        # Reset global service
        import src.services.database_service
        src.services.database_service._db_service = None
        
        with patch.object(DatabaseService, 'initialize', return_value=True):
            service = await get_database_service()
            
            assert service is not None
            assert isinstance(service, DatabaseService)
            
            # Second call should return same instance
            service2 = await get_database_service()
            assert service is service2
    
    @pytest.mark.asyncio
    async def test_close_database_service(self):
        """Test closing global database service."""
        from src.services.database_service import close_database_service
        import src.services.database_service
        
        # Set up mock service
        mock_service = AsyncMock()
        src.services.database_service._db_service = mock_service
        
        await close_database_service()
        
        mock_service.disconnect.assert_called_once()
        assert src.services.database_service._db_service is None


if __name__ == "__main__":
    pytest.main([__file__])


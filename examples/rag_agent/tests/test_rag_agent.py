"""
Tests for PydanticAI RAG Agent.

This module tests the core RAG agent functionality including
query processing, document retrieval, and multi-LLM support.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4
from datetime import datetime

from src.agents.rag_agent import PydanticRAGAgent, RAGQuery, RAGResponse, RAGContext
from src.models.document import Document, DocumentChunk, DocumentType, DocumentMetadata
from config.settings import RAGSettings


@pytest.fixture
def mock_settings():
    """Mock RAG settings for testing."""
    settings = RAGSettings()
    settings.default_provider = "openai"
    settings.openai_api_key = "test-key"
    return settings


@pytest.fixture
def sample_document():
    """Sample document for testing."""
    return Document(
        content="This is a test document about artificial intelligence and machine learning.",
        document_type=DocumentType.TEXT,
        source_path="/test/document.txt",
        metadata=DocumentMetadata(
            title="Test Document",
            author="Test Author",
            tags=["ai", "ml", "test"]
        )
    )


@pytest.fixture
def sample_chunks():
    """Sample document chunks for testing."""
    doc_id = uuid4()
    return [
        DocumentChunk(
            document_id=doc_id,
            content="This is about artificial intelligence.",
            chunk_metadata={
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 40,
                "page_number": 1
            },
            document_metadata=DocumentMetadata(title="Test Doc"),
            similarity_score=0.9
        ),
        DocumentChunk(
            document_id=doc_id,
            content="Machine learning is a subset of AI.",
            chunk_metadata={
                "chunk_index": 1,
                "start_char": 40,
                "end_char": 80,
                "page_number": 1
            },
            document_metadata=DocumentMetadata(title="Test Doc"),
            similarity_score=0.8
        )
    ]


class TestRAGAgent:
    """Test RAG agent functionality."""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, mock_settings):
        """Test RAG agent initialization."""
        with patch('src.agents.rag_agent.VectorStore') as mock_vector_store, \
             patch('src.agents.rag_agent.LLMProviderFactory') as mock_factory:
            
            mock_factory.return_value.get_model.return_value = Mock()
            
            agent = PydanticRAGAgent(mock_settings)
            
            assert agent.settings == mock_settings
            assert mock_vector_store.called
            assert mock_factory.called
    
    @pytest.mark.asyncio
    async def test_query_processing(self, mock_settings, sample_chunks):
        """Test query processing."""
        with patch('src.agents.rag_agent.VectorStore') as mock_vector_store, \
             patch('src.agents.rag_agent.LLMProviderFactory') as mock_factory:
            
            # Mock vector store search
            mock_vector_store.return_value.search = AsyncMock(return_value=sample_chunks)
            
            # Mock LLM provider
            mock_model = Mock()
            mock_factory.return_value.get_model.return_value = mock_model
            
            # Mock agent run result
            mock_result = Mock()
            mock_result.data = "This is a test response about AI and ML."
            mock_result.all_messages.return_value = []
            
            agent = PydanticRAGAgent(mock_settings)
            
            with patch.object(agent.agent, 'run', return_value=mock_result):
                query = RAGQuery(question="What is artificial intelligence?")
                response = await agent.query(query)
                
                assert isinstance(response, RAGResponse)
                assert response.answer == "This is a test response about AI and ML."
                assert response.confidence > 0
    
    @pytest.mark.asyncio
    async def test_query_with_context(self, mock_settings):
        """Test query processing with context."""
        with patch('src.agents.rag_agent.VectorStore'), \
             patch('src.agents.rag_agent.LLMProviderFactory') as mock_factory:
            
            mock_factory.return_value.get_model.return_value = Mock()
            
            agent = PydanticRAGAgent(mock_settings)
            
            # Mock agent run
            mock_result = Mock()
            mock_result.data = "Response with context"
            mock_result.all_messages.return_value = []
            
            with patch.object(agent.agent, 'run', return_value=mock_result):
                context = RAGContext(
                    user_id="test-user",
                    session_id="test-session",
                    conversation_history=[
                        {"role": "user", "content": "Previous question"},
                        {"role": "assistant", "content": "Previous answer"}
                    ]
                )
                
                query = RAGQuery(
                    question="Follow-up question",
                    context=context
                )
                
                response = await agent.query(query)
                
                assert response.answer == "Response with context"
                assert response.metadata["query_length"] > 0
    
    @pytest.mark.asyncio
    async def test_stream_query(self, mock_settings):
        """Test streaming query response."""
        with patch('src.agents.rag_agent.VectorStore'), \
             patch('src.agents.rag_agent.LLMProviderFactory') as mock_factory:
            
            mock_factory.return_value.get_model.return_value = Mock()
            
            agent = PydanticRAGAgent(mock_settings)
            
            # Mock streaming response
            async def mock_stream():
                yield "This "
                yield "is "
                yield "a "
                yield "stream"
            
            mock_stream_context = AsyncMock()
            mock_stream_context.__aenter__.return_value = mock_stream()
            mock_stream_context.__aexit__.return_value = None
            
            with patch.object(agent.agent, 'run_stream', return_value=mock_stream_context):
                query = RAGQuery(question="Test streaming")
                
                chunks = []
                async for chunk in agent.stream_query(query):
                    chunks.append(chunk)
                
                assert len(chunks) == 4
                assert "".join(chunks) == "This is a stream"
    
    @pytest.mark.asyncio
    async def test_add_documents(self, mock_settings, sample_document):
        """Test adding documents to knowledge base."""
        with patch('src.agents.rag_agent.VectorStore') as mock_vector_store, \
             patch('src.agents.rag_agent.LLMProviderFactory'):
            
            mock_vector_store.return_value.add_documents = AsyncMock(
                return_value={"added_documents": 1, "total_chunks": 5}
            )
            
            agent = PydanticRAGAgent(mock_settings)
            
            result = await agent.add_documents([sample_document])
            
            assert result["added_documents"] == 1
            assert result["total_chunks"] == 5
            mock_vector_store.return_value.add_documents.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_documents(self, mock_settings):
        """Test deleting documents from knowledge base."""
        with patch('src.agents.rag_agent.VectorStore') as mock_vector_store, \
             patch('src.agents.rag_agent.LLMProviderFactory'):
            
            mock_vector_store.return_value.delete_documents = AsyncMock(
                return_value={"deleted_documents": 2, "deleted_chunks": 10}
            )
            
            agent = PydanticRAGAgent(mock_settings)
            
            doc_ids = [str(uuid4()), str(uuid4())]
            result = await agent.delete_documents(doc_ids)
            
            assert result["deleted_documents"] == 2
            assert result["deleted_chunks"] == 10
    
    @pytest.mark.asyncio
    async def test_get_stats(self, mock_settings):
        """Test getting agent statistics."""
        with patch('src.agents.rag_agent.VectorStore') as mock_vector_store, \
             patch('src.agents.rag_agent.LLMProviderFactory'):
            
            mock_vector_store.return_value.get_stats = AsyncMock(
                return_value={"total_chunks": 100, "total_documents": 20}
            )
            
            agent = PydanticRAGAgent(mock_settings)
            
            stats = await agent.get_stats()
            
            assert stats["agent_name"] == mock_settings.agent_name
            assert stats["provider"] == mock_settings.default_provider
            assert stats["vector_store"]["total_chunks"] == 100
    
    @pytest.mark.asyncio
    async def test_switch_provider(self, mock_settings):
        """Test switching LLM provider."""
        with patch('src.agents.rag_agent.VectorStore'), \
             patch('src.agents.rag_agent.LLMProviderFactory') as mock_factory:
            
            # Setup mock factory
            mock_factory.return_value.get_model.return_value = Mock()
            
            # Add anthropic to available providers
            mock_settings.anthropic_api_key = "test-anthropic-key"
            
            agent = PydanticRAGAgent(mock_settings)
            
            # Test successful switch
            result = await agent.switch_provider("anthropic")
            
            assert result is True
            assert agent.settings.default_provider == "anthropic"
    
    @pytest.mark.asyncio
    async def test_switch_provider_invalid(self, mock_settings):
        """Test switching to invalid provider."""
        with patch('src.agents.rag_agent.VectorStore'), \
             patch('src.agents.rag_agent.LLMProviderFactory'):
            
            agent = PydanticRAGAgent(mock_settings)
            
            # Test invalid provider
            result = await agent.switch_provider("invalid_provider")
            
            assert result is False
            assert agent.settings.default_provider == "openai"  # Should remain unchanged
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_settings):
        """Test agent health check."""
        with patch('src.agents.rag_agent.VectorStore') as mock_vector_store, \
             patch('src.agents.rag_agent.LLMProviderFactory'):
            
            mock_vector_store.return_value.get_stats = AsyncMock(
                return_value={"document_count": 50}
            )
            
            agent = PydanticRAGAgent(mock_settings)
            
            # Mock successful query for LLM health check
            with patch.object(agent, 'query', return_value=RAGResponse(answer="Hello")):
                health = await agent.health_check()
                
                assert health["status"] in ["healthy", "degraded"]
                assert "components" in health
                assert "llm" in health["components"]
                assert "vector_store" in health["components"]
    
    def test_calculate_confidence(self, mock_settings):
        """Test confidence calculation."""
        with patch('src.agents.rag_agent.VectorStore'), \
             patch('src.agents.rag_agent.LLMProviderFactory'):
            
            agent = PydanticRAGAgent(mock_settings)
            
            # Test with sources and substantial answer
            confidence = agent._calculate_confidence(
                "What is AI?",
                "Artificial intelligence is a field of computer science that focuses on creating intelligent machines.",
                [Mock(), Mock()]  # Two sources
            )
            
            assert confidence > 0.5
            
            # Test with uncertainty
            confidence = agent._calculate_confidence(
                "What is AI?",
                "I'm not sure about this topic.",
                []
            )
            
            assert confidence < 0.5


class TestRAGQuery:
    """Test RAG query model."""
    
    def test_rag_query_creation(self):
        """Test RAG query creation."""
        query = RAGQuery(question="What is machine learning?")
        
        assert query.question == "What is machine learning?"
        assert query.context is None
        assert query.retrieval_params is None
        assert query.llm_params is None
    
    def test_rag_query_with_context(self):
        """Test RAG query with context."""
        context = RAGContext(
            user_id="test-user",
            session_id="test-session"
        )
        
        query = RAGQuery(
            question="Follow-up question",
            context=context,
            retrieval_params={"top_k": 10},
            llm_params={"temperature": 0.8}
        )
        
        assert query.context.user_id == "test-user"
        assert query.retrieval_params["top_k"] == 10
        assert query.llm_params["temperature"] == 0.8


class TestRAGResponse:
    """Test RAG response model."""
    
    def test_rag_response_creation(self, sample_chunks):
        """Test RAG response creation."""
        response = RAGResponse(
            answer="This is about AI and ML.",
            sources=sample_chunks,
            confidence=0.85,
            metadata={"model": "gpt-4"}
        )
        
        assert response.answer == "This is about AI and ML."
        assert len(response.sources) == 2
        assert response.confidence == 0.85
        assert response.metadata["model"] == "gpt-4"
        assert isinstance(response.timestamp, datetime)


class TestRAGContext:
    """Test RAG context."""
    
    def test_rag_context_creation(self):
        """Test RAG context creation."""
        context = RAGContext(
            user_id="test-user",
            session_id="test-session",
            conversation_history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            metadata={"source": "web"}
        )
        
        assert context.user_id == "test-user"
        assert context.session_id == "test-session"
        assert len(context.conversation_history) == 2
        assert context.metadata["source"] == "web"
    
    def test_rag_context_defaults(self):
        """Test RAG context with defaults."""
        context = RAGContext()
        
        assert context.user_id is None
        assert context.session_id is None
        assert context.conversation_history == []
        assert context.metadata == {}


class TestErrorHandling:
    """Test error handling in RAG agent."""
    
    @pytest.mark.asyncio
    async def test_query_error_handling(self, mock_settings):
        """Test query error handling."""
        with patch('src.agents.rag_agent.VectorStore'), \
             patch('src.agents.rag_agent.LLMProviderFactory'):
            
            agent = PydanticRAGAgent(mock_settings)
            
            # Mock agent run to raise exception
            with patch.object(agent.agent, 'run', side_effect=Exception("Test error")):
                query = RAGQuery(question="Test question")
                response = await agent.query(query)
                
                assert "error" in response.answer.lower()
                assert response.confidence == 0.0
                assert "error" in response.metadata
    
    @pytest.mark.asyncio
    async def test_add_documents_error_handling(self, mock_settings, sample_document):
        """Test add documents error handling."""
        with patch('src.agents.rag_agent.VectorStore') as mock_vector_store, \
             patch('src.agents.rag_agent.LLMProviderFactory'):
            
            mock_vector_store.return_value.add_documents = AsyncMock(
                side_effect=Exception("Vector store error")
            )
            
            agent = PydanticRAGAgent(mock_settings)
            
            result = await agent.add_documents([sample_document])
            
            assert result["added_documents"] == 0
            assert "error" in result
    
    @pytest.mark.asyncio
    async def test_health_check_error_handling(self, mock_settings):
        """Test health check error handling."""
        with patch('src.agents.rag_agent.VectorStore') as mock_vector_store, \
             patch('src.agents.rag_agent.LLMProviderFactory'):
            
            # Mock vector store to raise exception
            mock_vector_store.return_value.get_stats = AsyncMock(
                side_effect=Exception("Vector store down")
            )
            
            agent = PydanticRAGAgent(mock_settings)
            
            health = await agent.health_check()
            
            assert health["status"] in ["unhealthy", "degraded"]
            assert health["components"]["vector_store"]["status"] == "unhealthy"


if __name__ == "__main__":
    pytest.main([__file__])


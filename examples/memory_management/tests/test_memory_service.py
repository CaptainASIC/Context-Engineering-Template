"""
Tests for Memory Service.

This module tests the memory service including conversation context,
user preferences, entity relationships, and fact storage functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone, timedelta
from uuid import uuid4
import json

from src.services.memory_service import MemoryService, get_memory_service, close_memory_service
from src.models.memory_models import (
    ConversationMemory, UserPreference, EntityMemory, FactMemory,
    MemoryQuery, MemorySearchResult, MemoryType, MemorySource
)
from config.settings import MemoryManagementSettings, create_test_settings


@pytest.fixture
def mock_settings():
    """Mock memory management settings for testing."""
    return create_test_settings()


@pytest.fixture
def sample_conversation_data():
    """Sample conversation data for testing."""
    return {
        "user_id": "test_user_123",
        "conversation_id": "conv_456",
        "message": "I'm interested in learning about machine learning algorithms.",
        "role": "user",
        "metadata": {"source": "chat_interface"}
    }


@pytest.fixture
def sample_preference_data():
    """Sample user preference data for testing."""
    return {
        "user_id": "test_user_123",
        "preference_type": "communication_style",
        "preference_value": "detailed_explanations",
        "confidence": 0.9,
        "source": "inferred"
    }


@pytest.fixture
def sample_entity_data():
    """Sample entity data for testing."""
    return {
        "user_id": "test_user_123",
        "entity_name": "Python",
        "entity_type": "technology",
        "properties": {
            "category": "programming_language",
            "difficulty": "intermediate",
            "use_cases": ["web_development", "data_science", "automation"]
        },
        "relationships": [
            {
                "target_entity": "Django",
                "relationship_type": "framework_for",
                "properties": {"domain": "web_development"}
            }
        ]
    }


@pytest.fixture
def sample_fact_data():
    """Sample fact data for testing."""
    return {
        "user_id": "test_user_123",
        "fact": "User has 5 years of experience in software development",
        "fact_type": "professional",
        "confidence": 1.0,
        "source": "explicit"
    }


@pytest.fixture
def mock_mem0_memory():
    """Mock mem0 Memory instance."""
    mock_memory = Mock()
    
    # Mock add method
    mock_memory.add.return_value = {"id": str(uuid4())}
    
    # Mock search method
    mock_memory.search.return_value = [
        {
            "id": str(uuid4()),
            "memory": "Test memory content",
            "score": 0.85,
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "memory_type": "conversation"
            }
        }
    ]
    
    # Mock get_all method
    mock_memory.get_all.return_value = []
    
    # Mock update method
    mock_memory.update.return_value = True
    
    # Mock delete method
    mock_memory.delete.return_value = True
    
    return mock_memory


class TestMemoryService:
    """Test memory service functionality."""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, mock_settings):
        """Test memory service initialization."""
        service = MemoryService(mock_settings)
        
        assert service.settings == mock_settings
        assert service.memory is None
        assert len(service._collections) == 0
        assert len(service._memory_cache) == 0
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_settings):
        """Test successful memory service initialization."""
        with patch('src.services.memory_service.Memory') as mock_memory_class:
            mock_memory_instance = Mock()
            mock_memory_class.return_value = mock_memory_instance
            
            service = MemoryService(mock_settings)
            result = await service.initialize()
            
            assert result is True
            assert service.memory is not None
            mock_memory_class.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_failure(self, mock_settings):
        """Test failed memory service initialization."""
        with patch('src.services.memory_service.Memory', side_effect=Exception("Init failed")):
            service = MemoryService(mock_settings)
            result = await service.initialize()
            
            assert result is False
            assert service.memory is None
    
    @pytest.mark.asyncio
    async def test_build_mem0_config(self, mock_settings):
        """Test mem0 configuration building."""
        service = MemoryService(mock_settings)
        config = service._build_mem0_config()
        
        assert "llm" in config
        assert "embedder" in config
        assert "vector_store" in config
        assert "version" in config
        
        assert config["llm"]["provider"] == mock_settings.llm.provider
        assert config["embedder"]["provider"] == mock_settings.embedding.provider
    
    @pytest.mark.asyncio
    async def test_add_conversation_memory(self, mock_settings, sample_conversation_data, mock_mem0_memory):
        """Test adding conversation memory."""
        service = MemoryService(mock_settings)
        service.memory = mock_mem0_memory
        
        memory_id = await service.add_conversation_memory(**sample_conversation_data)
        
        assert memory_id is not None
        mock_mem0_memory.add.assert_called_once()
        assert service._stats["memories_created"] == 1
        
        # Check call arguments
        call_args = mock_mem0_memory.add.call_args
        assert call_args[1]["user_id"] == sample_conversation_data["user_id"]
        assert len(call_args[1]["messages"]) == 1
        assert call_args[1]["messages"][0]["role"] == sample_conversation_data["role"]
        assert call_args[1]["messages"][0]["content"] == sample_conversation_data["message"]
    
    @pytest.mark.asyncio
    async def test_add_user_preference(self, mock_settings, sample_preference_data, mock_mem0_memory):
        """Test adding user preference."""
        service = MemoryService(mock_settings)
        service.memory = mock_mem0_memory
        
        memory_id = await service.add_user_preference(**sample_preference_data)
        
        assert memory_id is not None
        mock_mem0_memory.add.assert_called_once()
        assert service._stats["memories_created"] == 1
        
        # Check preference message format
        call_args = mock_mem0_memory.add.call_args
        messages = call_args[1]["messages"]
        assert len(messages) == 1
        assert "prefers" in messages[0]["content"].lower()
        assert sample_preference_data["preference_type"] in messages[0]["content"]
    
    @pytest.mark.asyncio
    async def test_add_entity_memory(self, mock_settings, sample_entity_data, mock_mem0_memory):
        """Test adding entity memory."""
        service = MemoryService(mock_settings)
        service.memory = mock_mem0_memory
        
        memory_id = await service.add_entity_memory(**sample_entity_data)
        
        assert memory_id is not None
        mock_mem0_memory.add.assert_called_once()
        assert service._stats["memories_created"] == 1
        
        # Check entity description format
        call_args = mock_mem0_memory.add.call_args
        messages = call_args[1]["messages"]
        assert len(messages) == 1
        assert sample_entity_data["entity_name"] in messages[0]["content"]
        assert sample_entity_data["entity_type"] in messages[0]["content"]
    
    @pytest.mark.asyncio
    async def test_add_fact_memory(self, mock_settings, sample_fact_data, mock_mem0_memory):
        """Test adding fact memory."""
        service = MemoryService(mock_settings)
        service.memory = mock_mem0_memory
        
        memory_id = await service.add_fact_memory(**sample_fact_data)
        
        assert memory_id is not None
        mock_mem0_memory.add.assert_called_once()
        assert service._stats["memories_created"] == 1
        
        # Check fact message format
        call_args = mock_mem0_memory.add.call_args
        messages = call_args[1]["messages"]
        assert len(messages) == 1
        assert "Fact:" in messages[0]["content"]
        assert sample_fact_data["fact"] in messages[0]["content"]
    
    @pytest.mark.asyncio
    async def test_search_memories(self, mock_settings, mock_mem0_memory):
        """Test memory search functionality."""
        service = MemoryService(mock_settings)
        service.memory = mock_mem0_memory
        
        # Setup mock search results
        mock_search_results = [
            {
                "id": str(uuid4()),
                "memory": "Test conversation about Python programming",
                "score": 0.9,
                "metadata": {
                    "conversation_id": "conv_123",
                    "role": "user",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            },
            {
                "id": str(uuid4()),
                "memory": "User prefers detailed explanations",
                "score": 0.8,
                "metadata": {
                    "preference_type": "communication_style",
                    "preference_value": "detailed",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
        ]
        mock_mem0_memory.search.return_value = mock_search_results
        
        results = await service.search_memories(
            user_id="test_user",
            query="Python programming",
            limit=10,
            threshold=0.7
        )
        
        assert len(results) == 2
        assert all(isinstance(result, MemorySearchResult) for result in results)
        assert results[0].similarity_score == 0.9
        assert results[1].similarity_score == 0.8
        
        # Check that results are sorted by similarity score
        assert results[0].similarity_score >= results[1].similarity_score
        
        mock_mem0_memory.search.assert_called_once_with(
            query="Python programming",
            user_id="test_user",
            limit=10
        )
    
    @pytest.mark.asyncio
    async def test_search_memories_with_filters(self, mock_settings, mock_mem0_memory):
        """Test memory search with filters."""
        service = MemoryService(mock_settings)
        service.memory = mock_mem0_memory
        
        # Setup mock search results with different memory types
        mock_search_results = [
            {
                "id": str(uuid4()),
                "memory": "Conversation about Python",
                "score": 0.9,
                "metadata": {
                    "conversation_id": "conv_123",
                    "role": "user",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            },
            {
                "id": str(uuid4()),
                "memory": "User prefers Python",
                "score": 0.8,
                "metadata": {
                    "preference_type": "language",
                    "preference_value": "python",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
        ]
        mock_mem0_memory.search.return_value = mock_search_results
        
        # Search with memory type filter
        results = await service.search_memories(
            user_id="test_user",
            query="Python",
            memory_types=[MemoryType.CONVERSATION],
            limit=10
        )
        
        # Should only return conversation memories
        assert len(results) == 1
        assert results[0].memory_type == MemoryType.CONVERSATION
    
    @pytest.mark.asyncio
    async def test_search_memories_threshold_filter(self, mock_settings, mock_mem0_memory):
        """Test memory search with similarity threshold."""
        service = MemoryService(mock_settings)
        service.memory = mock_mem0_memory
        
        # Setup mock search results with different scores
        mock_search_results = [
            {
                "id": str(uuid4()),
                "memory": "High relevance result",
                "score": 0.9,
                "metadata": {"timestamp": datetime.now(timezone.utc).isoformat()}
            },
            {
                "id": str(uuid4()),
                "memory": "Low relevance result",
                "score": 0.5,
                "metadata": {"timestamp": datetime.now(timezone.utc).isoformat()}
            }
        ]
        mock_mem0_memory.search.return_value = mock_search_results
        
        # Search with high threshold
        results = await service.search_memories(
            user_id="test_user",
            query="test query",
            threshold=0.8
        )
        
        # Should only return high-scoring results
        assert len(results) == 1
        assert results[0].similarity_score == 0.9
    
    @pytest.mark.asyncio
    async def test_get_conversation_context(self, mock_settings, mock_mem0_memory):
        """Test getting conversation context."""
        service = MemoryService(mock_settings)
        service.memory = mock_mem0_memory
        
        # Setup mock search results for conversation
        conversation_id = "conv_123"
        mock_search_results = [
            {
                "id": str(uuid4()),
                "memory": "Hello, how can I help you?",
                "score": 1.0,
                "metadata": {
                    "conversation_id": conversation_id,
                    "role": "assistant",
                    "timestamp": "2024-01-01T10:00:00Z"
                }
            },
            {
                "id": str(uuid4()),
                "memory": "I need help with Python programming",
                "score": 1.0,
                "metadata": {
                    "conversation_id": conversation_id,
                    "role": "user",
                    "timestamp": "2024-01-01T10:01:00Z"
                }
            }
        ]
        mock_mem0_memory.search.return_value = mock_search_results
        
        context = await service.get_conversation_context(
            user_id="test_user",
            conversation_id=conversation_id,
            max_messages=10
        )
        
        assert len(context) == 2
        assert context[0]["role"] == "assistant"
        assert context[1]["role"] == "user"
        
        # Check that messages are sorted by timestamp
        assert context[0]["timestamp"] <= context[1]["timestamp"]
    
    @pytest.mark.asyncio
    async def test_get_user_preferences(self, mock_settings, mock_mem0_memory):
        """Test getting user preferences."""
        service = MemoryService(mock_settings)
        service.memory = mock_mem0_memory
        
        # Setup mock search results for preferences
        mock_search_results = [
            {
                "id": str(uuid4()),
                "memory": "User prefers detailed explanations",
                "score": 0.9,
                "metadata": {
                    "preference_type": "communication_style",
                    "preference_value": "detailed",
                    "confidence": 0.9,
                    "source": "inferred",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            },
            {
                "id": str(uuid4()),
                "memory": "User is interested in machine learning",
                "score": 0.8,
                "metadata": {
                    "preference_type": "topics_of_interest",
                    "preference_value": "machine_learning",
                    "confidence": 0.8,
                    "source": "explicit",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
        ]
        mock_mem0_memory.search.return_value = mock_search_results
        
        preferences = await service.get_user_preferences(
            user_id="test_user",
            min_confidence=0.7
        )
        
        assert len(preferences) == 2
        assert "communication_style" in preferences
        assert "topics_of_interest" in preferences
        
        assert preferences["communication_style"]["value"] == "detailed"
        assert preferences["communication_style"]["confidence"] == 0.9
        assert preferences["topics_of_interest"]["value"] == "machine_learning"
    
    @pytest.mark.asyncio
    async def test_get_user_preferences_confidence_filter(self, mock_settings, mock_mem0_memory):
        """Test user preferences with confidence filtering."""
        service = MemoryService(mock_settings)
        service.memory = mock_mem0_memory
        
        # Setup mock search results with different confidence levels
        mock_search_results = [
            {
                "id": str(uuid4()),
                "memory": "High confidence preference",
                "score": 0.9,
                "metadata": {
                    "preference_type": "high_confidence",
                    "preference_value": "value1",
                    "confidence": 0.9,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            },
            {
                "id": str(uuid4()),
                "memory": "Low confidence preference",
                "score": 0.8,
                "metadata": {
                    "preference_type": "low_confidence",
                    "preference_value": "value2",
                    "confidence": 0.4,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
        ]
        mock_mem0_memory.search.return_value = mock_search_results
        
        preferences = await service.get_user_preferences(
            user_id="test_user",
            min_confidence=0.7
        )
        
        # Should only return high confidence preferences
        assert len(preferences) == 1
        assert "high_confidence" in preferences
        assert "low_confidence" not in preferences
    
    @pytest.mark.asyncio
    async def test_update_memory(self, mock_settings, mock_mem0_memory):
        """Test memory update functionality."""
        service = MemoryService(mock_settings)
        service.memory = mock_mem0_memory
        
        memory_id = str(uuid4())
        updates = {"confidence": 0.9, "verified": True}
        
        result = await service.update_memory(
            memory_id=memory_id,
            user_id="test_user",
            updates=updates
        )
        
        assert result is True
        mock_mem0_memory.update.assert_called_once_with(
            memory_id=memory_id,
            data=updates
        )
        assert service._stats["memories_updated"] == 1
    
    @pytest.mark.asyncio
    async def test_delete_memory(self, mock_settings, mock_mem0_memory):
        """Test memory deletion."""
        service = MemoryService(mock_settings)
        service.memory = mock_mem0_memory
        
        memory_id = str(uuid4())
        
        result = await service.delete_memory(
            memory_id=memory_id,
            user_id="test_user"
        )
        
        assert result is True
        mock_mem0_memory.delete.assert_called_once_with(memory_id=memory_id)
        assert service._stats["memories_deleted"] == 1
    
    @pytest.mark.asyncio
    async def test_get_user_memory_summary(self, mock_settings, mock_mem0_memory):
        """Test user memory summary generation."""
        service = MemoryService(mock_settings)
        service.memory = mock_mem0_memory
        
        # Setup mock memories
        mock_memories = [
            {
                "id": str(uuid4()),
                "metadata": {
                    "conversation_id": "conv_1",
                    "role": "user",
                    "timestamp": "2024-01-01T10:00:00Z"
                }
            },
            {
                "id": str(uuid4()),
                "metadata": {
                    "preference_type": "communication_style",
                    "preference_value": "detailed",
                    "timestamp": "2024-01-01T11:00:00Z"
                }
            },
            {
                "id": str(uuid4()),
                "metadata": {
                    "entity_name": "Python",
                    "entity_type": "technology",
                    "timestamp": "2024-01-01T12:00:00Z"
                }
            }
        ]
        mock_mem0_memory.get_all.return_value = mock_memories
        
        summary = await service.get_user_memory_summary(user_id="test_user")
        
        assert summary["total_memories"] == 3
        assert len(summary["memory_types"]) > 0
        assert summary["preference_count"] == 1
        assert summary["conversation_count"] == 1
        assert "recent_activity" in summary
        assert "top_entities" in summary
        
        mock_mem0_memory.get_all.assert_called_once_with(user_id="test_user")
    
    @pytest.mark.asyncio
    async def test_memory_caching(self, mock_settings):
        """Test memory caching functionality."""
        service = MemoryService(mock_settings)
        
        # Test cache storage
        memory_id = str(uuid4())
        memory_record = ConversationMemory(
            user_id="test_user",
            conversation_id="conv_123",
            message="Test message",
            role="user"
        )
        
        service._cache_memory(memory_id, memory_record)
        
        # Test cache retrieval
        cached_memory = service._get_cached_memory(memory_id)
        assert cached_memory is not None
        assert cached_memory.message == "Test message"
        assert service._stats["cache_hits"] == 1
        
        # Test cache miss
        non_existent_memory = service._get_cached_memory("non_existent")
        assert non_existent_memory is None
        assert service._stats["cache_misses"] == 1
        
        # Test cache clearing
        service._clear_cache(memory_id)
        cleared_memory = service._get_cached_memory(memory_id)
        assert cleared_memory is None
        assert service._stats["cache_misses"] == 2
    
    @pytest.mark.asyncio
    async def test_memory_type_inference(self, mock_settings):
        """Test memory type inference from metadata."""
        service = MemoryService(mock_settings)
        
        # Test conversation memory
        conv_metadata = {"conversation_id": "conv_123", "role": "user"}
        assert service._infer_memory_type(conv_metadata) == MemoryType.CONVERSATION
        
        # Test preference memory
        pref_metadata = {"preference_type": "communication_style"}
        assert service._infer_memory_type(pref_metadata) == MemoryType.USER_PREFERENCE
        
        # Test entity memory
        entity_metadata = {"entity_name": "Python", "entity_type": "technology"}
        assert service._infer_memory_type(entity_metadata) == MemoryType.ENTITY
        
        # Test fact memory
        fact_metadata = {"fact": "User has 5 years experience"}
        assert service._infer_memory_type(fact_metadata) == MemoryType.FACT
        
        # Test unknown memory
        unknown_metadata = {"some_other_field": "value"}
        assert service._infer_memory_type(unknown_metadata) == "unknown"
    
    @pytest.mark.asyncio
    async def test_filter_application(self, mock_settings):
        """Test custom filter application."""
        service = MemoryService(mock_settings)
        
        metadata = {
            "conversation_id": "conv_123",
            "role": "user",
            "tags": ["important", "technical"]
        }
        
        # Test matching filters
        filters = {"role": "user", "conversation_id": "conv_123"}
        assert service._apply_filters(metadata, filters) is True
        
        # Test non-matching filters
        filters = {"role": "assistant"}
        assert service._apply_filters(metadata, filters) is False
        
        # Test list filters
        filters = {"tags": ["important"]}
        assert service._apply_filters(metadata, filters) is False  # tags not in list
        
        # Test missing field
        filters = {"non_existent_field": "value"}
        assert service._apply_filters(metadata, filters) is False
    
    @pytest.mark.asyncio
    async def test_get_statistics(self, mock_settings):
        """Test statistics retrieval."""
        service = MemoryService(mock_settings)
        
        # Simulate some operations
        service._stats["memories_created"] = 10
        service._stats["memories_retrieved"] = 50
        service._stats["cache_hits"] = 30
        service._stats["cache_misses"] = 20
        
        stats = await service.get_statistics()
        
        assert stats["memories_created"] == 10
        assert stats["memories_retrieved"] == 50
        assert stats["cache_hits"] == 30
        assert stats["cache_misses"] == 20
        assert stats["cache_hit_rate"] == 0.6  # 30/(30+20)
        assert "timestamp" in stats
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_settings, mock_mem0_memory):
        """Test health check when service is healthy."""
        service = MemoryService(mock_settings)
        service.memory = mock_mem0_memory
        
        mock_mem0_memory.get_all.return_value = []
        
        health = await service.health_check()
        
        assert health["status"] == "healthy"
        assert health["details"]["mem0"] == "ok"
        assert "statistics" in health["details"]
        assert "timestamp" in health
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, mock_settings):
        """Test health check when service is unhealthy."""
        service = MemoryService(mock_settings)
        # Don't initialize memory (service.memory remains None)
        
        health = await service.health_check()
        
        assert health["status"] == "unhealthy"
        assert health["details"]["mem0"] == "not initialized"
    
    @pytest.mark.asyncio
    async def test_health_check_error(self, mock_settings, mock_mem0_memory):
        """Test health check when there's an error."""
        service = MemoryService(mock_settings)
        service.memory = mock_mem0_memory
        
        mock_mem0_memory.get_all.side_effect = Exception("Connection failed")
        
        health = await service.health_check()
        
        assert health["status"] == "unhealthy"
        assert "error" in health["details"]
        assert "Connection failed" in health["details"]["error"]


class TestMemoryServiceIntegration:
    """Integration tests for memory service."""
    
    @pytest.mark.asyncio
    async def test_full_conversation_workflow(self, mock_settings, mock_mem0_memory):
        """Test complete conversation memory workflow."""
        service = MemoryService(mock_settings)
        service.memory = mock_mem0_memory
        
        user_id = "test_user"
        conversation_id = "conv_123"
        
        # Add multiple conversation messages
        messages = [
            {"message": "Hello", "role": "user"},
            {"message": "Hi! How can I help you?", "role": "assistant"},
            {"message": "I need help with Python", "role": "user"},
            {"message": "I'd be happy to help with Python!", "role": "assistant"}
        ]
        
        memory_ids = []
        for msg in messages:
            memory_id = await service.add_conversation_memory(
                user_id=user_id,
                conversation_id=conversation_id,
                **msg
            )
            memory_ids.append(memory_id)
        
        assert len(memory_ids) == 4
        assert service._stats["memories_created"] == 4
        
        # Mock search results for context retrieval
        mock_search_results = [
            {
                "id": memory_ids[i],
                "memory": msg["message"],
                "score": 1.0,
                "metadata": {
                    "conversation_id": conversation_id,
                    "role": msg["role"],
                    "timestamp": f"2024-01-01T10:0{i}:00Z"
                }
            }
            for i, msg in enumerate(messages)
        ]
        mock_mem0_memory.search.return_value = mock_search_results
        
        # Get conversation context
        context = await service.get_conversation_context(
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        assert len(context) == 4
        assert context[0]["content"] == "Hello"
        assert context[-1]["content"] == "I'd be happy to help with Python!"
    
    @pytest.mark.asyncio
    async def test_preference_learning_workflow(self, mock_settings, mock_mem0_memory):
        """Test user preference learning workflow."""
        service = MemoryService(mock_settings)
        service.memory = mock_mem0_memory
        
        user_id = "test_user"
        
        # Add various preferences
        preferences = [
            {
                "preference_type": "communication_style",
                "preference_value": "detailed_explanations",
                "confidence": 0.8,
                "source": "inferred"
            },
            {
                "preference_type": "topics_of_interest",
                "preference_value": "machine_learning",
                "confidence": 0.9,
                "source": "explicit"
            },
            {
                "preference_type": "expertise_level",
                "preference_value": "intermediate",
                "confidence": 0.7,
                "source": "inferred"
            }
        ]
        
        for pref in preferences:
            await service.add_user_preference(user_id=user_id, **pref)
        
        assert service._stats["memories_created"] == 3
        
        # Mock search results for preference retrieval
        mock_search_results = [
            {
                "id": str(uuid4()),
                "memory": f"User prefers {pref['preference_type']}: {pref['preference_value']}",
                "score": 0.9,
                "metadata": {
                    **pref,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
            for pref in preferences
        ]
        mock_mem0_memory.search.return_value = mock_search_results
        
        # Get user preferences
        user_prefs = await service.get_user_preferences(
            user_id=user_id,
            min_confidence=0.6
        )
        
        assert len(user_prefs) == 3
        assert "communication_style" in user_prefs
        assert "topics_of_interest" in user_prefs
        assert "expertise_level" in user_prefs
        
        assert user_prefs["communication_style"]["confidence"] == 0.8
        assert user_prefs["topics_of_interest"]["confidence"] == 0.9


class TestGlobalServiceManagement:
    """Test global service management functions."""
    
    @pytest.mark.asyncio
    async def test_get_memory_service(self):
        """Test getting global memory service."""
        # Reset global service
        import src.services.memory_service
        src.services.memory_service._memory_service = None
        
        with patch.object(MemoryService, 'initialize', return_value=True):
            service = await get_memory_service()
            
            assert service is not None
            assert isinstance(service, MemoryService)
            
            # Second call should return same instance
            service2 = await get_memory_service()
            assert service is service2
    
    @pytest.mark.asyncio
    async def test_close_memory_service(self):
        """Test closing global memory service."""
        import src.services.memory_service
        
        # Set up mock service
        mock_service = Mock()
        src.services.memory_service._memory_service = mock_service
        
        await close_memory_service()
        
        assert src.services.memory_service._memory_service is None


class TestErrorHandling:
    """Test error handling in memory service."""
    
    @pytest.mark.asyncio
    async def test_add_memory_error(self, mock_settings, mock_mem0_memory):
        """Test handling of memory addition errors."""
        service = MemoryService(mock_settings)
        service.memory = mock_mem0_memory
        
        mock_mem0_memory.add.side_effect = Exception("Add failed")
        
        with pytest.raises(Exception) as exc_info:
            await service.add_conversation_memory(
                user_id="test_user",
                conversation_id="conv_123",
                message="Test message",
                role="user"
            )
        
        assert "Add failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_search_memory_error(self, mock_settings, mock_mem0_memory):
        """Test handling of memory search errors."""
        service = MemoryService(mock_settings)
        service.memory = mock_mem0_memory
        
        mock_mem0_memory.search.side_effect = Exception("Search failed")
        
        with pytest.raises(Exception) as exc_info:
            await service.search_memories(
                user_id="test_user",
                query="test query"
            )
        
        assert "Search failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_update_memory_error(self, mock_settings, mock_mem0_memory):
        """Test handling of memory update errors."""
        service = MemoryService(mock_settings)
        service.memory = mock_mem0_memory
        
        mock_mem0_memory.update.side_effect = Exception("Update failed")
        
        result = await service.update_memory(
            memory_id=str(uuid4()),
            user_id="test_user",
            updates={"confidence": 0.9}
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_delete_memory_error(self, mock_settings, mock_mem0_memory):
        """Test handling of memory deletion errors."""
        service = MemoryService(mock_settings)
        service.memory = mock_mem0_memory
        
        mock_mem0_memory.delete.side_effect = Exception("Delete failed")
        
        result = await service.delete_memory(
            memory_id=str(uuid4()),
            user_id="test_user"
        )
        
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__])


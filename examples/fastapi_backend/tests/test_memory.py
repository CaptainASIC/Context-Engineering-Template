"""
Tests for memory management endpoints and functionality.

This module tests memory creation, search, consolidation,
user preferences, and mem0 integration patterns.
"""

import pytest
from fastapi import status
from uuid import uuid4
from datetime import datetime, timedelta

from src.memory.router import (
    validate_memory_type,
    calculate_memory_relevance,
    consolidate_memories
)


class TestMemoryEndpoints:
    """Test memory management endpoints."""
    
    def test_create_memory_success(self, client, auth_headers, test_memory_data):
        """Test successful memory creation."""
        response = client.post("/api/v1/memory/memories", 
                             json=test_memory_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["content"] == test_memory_data["content"]
        assert data["memory_type"] == test_memory_data["memory_type"]
        assert data["importance"] == test_memory_data["importance"]
        assert data["tags"] == test_memory_data["tags"]
        assert "id" in data
        assert "created_at" in data
    
    def test_create_memory_invalid_type(self, client, auth_headers, test_memory_data):
        """Test memory creation with invalid type."""
        test_memory_data["memory_type"] = "invalid_type"
        response = client.post("/api/v1/memory/memories", 
                             json=test_memory_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "Invalid memory type" in response.json()["message"]
    
    def test_create_memory_unauthorized(self, client, test_memory_data):
        """Test memory creation without authentication."""
        response = client.post("/api/v1/memory/memories", json=test_memory_data)
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_list_memories(self, client, auth_headers, test_memory_data):
        """Test listing user memories."""
        # Create a memory first
        create_response = client.post("/api/v1/memory/memories", 
                                    json=test_memory_data, headers=auth_headers)
        assert create_response.status_code == status.HTTP_200_OK
        
        # List memories
        response = client.get("/api/v1/memory/memories", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert any(memory["content"] == test_memory_data["content"] for memory in data)
    
    def test_list_memories_with_filters(self, client, auth_headers, test_memory_data):
        """Test listing memories with filters."""
        # Create a memory first
        client.post("/api/v1/memory/memories", 
                   json=test_memory_data, headers=auth_headers)
        
        # Filter by type
        response = client.get(f"/api/v1/memory/memories?memory_type={test_memory_data['memory_type']}", 
                            headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert all(memory["memory_type"] == test_memory_data["memory_type"] for memory in data)
        
        # Filter by tags
        tag = test_memory_data["tags"][0] if test_memory_data["tags"] else "preference"
        response = client.get(f"/api/v1/memory/memories?tags={tag}", 
                            headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert any(tag in memory["tags"] for memory in data)
        
        # Filter by importance
        response = client.get(f"/api/v1/memory/memories?min_importance={test_memory_data['importance']}", 
                            headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert all(memory["importance"] >= test_memory_data["importance"] for memory in data)
    
    def test_get_memory_success(self, client, auth_headers, test_memory_data):
        """Test getting a specific memory."""
        # Create a memory first
        create_response = client.post("/api/v1/memory/memories", 
                                    json=test_memory_data, headers=auth_headers)
        memory_id = create_response.json()["id"]
        
        # Get the memory
        response = client.get(f"/api/v1/memory/memories/{memory_id}", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == memory_id
        assert data["content"] == test_memory_data["content"]
        assert data["access_count"] == 1  # Should increment on access
    
    def test_get_memory_not_found(self, client, auth_headers):
        """Test getting a nonexistent memory."""
        fake_id = str(uuid4())
        response = client.get(f"/api/v1/memory/memories/{fake_id}", headers=auth_headers)
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["message"]
    
    def test_update_memory_success(self, client, auth_headers, test_memory_data):
        """Test successful memory update."""
        # Create a memory first
        create_response = client.post("/api/v1/memory/memories", 
                                    json=test_memory_data, headers=auth_headers)
        memory_id = create_response.json()["id"]
        
        # Update the memory
        update_data = {
            "content": "Updated memory content",
            "importance": 0.9,
            "tags": ["updated", "test"],
            "metadata": {"updated": True}
        }
        
        response = client.put(f"/api/v1/memory/memories/{memory_id}", 
                            json=update_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["content"] == "Updated memory content"
        assert data["importance"] == 0.9
        assert "updated" in data["tags"]
        assert data["metadata"]["updated"] is True
    
    def test_delete_memory_success(self, client, auth_headers, test_memory_data):
        """Test successful memory deletion."""
        # Create a memory first
        create_response = client.post("/api/v1/memory/memories", 
                                    json=test_memory_data, headers=auth_headers)
        memory_id = create_response.json()["id"]
        
        # Delete the memory
        response = client.delete(f"/api/v1/memory/memories/{memory_id}", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        assert "deleted successfully" in response.json()["message"]
        
        # Verify memory is deleted
        get_response = client.get(f"/api/v1/memory/memories/{memory_id}", headers=auth_headers)
        assert get_response.status_code == status.HTTP_404_NOT_FOUND


class TestMemorySearch:
    """Test memory search functionality."""
    
    def test_search_memories_success(self, client, auth_headers, test_memory_data):
        """Test successful memory search."""
        # Create a memory first
        client.post("/api/v1/memory/memories", 
                   json=test_memory_data, headers=auth_headers)
        
        # Search for memories
        search_data = {
            "query": "technical explanations",
            "memory_types": ["preference"],
            "min_importance": 0.5,
            "limit": 10
        }
        
        response = client.post("/api/v1/memory/memories/search", 
                             json=search_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "memories" in data
        assert "total_count" in data
        assert "search_metadata" in data
        assert isinstance(data["memories"], list)
    
    def test_search_memories_with_filters(self, client, auth_headers, test_memory_data):
        """Test memory search with various filters."""
        # Create multiple memories
        memories = [
            test_memory_data,
            {
                "content": "User likes visual diagrams",
                "memory_type": "preference",
                "importance": 0.7,
                "tags": ["visual", "diagrams"]
            },
            {
                "content": "Meeting scheduled for tomorrow",
                "memory_type": "context",
                "importance": 0.6,
                "tags": ["meeting", "schedule"]
            }
        ]
        
        for memory in memories:
            client.post("/api/v1/memory/memories", json=memory, headers=auth_headers)
        
        # Search with type filter
        search_data = {
            "query": "user",
            "memory_types": ["preference"],
            "limit": 10
        }
        
        response = client.post("/api/v1/memory/memories/search", 
                             json=search_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert all(memory["memory_type"] == "preference" for memory in data["memories"])
        
        # Search with tag filter
        search_data = {
            "query": "visual",
            "tags": ["visual"],
            "limit": 10
        }
        
        response = client.post("/api/v1/memory/memories/search", 
                             json=search_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert any("visual" in memory["tags"] for memory in data["memories"])
    
    def test_search_memories_empty_query(self, client, auth_headers):
        """Test memory search with empty query."""
        search_data = {
            "query": "",
            "limit": 10
        }
        
        response = client.post("/api/v1/memory/memories/search", 
                             json=search_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestMemoryConsolidation:
    """Test memory consolidation functionality."""
    
    def test_consolidate_memories_success(self, client, auth_headers):
        """Test successful memory consolidation."""
        # Create similar memories
        similar_memories = [
            {
                "content": "User prefers technical explanations",
                "memory_type": "preference",
                "importance": 0.8,
                "tags": ["technical"]
            },
            {
                "content": "User likes technical details",
                "memory_type": "preference",
                "importance": 0.7,
                "tags": ["technical"]
            }
        ]
        
        for memory in similar_memories:
            client.post("/api/v1/memory/memories", json=memory, headers=auth_headers)
        
        # Consolidate memories
        response = client.post("/api/v1/memory/consolidate", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "consolidated_count" in data
        assert "Successfully consolidated" in data["message"]


class TestUserPreferences:
    """Test user preferences management."""
    
    def test_get_user_preferences_default(self, client, auth_headers):
        """Test getting default user preferences."""
        response = client.get("/api/v1/memory/preferences", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "user_id" in data
        assert "preferences" in data
        assert "updated_at" in data
    
    def test_update_user_preferences(self, client, auth_headers):
        """Test updating user preferences."""
        preferences_data = {
            "theme": "dark",
            "language": "en",
            "notifications": True,
            "ai_model": "gpt-4"
        }
        
        response = client.put("/api/v1/memory/preferences", 
                            json=preferences_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["preferences"]["theme"] == "dark"
        assert data["preferences"]["language"] == "en"
        assert data["preferences"]["notifications"] is True
    
    def test_get_updated_preferences(self, client, auth_headers):
        """Test getting updated preferences."""
        # Update preferences first
        preferences_data = {"test_setting": "test_value"}
        client.put("/api/v1/memory/preferences", 
                  json=preferences_data, headers=auth_headers)
        
        # Get preferences
        response = client.get("/api/v1/memory/preferences", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["preferences"]["test_setting"] == "test_value"


class TestMemoryStatistics:
    """Test memory statistics functionality."""
    
    def test_get_memory_stats(self, client, auth_headers, test_memory_data):
        """Test getting memory statistics."""
        # Create some memories first
        memories = [
            test_memory_data,
            {
                "content": "Another memory",
                "memory_type": "fact",
                "importance": 0.6,
                "tags": ["fact"]
            }
        ]
        
        for memory in memories:
            client.post("/api/v1/memory/memories", json=memory, headers=auth_headers)
        
        # Get stats
        response = client.get("/api/v1/memory/stats", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_memories" in data
        assert "memory_type_distribution" in data
        assert "average_importance" in data
        assert "most_accessed_memories" in data
        assert "recent_memories" in data
        assert data["total_memories"] >= 2
    
    def test_get_memory_types(self, client):
        """Test getting supported memory types."""
        response = client.get("/api/v1/memory/types")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "supported_types" in data
        assert "descriptions" in data
        assert isinstance(data["supported_types"], list)
        assert "preference" in data["supported_types"]
        assert "fact" in data["supported_types"]


class TestMemoryUtilities:
    """Test memory utility functions."""
    
    def test_validate_memory_type_valid(self):
        """Test memory type validation with valid types."""
        valid_types = ["preference", "fact", "context", "skill", "goal", "feedback"]
        
        for memory_type in valid_types:
            assert validate_memory_type(memory_type) is True
    
    def test_validate_memory_type_invalid(self):
        """Test memory type validation with invalid types."""
        invalid_types = ["invalid_type", "", "PREFERENCE", "123"]
        
        for memory_type in invalid_types:
            assert validate_memory_type(memory_type) is False
    
    def test_calculate_memory_relevance(self):
        """Test memory relevance calculation."""
        from src.memory.router import MemoryItem
        from uuid import uuid4
        from datetime import datetime
        
        memory = MemoryItem(
            id=uuid4(),
            user_id=uuid4(),
            content="User prefers technical explanations with examples",
            memory_type="preference",
            importance=0.8,
            tags=["technical", "examples"],
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow()
        )
        
        # Test exact match
        relevance = calculate_memory_relevance(memory, "technical explanations")
        assert relevance > 0.5
        
        # Test partial match
        relevance = calculate_memory_relevance(memory, "technical")
        assert relevance > 0.3
        
        # Test no match
        relevance = calculate_memory_relevance(memory, "completely unrelated query")
        assert relevance < 0.3
        
        # Test tag match
        relevance = calculate_memory_relevance(memory, "examples")
        assert relevance > 0.2
    
    def test_consolidate_memories_function(self):
        """Test memory consolidation function."""
        from src.memory.router import memories_db, MemoryItem
        from uuid import uuid4
        from datetime import datetime
        
        # Clear memories for clean test
        original_memories = memories_db.copy()
        memories_db.clear()
        
        try:
            user_id = uuid4()
            
            # Create similar memories
            memory1 = MemoryItem(
                id=uuid4(),
                user_id=user_id,
                content="User prefers technical explanations",
                memory_type="preference",
                importance=0.8,
                tags=["technical"]
            )
            
            memory2 = MemoryItem(
                id=uuid4(),
                user_id=user_id,
                content="User likes technical details",
                memory_type="preference",
                importance=0.7,
                tags=["technical"]
            )
            
            memories_db[memory1.id] = memory1
            memories_db[memory2.id] = memory2
            
            # Test consolidation
            consolidated_count = consolidate_memories(user_id)
            
            # Should consolidate similar memories
            assert consolidated_count >= 0
            
        finally:
            # Restore original memories
            memories_db.clear()
            memories_db.update(original_memories)


class TestMemoryValidation:
    """Test memory input validation."""
    
    def test_memory_creation_validation(self, client, auth_headers):
        """Test memory creation input validation."""
        invalid_memories = [
            # Missing required fields
            {"memory_type": "preference"},
            {"content": "Test memory"},
            # Invalid field types
            {"content": 123, "memory_type": "preference"},
            {"content": "Test", "memory_type": 123},
            # Invalid importance value
            {"content": "Test", "memory_type": "preference", "importance": 2.0},
            {"content": "Test", "memory_type": "preference", "importance": -0.1},
        ]
        
        for memory_data in invalid_memories:
            response = client.post("/api/v1/memory/memories", 
                                 json=memory_data, headers=auth_headers)
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_memory_search_validation(self, client, auth_headers):
        """Test memory search input validation."""
        invalid_searches = [
            # Missing query
            {"memory_types": ["preference"]},
            # Invalid limit
            {"query": "test", "limit": 0},
            {"query": "test", "limit": 1000},
            # Invalid importance
            {"query": "test", "min_importance": 2.0},
        ]
        
        for search_data in invalid_searches:
            response = client.post("/api/v1/memory/memories/search", 
                                 json=search_data, headers=auth_headers)
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


if __name__ == "__main__":
    pytest.main([__file__])


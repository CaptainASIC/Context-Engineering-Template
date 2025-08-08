"""
Tests for knowledge graph endpoints and functionality.

This module tests entity management, relationship creation,
graph queries, and Neo4j integration patterns.
"""

import pytest
from fastapi import status
from uuid import uuid4

from src.knowledge.router import (
    validate_entity_type, 
    validate_relationship_type, 
    find_entity_neighbors
)


class TestKnowledgeGraphEndpoints:
    """Test knowledge graph management endpoints."""
    
    def test_create_entity_success(self, client, auth_headers, test_entity_data):
        """Test successful entity creation."""
        response = client.post("/api/v1/knowledge/entities", 
                             json=test_entity_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == test_entity_data["name"]
        assert data["type"] == test_entity_data["type"]
        assert data["description"] == test_entity_data["description"]
        assert data["properties"] == test_entity_data["properties"]
        assert "id" in data
        assert "created_at" in data
    
    def test_create_entity_invalid_type(self, client, auth_headers, test_entity_data):
        """Test entity creation with invalid type."""
        test_entity_data["type"] = "invalid_type"
        response = client.post("/api/v1/knowledge/entities", 
                             json=test_entity_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "Invalid entity type" in response.json()["message"]
    
    def test_create_entity_unauthorized(self, client, test_entity_data):
        """Test entity creation without authentication."""
        response = client.post("/api/v1/knowledge/entities", json=test_entity_data)
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_list_entities(self, client, auth_headers, test_entity_data):
        """Test listing entities."""
        # Create an entity first
        create_response = client.post("/api/v1/knowledge/entities", 
                                    json=test_entity_data, headers=auth_headers)
        assert create_response.status_code == status.HTTP_200_OK
        
        # List entities
        response = client.get("/api/v1/knowledge/entities", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert any(entity["name"] == test_entity_data["name"] for entity in data)
    
    def test_list_entities_with_filters(self, client, auth_headers, test_entity_data):
        """Test listing entities with filters."""
        # Create an entity first
        client.post("/api/v1/knowledge/entities", 
                   json=test_entity_data, headers=auth_headers)
        
        # Filter by type
        response = client.get(f"/api/v1/knowledge/entities?entity_type={test_entity_data['type']}", 
                            headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert all(entity["type"] == test_entity_data["type"] for entity in data)
        
        # Search by name
        response = client.get(f"/api/v1/knowledge/entities?search={test_entity_data['name']}", 
                            headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert any(test_entity_data["name"].lower() in entity["name"].lower() for entity in data)
    
    def test_get_entity_success(self, client, auth_headers, test_entity_data):
        """Test getting a specific entity."""
        # Create an entity first
        create_response = client.post("/api/v1/knowledge/entities", 
                                    json=test_entity_data, headers=auth_headers)
        entity_id = create_response.json()["id"]
        
        # Get the entity
        response = client.get(f"/api/v1/knowledge/entities/{entity_id}", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == entity_id
        assert data["name"] == test_entity_data["name"]
    
    def test_get_entity_not_found(self, client, auth_headers):
        """Test getting a nonexistent entity."""
        fake_id = str(uuid4())
        response = client.get(f"/api/v1/knowledge/entities/{fake_id}", headers=auth_headers)
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["message"]
    
    def test_update_entity_success(self, client, auth_headers, test_entity_data):
        """Test successful entity update."""
        # Create an entity first
        create_response = client.post("/api/v1/knowledge/entities", 
                                    json=test_entity_data, headers=auth_headers)
        entity_id = create_response.json()["id"]
        
        # Update the entity
        updated_data = test_entity_data.copy()
        updated_data["name"] = "Updated Entity Name"
        updated_data["properties"]["updated"] = True
        
        response = client.put(f"/api/v1/knowledge/entities/{entity_id}", 
                            json=updated_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == "Updated Entity Name"
        assert data["properties"]["updated"] is True
    
    def test_delete_entity_success(self, client, auth_headers, test_entity_data):
        """Test successful entity deletion."""
        # Create an entity first
        create_response = client.post("/api/v1/knowledge/entities", 
                                    json=test_entity_data, headers=auth_headers)
        entity_id = create_response.json()["id"]
        
        # Delete the entity
        response = client.delete(f"/api/v1/knowledge/entities/{entity_id}", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        assert "deleted successfully" in response.json()["message"]
        
        # Verify entity is deleted
        get_response = client.get(f"/api/v1/knowledge/entities/{entity_id}", headers=auth_headers)
        assert get_response.status_code == status.HTTP_404_NOT_FOUND


class TestKnowledgeGraphRelationships:
    """Test relationship management."""
    
    def test_create_relationship_success(self, client, auth_headers, test_entity_data):
        """Test successful relationship creation."""
        # Create two entities first
        entity1_response = client.post("/api/v1/knowledge/entities", 
                                     json=test_entity_data, headers=auth_headers)
        entity1_id = entity1_response.json()["id"]
        
        entity2_data = test_entity_data.copy()
        entity2_data["name"] = "Jane Doe"
        entity2_response = client.post("/api/v1/knowledge/entities", 
                                     json=entity2_data, headers=auth_headers)
        entity2_id = entity2_response.json()["id"]
        
        # Create relationship
        relationship_data = {
            "source_id": entity1_id,
            "target_id": entity2_id,
            "type": "related_to",
            "properties": {"context": "colleagues"},
            "strength": 0.8
        }
        
        response = client.post("/api/v1/knowledge/relationships", 
                             json=relationship_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["source_id"] == entity1_id
        assert data["target_id"] == entity2_id
        assert data["type"] == "related_to"
        assert data["strength"] == 0.8
        assert "id" in data
    
    def test_create_relationship_invalid_entities(self, client, auth_headers):
        """Test relationship creation with invalid entities."""
        fake_id1 = str(uuid4())
        fake_id2 = str(uuid4())
        
        relationship_data = {
            "source_id": fake_id1,
            "target_id": fake_id2,
            "type": "related_to"
        }
        
        response = client.post("/api/v1/knowledge/relationships", 
                             json=relationship_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_create_relationship_invalid_type(self, client, auth_headers, test_entity_data):
        """Test relationship creation with invalid type."""
        # Create two entities first
        entity1_response = client.post("/api/v1/knowledge/entities", 
                                     json=test_entity_data, headers=auth_headers)
        entity1_id = entity1_response.json()["id"]
        
        entity2_data = test_entity_data.copy()
        entity2_data["name"] = "Jane Doe"
        entity2_response = client.post("/api/v1/knowledge/entities", 
                                     json=entity2_data, headers=auth_headers)
        entity2_id = entity2_response.json()["id"]
        
        # Create relationship with invalid type
        relationship_data = {
            "source_id": entity1_id,
            "target_id": entity2_id,
            "type": "invalid_relationship_type"
        }
        
        response = client.post("/api/v1/knowledge/relationships", 
                             json=relationship_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "Invalid relationship type" in response.json()["message"]
    
    def test_create_self_relationship(self, client, auth_headers, test_entity_data):
        """Test creating relationship to self (should fail)."""
        # Create an entity first
        entity_response = client.post("/api/v1/knowledge/entities", 
                                    json=test_entity_data, headers=auth_headers)
        entity_id = entity_response.json()["id"]
        
        # Try to create self-relationship
        relationship_data = {
            "source_id": entity_id,
            "target_id": entity_id,
            "type": "related_to"
        }
        
        response = client.post("/api/v1/knowledge/relationships", 
                             json=relationship_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Cannot create relationship to self" in response.json()["message"]
    
    def test_list_relationships(self, client, auth_headers, test_entity_data):
        """Test listing relationships."""
        # Create entities and relationship first
        entity1_response = client.post("/api/v1/knowledge/entities", 
                                     json=test_entity_data, headers=auth_headers)
        entity1_id = entity1_response.json()["id"]
        
        entity2_data = test_entity_data.copy()
        entity2_data["name"] = "Jane Doe"
        entity2_response = client.post("/api/v1/knowledge/entities", 
                                     json=entity2_data, headers=auth_headers)
        entity2_id = entity2_response.json()["id"]
        
        relationship_data = {
            "source_id": entity1_id,
            "target_id": entity2_id,
            "type": "related_to"
        }
        client.post("/api/v1/knowledge/relationships", 
                   json=relationship_data, headers=auth_headers)
        
        # List relationships
        response = client.get("/api/v1/knowledge/relationships", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
    
    def test_list_relationships_with_filters(self, client, auth_headers, test_entity_data):
        """Test listing relationships with filters."""
        # Create entities and relationship first
        entity1_response = client.post("/api/v1/knowledge/entities", 
                                     json=test_entity_data, headers=auth_headers)
        entity1_id = entity1_response.json()["id"]
        
        entity2_data = test_entity_data.copy()
        entity2_data["name"] = "Jane Doe"
        entity2_response = client.post("/api/v1/knowledge/entities", 
                                     json=entity2_data, headers=auth_headers)
        entity2_id = entity2_response.json()["id"]
        
        relationship_data = {
            "source_id": entity1_id,
            "target_id": entity2_id,
            "type": "related_to"
        }
        client.post("/api/v1/knowledge/relationships", 
                   json=relationship_data, headers=auth_headers)
        
        # Filter by entity
        response = client.get(f"/api/v1/knowledge/relationships?entity_id={entity1_id}", 
                            headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert all(rel["source_id"] == entity1_id or rel["target_id"] == entity1_id 
                  for rel in data)
        
        # Filter by type
        response = client.get("/api/v1/knowledge/relationships?rel_type=related_to", 
                            headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert all(rel["type"] == "related_to" for rel in data)


class TestKnowledgeGraphQueries:
    """Test graph query functionality."""
    
    def test_get_entity_neighbors(self, client, auth_headers, test_entity_data):
        """Test getting entity neighbors."""
        # Create entities and relationships
        entity1_response = client.post("/api/v1/knowledge/entities", 
                                     json=test_entity_data, headers=auth_headers)
        entity1_id = entity1_response.json()["id"]
        
        entity2_data = test_entity_data.copy()
        entity2_data["name"] = "Jane Doe"
        entity2_response = client.post("/api/v1/knowledge/entities", 
                                     json=entity2_data, headers=auth_headers)
        entity2_id = entity2_response.json()["id"]
        
        # Create relationship
        relationship_data = {
            "source_id": entity1_id,
            "target_id": entity2_id,
            "type": "related_to"
        }
        client.post("/api/v1/knowledge/relationships", 
                   json=relationship_data, headers=auth_headers)
        
        # Get neighbors
        response = client.get(f"/api/v1/knowledge/entities/{entity1_id}/neighbors", 
                            headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        # Should find entity2 as neighbor
        neighbor_ids = [entity["id"] for entity in data]
        assert entity2_id in neighbor_ids
    
    def test_graph_query_neighbors(self, client, auth_headers, test_entity_data):
        """Test graph query for neighbors."""
        # Create an entity first
        entity_response = client.post("/api/v1/knowledge/entities", 
                                    json=test_entity_data, headers=auth_headers)
        entity_id = entity_response.json()["id"]
        
        # Query neighbors
        query_data = {
            "query_type": "neighbors",
            "parameters": {
                "entity_id": entity_id,
                "max_depth": 2
            },
            "limit": 10
        }
        
        response = client.post("/api/v1/knowledge/query", 
                             json=query_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "entities" in data
        assert "metadata" in data
        assert data["metadata"]["query_type"] == "neighbors"
    
    def test_graph_query_invalid_type(self, client, auth_headers):
        """Test graph query with invalid type."""
        query_data = {
            "query_type": "invalid_query_type",
            "parameters": {},
            "limit": 10
        }
        
        response = client.post("/api/v1/knowledge/query", 
                             json=query_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "Unsupported query type" in response.json()["message"]
    
    def test_get_graph_schema(self, client, auth_headers):
        """Test getting graph schema information."""
        response = client.get("/api/v1/knowledge/schema", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "supported_entity_types" in data
        assert "supported_relationship_types" in data
        assert "statistics" in data
        assert isinstance(data["supported_entity_types"], list)
        assert isinstance(data["supported_relationship_types"], list)


class TestKnowledgeGraphUtilities:
    """Test knowledge graph utility functions."""
    
    def test_validate_entity_type_valid(self):
        """Test entity type validation with valid types."""
        valid_types = ["person", "policy", "use_case", "concept", "document", "organization"]
        
        for entity_type in valid_types:
            assert validate_entity_type(entity_type) is True
    
    def test_validate_entity_type_invalid(self):
        """Test entity type validation with invalid types."""
        invalid_types = ["invalid_type", "", "PERSON", "123"]
        
        for entity_type in invalid_types:
            assert validate_entity_type(entity_type) is False
    
    def test_validate_relationship_type_valid(self):
        """Test relationship type validation with valid types."""
        valid_types = ["related_to", "implements", "uses", "created_by", "part_of"]
        
        for rel_type in valid_types:
            assert validate_relationship_type(rel_type) is True
    
    def test_validate_relationship_type_invalid(self):
        """Test relationship type validation with invalid types."""
        invalid_types = ["invalid_relationship", "", "RELATED_TO", "123"]
        
        for rel_type in invalid_types:
            assert validate_relationship_type(rel_type) is False
    
    def test_find_entity_neighbors_empty(self):
        """Test finding neighbors with no relationships."""
        from src.knowledge.router import relationships_db
        
        # Clear relationships for clean test
        original_relationships = relationships_db.copy()
        relationships_db.clear()
        
        try:
            entity_id = uuid4()
            neighbors = find_entity_neighbors(entity_id, max_depth=2)
            assert neighbors == []
        finally:
            # Restore original relationships
            relationships_db.update(original_relationships)


class TestKnowledgeGraphValidation:
    """Test knowledge graph input validation."""
    
    def test_entity_creation_validation(self, client, auth_headers):
        """Test entity creation input validation."""
        invalid_entities = [
            # Missing required fields
            {"name": "Test"},
            {"type": "person"},
            # Invalid field types
            {"type": "person", "name": 123, "description": "Test"},
            {"type": "person", "name": "Test", "properties": "not_a_dict"},
        ]
        
        for entity_data in invalid_entities:
            response = client.post("/api/v1/knowledge/entities", 
                                 json=entity_data, headers=auth_headers)
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_relationship_creation_validation(self, client, auth_headers):
        """Test relationship creation input validation."""
        invalid_relationships = [
            # Missing required fields
            {"source_id": str(uuid4())},
            {"target_id": str(uuid4())},
            # Invalid field types
            {"source_id": "not_uuid", "target_id": str(uuid4()), "type": "related_to"},
            {"source_id": str(uuid4()), "target_id": str(uuid4()), "type": 123},
            # Invalid strength value
            {"source_id": str(uuid4()), "target_id": str(uuid4()), 
             "type": "related_to", "strength": 2.0},
        ]
        
        for rel_data in invalid_relationships:
            response = client.post("/api/v1/knowledge/relationships", 
                                 json=rel_data, headers=auth_headers)
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


if __name__ == "__main__":
    pytest.main([__file__])


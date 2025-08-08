"""
Tests for Neo4j Service.

This module tests the Neo4j database service including
connection management, CRUD operations, and graph queries.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from src.services.neo4j_service import Neo4jService
from src.models.graph_models import PersonNode, PolicyNode, WorksForRelationship
from config.settings import KnowledgeGraphSettings, Neo4jConfig


@pytest.fixture
def mock_settings():
    """Mock knowledge graph settings for testing."""
    settings = KnowledgeGraphSettings()
    settings.neo4j = Neo4jConfig(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="test-password",
        database="test"
    )
    return settings


@pytest.fixture
def sample_person_data():
    """Sample person data for testing."""
    return {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "title": "Software Engineer",
        "department": "Engineering",
        "skills": ["Python", "Neo4j", "GraphQL"]
    }


@pytest.fixture
def sample_policy_data():
    """Sample policy data for testing."""
    return {
        "name": "Data Privacy Policy",
        "description": "Policy for handling personal data",
        "version": "2.0",
        "status": "active",
        "category": "privacy"
    }


class TestNeo4jService:
    """Test Neo4j service functionality."""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, mock_settings):
        """Test Neo4j service initialization."""
        service = Neo4jService(mock_settings)
        
        assert service.settings == mock_settings
        assert service.driver is None
    
    @pytest.mark.asyncio
    async def test_connect_success(self, mock_settings):
        """Test successful connection to Neo4j."""
        with patch('src.services.neo4j_service.AsyncGraphDatabase') as mock_db:
            mock_driver = AsyncMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_db.driver.return_value = mock_driver
            
            service = Neo4jService(mock_settings)
            result = await service.connect()
            
            assert result is True
            assert service.driver == mock_driver
            mock_driver.verify_connectivity.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, mock_settings):
        """Test failed connection to Neo4j."""
        with patch('src.services.neo4j_service.AsyncGraphDatabase') as mock_db:
            mock_db.driver.side_effect = Exception("Connection failed")
            
            service = Neo4jService(mock_settings)
            result = await service.connect()
            
            assert result is False
            assert service.driver is None
    
    @pytest.mark.asyncio
    async def test_disconnect(self, mock_settings):
        """Test disconnection from Neo4j."""
        with patch('src.services.neo4j_service.AsyncGraphDatabase') as mock_db:
            mock_driver = AsyncMock()
            mock_db.driver.return_value = mock_driver
            
            service = Neo4jService(mock_settings)
            await service.connect()
            await service.disconnect()
            
            mock_driver.close.assert_called_once()
            assert service.driver is None
    
    @pytest.mark.asyncio
    async def test_execute_query(self, mock_settings):
        """Test query execution."""
        with patch('src.services.neo4j_service.AsyncGraphDatabase') as mock_db:
            # Mock session and result
            mock_result = AsyncMock()
            mock_result.data.return_value = [{"test": "value"}]
            
            mock_session = AsyncMock()
            mock_session.run.return_value = mock_result
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = None
            
            mock_driver = AsyncMock()
            mock_driver.session.return_value = mock_session
            mock_driver.verify_connectivity = AsyncMock()
            mock_db.driver.return_value = mock_driver
            
            service = Neo4jService(mock_settings)
            await service.connect()
            
            result = await service.execute_query("RETURN 1 as test", {"param": "value"})
            
            assert result == [{"test": "value"}]
            mock_session.run.assert_called_once_with("RETURN 1 as test", {"param": "value"})
    
    @pytest.mark.asyncio
    async def test_create_node(self, mock_settings, sample_person_data):
        """Test node creation."""
        with patch('src.services.neo4j_service.AsyncGraphDatabase') as mock_db:
            # Mock the query execution
            mock_result = [{"n": sample_person_data, "node_id": 123}]
            
            service = Neo4jService(mock_settings)
            service.execute_query = AsyncMock(return_value=mock_result)
            
            result = await service.create_node("Person", sample_person_data)
            
            assert result["id"] == 123
            assert result["label"] == "Person"
            assert result["properties"]["name"] == sample_person_data["name"]
            
            # Verify query was called
            service.execute_query.assert_called_once()
            call_args = service.execute_query.call_args
            assert "CREATE" in call_args[0][0] or "MERGE" in call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_create_node_with_merge(self, mock_settings, sample_person_data):
        """Test node creation with merge option."""
        with patch('src.services.neo4j_service.AsyncGraphDatabase'):
            mock_result = [{"n": sample_person_data, "node_id": 123}]
            
            service = Neo4jService(mock_settings)
            service.execute_query = AsyncMock(return_value=mock_result)
            
            result = await service.create_node("Person", sample_person_data, merge=True)
            
            assert result["id"] == 123
            
            # Verify MERGE was used
            call_args = service.execute_query.call_args
            assert "MERGE" in call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_create_relationship(self, mock_settings):
        """Test relationship creation."""
        with patch('src.services.neo4j_service.AsyncGraphDatabase'):
            mock_result = [{"r": {"role": "Engineer"}, "rel_id": 456}]
            
            service = Neo4jService(mock_settings)
            service.execute_query = AsyncMock(return_value=mock_result)
            
            result = await service.create_relationship(
                from_node_id=123,
                to_node_id=789,
                relationship_type="WORKS_FOR",
                properties={"role": "Engineer"}
            )
            
            assert result["id"] == 456
            assert result["type"] == "WORKS_FOR"
            assert result["from_node_id"] == 123
            assert result["to_node_id"] == 789
            assert result["properties"]["role"] == "Engineer"
    
    @pytest.mark.asyncio
    async def test_find_nodes(self, mock_settings):
        """Test finding nodes."""
        with patch('src.services.neo4j_service.AsyncGraphDatabase'):
            mock_result = [
                {"n": {"name": "John"}, "node_id": 123, "labels": ["Person"]},
                {"n": {"name": "Jane"}, "node_id": 124, "labels": ["Person"]}
            ]
            
            service = Neo4jService(mock_settings)
            service.execute_query = AsyncMock(return_value=mock_result)
            
            result = await service.find_nodes(
                label="Person",
                properties={"department": "Engineering"},
                limit=10
            )
            
            assert len(result) == 2
            assert result[0]["id"] == 123
            assert result[0]["labels"] == ["Person"]
            assert result[1]["properties"]["name"] == "Jane"
    
    @pytest.mark.asyncio
    async def test_find_relationships(self, mock_settings):
        """Test finding relationships."""
        with patch('src.services.neo4j_service.AsyncGraphDatabase'):
            mock_result = [
                {
                    "r": {"role": "Manager"},
                    "rel_id": 456,
                    "rel_type": "MANAGES",
                    "from_id": 123,
                    "to_id": 124
                }
            ]
            
            service = Neo4jService(mock_settings)
            service.execute_query = AsyncMock(return_value=mock_result)
            
            result = await service.find_relationships(
                from_node_id=123,
                relationship_type="MANAGES"
            )
            
            assert len(result) == 1
            assert result[0]["id"] == 456
            assert result[0]["type"] == "MANAGES"
            assert result[0]["properties"]["role"] == "Manager"
    
    @pytest.mark.asyncio
    async def test_update_node(self, mock_settings):
        """Test node update."""
        with patch('src.services.neo4j_service.AsyncGraphDatabase'):
            updated_data = {"name": "John Updated", "title": "Senior Engineer"}
            mock_result = [{"n": updated_data, "node_id": 123, "labels": ["Person"]}]
            
            service = Neo4jService(mock_settings)
            service.execute_query = AsyncMock(return_value=mock_result)
            
            result = await service.update_node(123, {"title": "Senior Engineer"})
            
            assert result["id"] == 123
            assert result["properties"]["title"] == "Senior Engineer"
    
    @pytest.mark.asyncio
    async def test_delete_node(self, mock_settings):
        """Test node deletion."""
        with patch('src.services.neo4j_service.AsyncGraphDatabase'):
            service = Neo4jService(mock_settings)
            service.execute_write_query = AsyncMock(return_value={"nodes_deleted": 1})
            
            result = await service.delete_node(123)
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_delete_node_with_force(self, mock_settings):
        """Test forced node deletion."""
        with patch('src.services.neo4j_service.AsyncGraphDatabase'):
            service = Neo4jService(mock_settings)
            service.execute_write_query = AsyncMock(return_value={"nodes_deleted": 1})
            
            result = await service.delete_node(123, force=True)
            
            assert result is True
            
            # Verify DETACH DELETE was used
            call_args = service.execute_write_query.call_args
            assert "DETACH DELETE" in call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_find_shortest_path(self, mock_settings):
        """Test shortest path finding."""
        with patch('src.services.neo4j_service.AsyncGraphDatabase'):
            mock_result = [{
                "path_length": 2,
                "nodes": [
                    {"id": 123, "labels": ["Person"], "properties": {"name": "John"}},
                    {"id": 456, "labels": ["Organization"], "properties": {"name": "Company"}},
                    {"id": 789, "labels": ["Person"], "properties": {"name": "Jane"}}
                ],
                "relationships": [
                    {"id": 1, "type": "WORKS_FOR", "properties": {}},
                    {"id": 2, "type": "MANAGES", "properties": {}}
                ]
            }]
            
            service = Neo4jService(mock_settings)
            service.execute_query = AsyncMock(return_value=mock_result)
            
            result = await service.find_shortest_path(123, 789, max_length=5)
            
            assert result is not None
            assert result["length"] == 2
            assert len(result["nodes"]) == 3
            assert len(result["relationships"]) == 2
    
    @pytest.mark.asyncio
    async def test_get_node_neighbors(self, mock_settings):
        """Test getting node neighbors."""
        with patch('src.services.neo4j_service.AsyncGraphDatabase'):
            mock_result = [
                {
                    "neighbor": {"name": "Company A"},
                    "neighbor_id": 456,
                    "neighbor_labels": ["Organization"],
                    "r": {"role": "Engineer"},
                    "rel_id": 789,
                    "rel_type": "WORKS_FOR"
                }
            ]
            
            service = Neo4jService(mock_settings)
            service.execute_query = AsyncMock(return_value=mock_result)
            
            result = await service.get_node_neighbors(123, direction="out")
            
            assert len(result) == 1
            assert result[0]["node"]["id"] == 456
            assert result[0]["relationship"]["type"] == "WORKS_FOR"
    
    @pytest.mark.asyncio
    async def test_get_database_stats(self, mock_settings):
        """Test getting database statistics."""
        with patch('src.services.neo4j_service.AsyncGraphDatabase'):
            service = Neo4jService(mock_settings)
            
            # Mock different query results
            def mock_execute_query(query, params=None):
                if "count(n)" in query:
                    return [{"count": 100}]
                elif "count(r)" in query:
                    return [{"count": 50}]
                elif "db.labels()" in query:
                    return [{"labels": ["Person", "Organization"]}]
                elif "db.relationshipTypes()" in query:
                    return [{"types": ["WORKS_FOR", "MANAGES"]}]
                elif "db.propertyKeys()" in query:
                    return [{"keys": ["name", "email", "title"]}]
                elif "Person" in query and "count(n)" in query:
                    return [{"count": 60}]
                elif "Organization" in query and "count(n)" in query:
                    return [{"count": 40}]
                else:
                    return []
            
            service.execute_query = AsyncMock(side_effect=mock_execute_query)
            
            result = await service.get_database_stats()
            
            assert result["node_count"] == 100
            assert result["relationship_count"] == 50
            assert "Person" in result["node_labels"]
            assert "WORKS_FOR" in result["relationship_types"]
            assert result["node_label_counts"]["Person"] == 60
    
    @pytest.mark.asyncio
    async def test_create_indexes(self, mock_settings):
        """Test index creation."""
        with patch('src.services.neo4j_service.AsyncGraphDatabase'):
            service = Neo4jService(mock_settings)
            service.execute_write_query = AsyncMock(return_value={})
            
            result = await service.create_indexes()
            
            assert "created" in result
            assert "failed" in result
            assert len(result["created"]) > 0  # Should create some indexes
    
    @pytest.mark.asyncio
    async def test_create_constraints(self, mock_settings):
        """Test constraint creation."""
        with patch('src.services.neo4j_service.AsyncGraphDatabase'):
            service = Neo4jService(mock_settings)
            service.execute_write_query = AsyncMock(return_value={})
            
            result = await service.create_constraints()
            
            assert "created" in result
            assert "failed" in result
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_settings):
        """Test health check when system is healthy."""
        with patch('src.services.neo4j_service.AsyncGraphDatabase'):
            service = Neo4jService(mock_settings)
            service.execute_query = AsyncMock(return_value=[{"test": 1}])
            service.get_database_stats = AsyncMock(return_value={
                "node_count": 100,
                "relationship_count": 50
            })
            
            result = await service.health_check()
            
            assert result["status"] == "healthy"
            assert result["details"]["connectivity"] == "ok"
            assert result["details"]["stats"]["nodes"] == 100
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, mock_settings):
        """Test health check when system is unhealthy."""
        with patch('src.services.neo4j_service.AsyncGraphDatabase'):
            service = Neo4jService(mock_settings)
            service.execute_query = AsyncMock(side_effect=Exception("Connection failed"))
            
            result = await service.health_check()
            
            assert result["status"] == "unhealthy"
            assert "error" in result["details"]


class TestNodeValidation:
    """Test node property validation."""
    
    @pytest.mark.asyncio
    async def test_valid_node_creation(self, mock_settings, sample_person_data):
        """Test creation of valid node."""
        with patch('src.services.neo4j_service.AsyncGraphDatabase'):
            service = Neo4jService(mock_settings)
            service.execute_query = AsyncMock(return_value=[{"n": sample_person_data, "node_id": 123}])
            
            result = await service.create_node("Person", sample_person_data)
            
            assert result["id"] == 123
    
    @pytest.mark.asyncio
    async def test_invalid_node_creation(self, mock_settings):
        """Test creation of invalid node."""
        with patch('src.services.neo4j_service.AsyncGraphDatabase'):
            service = Neo4jService(mock_settings)
            
            # Missing required 'name' field
            invalid_data = {"email": "test@example.com"}
            
            with pytest.raises(ValueError) as exc_info:
                await service.create_node("Person", invalid_data)
            
            assert "validation errors" in str(exc_info.value)


class TestRelationshipValidation:
    """Test relationship property validation."""
    
    @pytest.mark.asyncio
    async def test_valid_relationship_creation(self, mock_settings):
        """Test creation of valid relationship."""
        with patch('src.services.neo4j_service.AsyncGraphDatabase'):
            service = Neo4jService(mock_settings)
            service.execute_query = AsyncMock(return_value=[{"r": {"role": "Engineer"}, "rel_id": 456}])
            
            result = await service.create_relationship(
                from_node_id=123,
                to_node_id=789,
                relationship_type="WORKS_FOR",
                properties={"role": "Engineer"}
            )
            
            assert result["id"] == 456
    
    @pytest.mark.asyncio
    async def test_invalid_relationship_creation(self, mock_settings):
        """Test creation of invalid relationship."""
        with patch('src.services.neo4j_service.AsyncGraphDatabase'):
            service = Neo4jService(mock_settings)
            
            # Invalid property type
            invalid_props = {"strength": "very strong"}  # Should be float
            
            with pytest.raises(ValueError) as exc_info:
                await service.create_relationship(
                    from_node_id=123,
                    to_node_id=789,
                    relationship_type="RELATES_TO",
                    properties=invalid_props
                )
            
            assert "validation errors" in str(exc_info.value)


class TestErrorHandling:
    """Test error handling in Neo4j service."""
    
    @pytest.mark.asyncio
    async def test_query_execution_error(self, mock_settings):
        """Test handling of query execution errors."""
        with patch('src.services.neo4j_service.AsyncGraphDatabase') as mock_db:
            mock_session = AsyncMock()
            mock_session.run.side_effect = Exception("Query failed")
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = None
            
            mock_driver = AsyncMock()
            mock_driver.session.return_value = mock_session
            mock_driver.verify_connectivity = AsyncMock()
            mock_db.driver.return_value = mock_driver
            
            service = Neo4jService(mock_settings)
            await service.connect()
            
            with pytest.raises(Exception) as exc_info:
                await service.execute_query("INVALID QUERY")
            
            assert "Query failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_node_creation_error(self, mock_settings):
        """Test handling of node creation errors."""
        with patch('src.services.neo4j_service.AsyncGraphDatabase'):
            service = Neo4jService(mock_settings)
            service.execute_query = AsyncMock(side_effect=Exception("Creation failed"))
            
            with pytest.raises(Exception) as exc_info:
                await service.create_node("Person", {"name": "Test"})
            
            assert "Creation failed" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])


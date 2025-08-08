"""
Pytest configuration and fixtures for FastAPI backend tests.

This module provides shared fixtures and configuration for testing
the Context Engineering Platform API.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from src.main import create_application
from src.common.config import get_settings


@pytest.fixture
def test_settings():
    """Test settings fixture."""
    with patch('src.common.config.get_settings') as mock_settings:
        settings = get_settings()
        settings.environment = "testing"
        settings.debug = True
        settings.secret_key = "test-secret-key"
        mock_settings.return_value = settings
        yield settings


@pytest.fixture
def app(test_settings):
    """FastAPI application fixture."""
    return create_application()


@pytest.fixture
def client(app):
    """Test client fixture."""
    return TestClient(app)


@pytest.fixture
def auth_headers(client):
    """Authenticated user headers fixture."""
    # Login with test user
    login_data = {
        "email": "admin@example.com",
        "password": "secret"
    }
    response = client.post("/api/v1/auth/login", json=login_data)
    assert response.status_code == 200
    
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def test_user():
    """Test user data fixture."""
    return {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "email": "admin@example.com",
        "full_name": "System Administrator",
        "is_active": True
    }


@pytest.fixture
def test_agent_config():
    """Test agent configuration fixture."""
    return {
        "name": "Test Agent",
        "description": "A test AI agent",
        "model_provider": "openai",
        "model_name": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000,
        "system_prompt": "You are a helpful assistant.",
        "tools_enabled": [],
        "rag_enabled": True,
        "memory_enabled": True
    }


@pytest.fixture
def test_entity_data():
    """Test entity data fixture."""
    return {
        "type": "person",
        "name": "John Doe",
        "description": "A test person entity",
        "properties": {"role": "developer", "team": "engineering"}
    }


@pytest.fixture
def test_memory_data():
    """Test memory data fixture."""
    return {
        "content": "User prefers technical explanations",
        "memory_type": "preference",
        "importance": 0.8,
        "tags": ["preference", "communication"],
        "metadata": {"source": "conversation"}
    }


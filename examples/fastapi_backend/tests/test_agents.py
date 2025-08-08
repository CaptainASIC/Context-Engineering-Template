"""
Tests for AI agents endpoints and functionality.

This module tests agent creation, management, chat functionality,
and multi-LLM integration patterns.
"""

import pytest
from fastapi import status
from unittest.mock import patch, AsyncMock
from uuid import uuid4

from src.agents.router import validate_model_provider, simulate_llm_response


class TestAgentEndpoints:
    """Test agent management endpoints."""
    
    def test_create_agent_success(self, client, auth_headers, test_agent_config):
        """Test successful agent creation."""
        response = client.post("/api/v1/agents/", json=test_agent_config, headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["config"]["name"] == test_agent_config["name"]
        assert data["config"]["model_provider"] == test_agent_config["model_provider"]
        assert data["is_active"] is True
        assert "id" in data
        assert "created_at" in data
    
    def test_create_agent_invalid_provider(self, client, auth_headers, test_agent_config):
        """Test agent creation with invalid model provider."""
        test_agent_config["model_provider"] = "invalid_provider"
        response = client.post("/api/v1/agents/", json=test_agent_config, headers=auth_headers)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "Invalid model provider" in response.json()["message"]
    
    def test_create_agent_invalid_model(self, client, auth_headers, test_agent_config):
        """Test agent creation with invalid model name."""
        test_agent_config["model_name"] = "invalid_model"
        response = client.post("/api/v1/agents/", json=test_agent_config, headers=auth_headers)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "Invalid model provider" in response.json()["message"]
    
    def test_create_agent_unauthorized(self, client, test_agent_config):
        """Test agent creation without authentication."""
        response = client.post("/api/v1/agents/", json=test_agent_config)
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_list_agents(self, client, auth_headers, test_agent_config):
        """Test listing user agents."""
        # Create an agent first
        create_response = client.post("/api/v1/agents/", json=test_agent_config, headers=auth_headers)
        assert create_response.status_code == status.HTTP_200_OK
        
        # List agents
        response = client.get("/api/v1/agents/", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["config"]["name"] == test_agent_config["name"]
    
    def test_get_agent_success(self, client, auth_headers, test_agent_config):
        """Test getting a specific agent."""
        # Create an agent first
        create_response = client.post("/api/v1/agents/", json=test_agent_config, headers=auth_headers)
        agent_id = create_response.json()["id"]
        
        # Get the agent
        response = client.get(f"/api/v1/agents/{agent_id}", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == agent_id
        assert data["config"]["name"] == test_agent_config["name"]
    
    def test_get_agent_not_found(self, client, auth_headers):
        """Test getting a nonexistent agent."""
        fake_id = str(uuid4())
        response = client.get(f"/api/v1/agents/{fake_id}", headers=auth_headers)
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["message"]
    
    def test_update_agent_success(self, client, auth_headers, test_agent_config):
        """Test successful agent update."""
        # Create an agent first
        create_response = client.post("/api/v1/agents/", json=test_agent_config, headers=auth_headers)
        agent_id = create_response.json()["id"]
        
        # Update the agent
        updated_config = test_agent_config.copy()
        updated_config["name"] = "Updated Agent Name"
        updated_config["temperature"] = 0.9
        
        response = client.put(f"/api/v1/agents/{agent_id}", json=updated_config, headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["config"]["name"] == "Updated Agent Name"
        assert data["config"]["temperature"] == 0.9
    
    def test_delete_agent_success(self, client, auth_headers, test_agent_config):
        """Test successful agent deletion."""
        # Create an agent first
        create_response = client.post("/api/v1/agents/", json=test_agent_config, headers=auth_headers)
        agent_id = create_response.json()["id"]
        
        # Delete the agent
        response = client.delete(f"/api/v1/agents/{agent_id}", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        assert "deleted successfully" in response.json()["message"]
        
        # Verify agent is deleted
        get_response = client.get(f"/api/v1/agents/{agent_id}", headers=auth_headers)
        assert get_response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_get_supported_providers(self, client):
        """Test getting supported model providers."""
        response = client.get("/api/v1/agents/providers/supported")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "openai" in data
        assert "anthropic" in data
        assert "google" in data
        assert "ollama" in data
        assert isinstance(data["openai"], list)


class TestAgentChat:
    """Test agent chat functionality."""
    
    @pytest.mark.asyncio
    async def test_chat_with_agent_success(self, client, auth_headers, test_agent_config):
        """Test successful chat with agent."""
        # Create an agent first
        create_response = client.post("/api/v1/agents/", json=test_agent_config, headers=auth_headers)
        agent_id = create_response.json()["id"]
        
        # Chat with the agent
        chat_data = {
            "message": "Hello, how can you help me?",
            "context": {"session_id": "test_session"},
            "stream": False
        }
        
        response = client.post(f"/api/v1/agents/{agent_id}/chat", json=chat_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert data["agent_id"] == agent_id
        assert "conversation_id" in data
        assert "metadata" in data
        assert isinstance(data["sources"], list)
    
    def test_chat_with_nonexistent_agent(self, client, auth_headers):
        """Test chat with nonexistent agent."""
        fake_id = str(uuid4())
        chat_data = {"message": "Hello"}
        
        response = client.post(f"/api/v1/agents/{fake_id}/chat", json=chat_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_chat_unauthorized(self, client, test_agent_config):
        """Test chat without authentication."""
        # Create agent with auth first
        fake_id = str(uuid4())
        chat_data = {"message": "Hello"}
        
        response = client.post(f"/api/v1/agents/{fake_id}/chat", json=chat_data)
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_get_agent_conversations(self, client, auth_headers, test_agent_config):
        """Test getting agent conversations."""
        # Create an agent first
        create_response = client.post("/api/v1/agents/", json=test_agent_config, headers=auth_headers)
        agent_id = create_response.json()["id"]
        
        # Have a conversation
        chat_data = {"message": "Test message"}
        client.post(f"/api/v1/agents/{agent_id}/chat", json=chat_data, headers=auth_headers)
        
        # Get conversations
        response = client.get(f"/api/v1/agents/{agent_id}/conversations", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["agent_id"] == agent_id
    
    def test_get_agent_stats(self, client, auth_headers, test_agent_config):
        """Test getting agent statistics."""
        # Create an agent first
        create_response = client.post("/api/v1/agents/", json=test_agent_config, headers=auth_headers)
        agent_id = create_response.json()["id"]
        
        # Get stats
        response = client.get(f"/api/v1/agents/{agent_id}/stats", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_conversations" in data
        assert "total_messages" in data
        assert "average_response_time" in data
        assert "success_rate" in data


class TestAgentUtilities:
    """Test agent utility functions."""
    
    def test_validate_model_provider_valid(self):
        """Test model provider validation with valid combinations."""
        assert validate_model_provider("openai", "gpt-4") is True
        assert validate_model_provider("anthropic", "claude-3-opus") is True
        assert validate_model_provider("google", "gemini-pro") is True
        assert validate_model_provider("ollama", "llama2") is True
    
    def test_validate_model_provider_invalid(self):
        """Test model provider validation with invalid combinations."""
        assert validate_model_provider("invalid_provider", "gpt-4") is False
        assert validate_model_provider("openai", "invalid_model") is False
        assert validate_model_provider("", "") is False
    
    @pytest.mark.asyncio
    async def test_simulate_llm_response(self, test_agent_config):
        """Test LLM response simulation."""
        from src.agents.router import AgentConfig
        
        config = AgentConfig(**test_agent_config)
        message = "Hello, how are you?"
        
        response = await simulate_llm_response(message, config)
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert message in response
        assert config.model_provider in response
        assert config.model_name in response
    
    @pytest.mark.asyncio
    async def test_simulate_llm_response_different_providers(self, test_agent_config):
        """Test LLM response simulation with different providers."""
        from src.agents.router import AgentConfig
        
        providers = ["openai", "anthropic", "google", "ollama"]
        message = "Test message"
        
        for provider in providers:
            config_data = test_agent_config.copy()
            config_data["model_provider"] = provider
            config_data["model_name"] = {
                "openai": "gpt-4",
                "anthropic": "claude-3-opus",
                "google": "gemini-pro",
                "ollama": "llama2"
            }[provider]
            
            config = AgentConfig(**config_data)
            response = await simulate_llm_response(message, config)
            
            assert provider in response
            assert config.model_name in response


class TestAgentValidation:
    """Test agent input validation."""
    
    def test_agent_config_validation(self, client, auth_headers):
        """Test agent configuration validation."""
        invalid_configs = [
            # Missing required fields
            {"name": "Test"},
            # Invalid temperature
            {"name": "Test", "description": "Test", "model_provider": "openai", 
             "model_name": "gpt-4", "temperature": 3.0, "system_prompt": "Test"},
            # Invalid max_tokens
            {"name": "Test", "description": "Test", "model_provider": "openai", 
             "model_name": "gpt-4", "max_tokens": 0, "system_prompt": "Test"},
        ]
        
        for config in invalid_configs:
            response = client.post("/api/v1/agents/", json=config, headers=auth_headers)
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_chat_message_validation(self, client, auth_headers, test_agent_config):
        """Test chat message validation."""
        # Create an agent first
        create_response = client.post("/api/v1/agents/", json=test_agent_config, headers=auth_headers)
        agent_id = create_response.json()["id"]
        
        # Test empty message
        response = client.post(f"/api/v1/agents/{agent_id}/chat", 
                             json={"message": ""}, headers=auth_headers)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Test missing message
        response = client.post(f"/api/v1/agents/{agent_id}/chat", 
                             json={}, headers=auth_headers)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


if __name__ == "__main__":
    pytest.main([__file__])


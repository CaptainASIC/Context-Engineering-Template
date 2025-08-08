"""
AI Agents router with PydanticAI integration.

This module provides endpoints for managing AI agents, handling multi-LLM
conversations, and RAG (Retrieval-Augmented Generation) functionality.
"""

from datetime import datetime
from typing import Annotated, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from loguru import logger
from pydantic import BaseModel, Field

from src.auth.router import get_current_active_user
from src.common.exceptions import (
    AgentException,
    AgentNotFoundException,
    ModelProviderException,
    ValidationException
)

router = APIRouter()

# Agent models
class AgentConfig(BaseModel):
    """Agent configuration model."""
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    model_provider: str = Field(..., description="LLM provider (openai, anthropic, google, ollama)")
    model_name: str = Field(..., description="Specific model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: int = Field(default=1000, ge=1, le=4000, description="Maximum tokens")
    system_prompt: str = Field(..., description="System prompt for the agent")
    tools_enabled: List[str] = Field(default=[], description="Enabled tools for the agent")
    rag_enabled: bool = Field(default=False, description="Enable RAG functionality")
    memory_enabled: bool = Field(default=True, description="Enable conversation memory")


class Agent(BaseModel):
    """Agent model."""
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    config: AgentConfig
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        from_attributes = True


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict] = Field(default={}, description="Additional metadata")


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="User message")
    context: Optional[Dict] = Field(default={}, description="Additional context")
    stream: bool = Field(default=False, description="Enable streaming response")


class ChatResponse(BaseModel):
    """Chat response model."""
    message: str = Field(..., description="Agent response")
    agent_id: UUID = Field(..., description="Agent ID")
    conversation_id: UUID = Field(..., description="Conversation ID")
    metadata: Dict = Field(default={}, description="Response metadata")
    sources: List[Dict] = Field(default=[], description="RAG sources if applicable")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Conversation(BaseModel):
    """Conversation model."""
    id: UUID = Field(default_factory=uuid4)
    agent_id: UUID
    user_id: UUID
    title: str = Field(..., description="Conversation title")
    messages: List[ChatMessage] = Field(default=[], description="Conversation messages")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        from_attributes = True


class AgentStats(BaseModel):
    """Agent statistics model."""
    total_conversations: int
    total_messages: int
    average_response_time: float
    success_rate: float
    last_used: Optional[datetime]


# Mock data storage (in production, use a real database)
agents_db: Dict[UUID, Agent] = {}
conversations_db: Dict[UUID, Conversation] = {}

# Supported model providers
SUPPORTED_PROVIDERS = {
    "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
    "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
    "google": ["gemini-pro", "gemini-pro-vision"],
    "ollama": ["llama2", "mistral", "codellama"]
}


def validate_model_provider(provider: str, model: str) -> bool:
    """Validate model provider and model combination."""
    if provider not in SUPPORTED_PROVIDERS:
        return False
    return model in SUPPORTED_PROVIDERS[provider]


async def simulate_llm_response(message: str, config: AgentConfig) -> str:
    """
    Simulate LLM response based on provider and configuration.
    In production, this would integrate with actual LLM APIs.
    """
    # Simulate different provider responses
    provider_responses = {
        "openai": f"[OpenAI {config.model_name}] I understand your message: '{message}'. How can I help you further?",
        "anthropic": f"[Anthropic {config.model_name}] Thank you for your message: '{message}'. I'm here to assist you.",
        "google": f"[Google {config.model_name}] I've received your message: '{message}'. Let me help you with that.",
        "ollama": f"[Ollama {config.model_name}] Processing your message: '{message}'. Here's my response."
    }
    
    base_response = provider_responses.get(config.model_provider, "I'm processing your request.")
    
    # Add system prompt context if available
    if config.system_prompt:
        base_response += f" (Following system prompt: {config.system_prompt[:50]}...)"
    
    return base_response


@router.post("/", response_model=Agent)
async def create_agent(
    agent_config: AgentConfig,
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> Agent:
    """
    Create a new AI agent.
    
    - **name**: Agent name
    - **description**: Agent description
    - **model_provider**: LLM provider (openai, anthropic, google, ollama)
    - **model_name**: Specific model name
    - **system_prompt**: System prompt for the agent
    - **tools_enabled**: List of enabled tools
    - **rag_enabled**: Enable RAG functionality
    - **memory_enabled**: Enable conversation memory
    """
    # Validate model provider and model
    if not validate_model_provider(agent_config.model_provider, agent_config.model_name):
        raise ValidationException(
            f"Invalid model provider '{agent_config.model_provider}' or model '{agent_config.model_name}'"
        )
    
    agent = Agent(
        user_id=UUID(current_user["id"]),
        config=agent_config
    )
    
    agents_db[agent.id] = agent
    
    logger.info(f"Created agent {agent.id} for user {current_user['email']}")
    
    return agent


@router.get("/", response_model=List[Agent])
async def list_agents(
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> List[Agent]:
    """
    List all agents for the current user.
    
    Returns a list of agents owned by the authenticated user.
    """
    user_agents = [
        agent for agent in agents_db.values()
        if agent.user_id == UUID(current_user["id"])
    ]
    
    return user_agents


@router.get("/{agent_id}", response_model=Agent)
async def get_agent(
    agent_id: UUID,
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> Agent:
    """
    Get a specific agent by ID.
    
    Returns agent details if the user owns the agent.
    """
    agent = agents_db.get(agent_id)
    if not agent:
        raise AgentNotFoundException(str(agent_id))
    
    if agent.user_id != UUID(current_user["id"]):
        raise AgentNotFoundException(str(agent_id))
    
    return agent


@router.put("/{agent_id}", response_model=Agent)
async def update_agent(
    agent_id: UUID,
    agent_config: AgentConfig,
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> Agent:
    """
    Update an existing agent.
    
    Updates the agent configuration with new settings.
    """
    agent = agents_db.get(agent_id)
    if not agent:
        raise AgentNotFoundException(str(agent_id))
    
    if agent.user_id != UUID(current_user["id"]):
        raise AgentNotFoundException(str(agent_id))
    
    # Validate new model provider and model
    if not validate_model_provider(agent_config.model_provider, agent_config.model_name):
        raise ValidationException(
            f"Invalid model provider '{agent_config.model_provider}' or model '{agent_config.model_name}'"
        )
    
    agent.config = agent_config
    agent.updated_at = datetime.utcnow()
    
    logger.info(f"Updated agent {agent_id} for user {current_user['email']}")
    
    return agent


@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: UUID,
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> dict:
    """
    Delete an agent.
    
    Permanently removes the agent and all associated conversations.
    """
    agent = agents_db.get(agent_id)
    if not agent:
        raise AgentNotFoundException(str(agent_id))
    
    if agent.user_id != UUID(current_user["id"]):
        raise AgentNotFoundException(str(agent_id))
    
    # Remove agent and associated conversations
    del agents_db[agent_id]
    
    # Remove conversations for this agent
    conversations_to_remove = [
        conv_id for conv_id, conv in conversations_db.items()
        if conv.agent_id == agent_id
    ]
    for conv_id in conversations_to_remove:
        del conversations_db[conv_id]
    
    logger.info(f"Deleted agent {agent_id} and {len(conversations_to_remove)} conversations")
    
    return {"message": f"Agent {agent_id} deleted successfully"}


@router.post("/{agent_id}/chat", response_model=ChatResponse)
async def chat_with_agent(
    agent_id: UUID,
    chat_request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> ChatResponse:
    """
    Chat with an AI agent.
    
    - **message**: User message
    - **context**: Additional context for the conversation
    - **stream**: Enable streaming response (not implemented in this example)
    
    Returns the agent's response with metadata and sources if RAG is enabled.
    """
    agent = agents_db.get(agent_id)
    if not agent:
        raise AgentNotFoundException(str(agent_id))
    
    if agent.user_id != UUID(current_user["id"]):
        raise AgentNotFoundException(str(agent_id))
    
    if not agent.is_active:
        raise AgentException("Agent is not active")
    
    try:
        # Find or create conversation
        conversation = None
        for conv in conversations_db.values():
            if conv.agent_id == agent_id and conv.user_id == UUID(current_user["id"]):
                conversation = conv
                break
        
        if not conversation:
            conversation = Conversation(
                agent_id=agent_id,
                user_id=UUID(current_user["id"]),
                title=f"Chat with {agent.config.name}"
            )
            conversations_db[conversation.id] = conversation
        
        # Add user message to conversation
        user_message = ChatMessage(
            role="user",
            content=chat_request.message
        )
        conversation.messages.append(user_message)
        
        # Generate agent response
        response_content = await simulate_llm_response(chat_request.message, agent.config)
        
        # Add agent message to conversation
        agent_message = ChatMessage(
            role="assistant",
            content=response_content,
            metadata={
                "model_provider": agent.config.model_provider,
                "model_name": agent.config.model_name,
                "temperature": agent.config.temperature
            }
        )
        conversation.messages.append(agent_message)
        conversation.updated_at = datetime.utcnow()
        
        # Simulate RAG sources if enabled
        sources = []
        if agent.config.rag_enabled:
            sources = [
                {
                    "title": "Knowledge Base Article",
                    "content": "Relevant information from knowledge base...",
                    "similarity": 0.85,
                    "source": "internal_kb"
                }
            ]
        
        response = ChatResponse(
            message=response_content,
            agent_id=agent_id,
            conversation_id=conversation.id,
            metadata={
                "model_provider": agent.config.model_provider,
                "model_name": agent.config.model_name,
                "rag_enabled": agent.config.rag_enabled,
                "memory_enabled": agent.config.memory_enabled
            },
            sources=sources
        )
        
        logger.info(f"Chat response generated for agent {agent_id}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating chat response: {e}")
        raise AgentException(f"Failed to generate response: {str(e)}")


@router.get("/{agent_id}/conversations", response_model=List[Conversation])
async def get_agent_conversations(
    agent_id: UUID,
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> List[Conversation]:
    """
    Get all conversations for a specific agent.
    
    Returns a list of conversations for the specified agent.
    """
    agent = agents_db.get(agent_id)
    if not agent:
        raise AgentNotFoundException(str(agent_id))
    
    if agent.user_id != UUID(current_user["id"]):
        raise AgentNotFoundException(str(agent_id))
    
    agent_conversations = [
        conv for conv in conversations_db.values()
        if conv.agent_id == agent_id and conv.user_id == UUID(current_user["id"])
    ]
    
    return agent_conversations


@router.get("/{agent_id}/stats", response_model=AgentStats)
async def get_agent_stats(
    agent_id: UUID,
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> AgentStats:
    """
    Get agent statistics.
    
    Returns usage statistics for the specified agent.
    """
    agent = agents_db.get(agent_id)
    if not agent:
        raise AgentNotFoundException(str(agent_id))
    
    if agent.user_id != UUID(current_user["id"]):
        raise AgentNotFoundException(str(agent_id))
    
    # Calculate stats from conversations
    agent_conversations = [
        conv for conv in conversations_db.values()
        if conv.agent_id == agent_id
    ]
    
    total_messages = sum(len(conv.messages) for conv in agent_conversations)
    last_used = max((conv.updated_at for conv in agent_conversations), default=None)
    
    stats = AgentStats(
        total_conversations=len(agent_conversations),
        total_messages=total_messages,
        average_response_time=1.2,  # Simulated
        success_rate=0.95,  # Simulated
        last_used=last_used
    )
    
    return stats


@router.get("/providers/supported")
async def get_supported_providers() -> Dict[str, List[str]]:
    """
    Get supported model providers and their available models.
    
    Returns a dictionary of supported providers and their models.
    """
    return SUPPORTED_PROVIDERS


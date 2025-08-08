"""
Configuration settings for the PydanticAI RAG Agent.

This module provides configuration management for multi-LLM support,
vector databases, and retrieval settings.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class LLMProviderConfig(BaseModel):
    """Configuration for LLM providers."""
    name: str = Field(..., description="Provider name")
    api_key: Optional[str] = Field(None, description="API key")
    base_url: Optional[str] = Field(None, description="Base URL for API")
    model: str = Field(..., description="Model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=8000)
    timeout: int = Field(default=30, ge=1, le=300)
    enabled: bool = Field(default=True)


class VectorDBConfig(BaseModel):
    """Configuration for vector database."""
    provider: str = Field(default="chroma", description="Vector DB provider")
    collection_name: str = Field(default="rag_documents", description="Collection name")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model")
    dimension: int = Field(default=384, description="Vector dimension")
    distance_metric: str = Field(default="cosine", description="Distance metric")
    persist_directory: str = Field(default="./data/vectordb", description="Persistence directory")


class RetrievalConfig(BaseModel):
    """Configuration for retrieval settings."""
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")
    chunk_size: int = Field(default=500, ge=100, le=2000, description="Document chunk size")
    chunk_overlap: int = Field(default=50, ge=0, le=500, description="Chunk overlap")
    rerank: bool = Field(default=True, description="Enable reranking")
    rerank_top_k: int = Field(default=10, ge=1, le=50, description="Rerank top K")


class RAGSettings(BaseSettings):
    """Main RAG agent settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application settings
    app_name: str = Field(default="PydanticAI RAG Agent", description="Application name")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Log level")
    
    # LLM Provider configurations
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    google_api_key: Optional[str] = Field(default=None, description="Google API key")
    openrouter_api_key: Optional[str] = Field(default=None, description="OpenRouter API key")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    
    # Default LLM provider
    default_provider: str = Field(default="openai", description="Default LLM provider")
    
    # Vector database settings
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    
    # Retrieval settings
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    
    # Document processing
    supported_file_types: List[str] = Field(
        default=["pdf", "txt", "md", "docx", "html"],
        description="Supported file types"
    )
    max_file_size_mb: int = Field(default=10, description="Maximum file size in MB")
    
    # Agent settings
    agent_name: str = Field(default="RAG Assistant", description="Agent name")
    agent_description: str = Field(
        default="A helpful AI assistant with access to your documents",
        description="Agent description"
    )
    system_prompt: str = Field(
        default="""You are a helpful AI assistant with access to a knowledge base. 
        Use the provided context to answer questions accurately and cite your sources when possible.
        If you cannot find relevant information in the context, say so clearly.""",
        description="System prompt"
    )
    
    def get_llm_providers(self) -> Dict[str, LLMProviderConfig]:
        """Get configured LLM providers."""
        providers = {}
        
        # OpenAI
        if self.openai_api_key:
            providers["openai"] = LLMProviderConfig(
                name="openai",
                api_key=self.openai_api_key,
                model="gpt-4",
                temperature=0.7,
                max_tokens=1000
            )
        
        # Anthropic
        if self.anthropic_api_key:
            providers["anthropic"] = LLMProviderConfig(
                name="anthropic",
                api_key=self.anthropic_api_key,
                model="claude-3-sonnet-20240229",
                temperature=0.7,
                max_tokens=1000
            )
        
        # Google
        if self.google_api_key:
            providers["google"] = LLMProviderConfig(
                name="google",
                api_key=self.google_api_key,
                model="gemini-pro",
                temperature=0.7,
                max_tokens=1000
            )
        
        # OpenRouter
        if self.openrouter_api_key:
            providers["openrouter"] = LLMProviderConfig(
                name="openrouter",
                api_key=self.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                model="meta-llama/llama-3.1-8b-instruct:free",
                temperature=0.7,
                max_tokens=1000
            )
        
        # Ollama (always available if running)
        providers["ollama"] = LLMProviderConfig(
            name="ollama",
            base_url=self.ollama_base_url,
            model="llama2",
            temperature=0.7,
            max_tokens=1000
        )
        
        return providers
    
    def get_active_provider(self) -> Optional[LLMProviderConfig]:
        """Get the active LLM provider configuration."""
        providers = self.get_llm_providers()
        return providers.get(self.default_provider)
    
    def validate_settings(self) -> List[str]:
        """Validate settings and return any issues."""
        issues = []
        
        # Check if at least one LLM provider is configured
        providers = self.get_llm_providers()
        active_providers = [p for p in providers.values() if p.enabled]
        
        if not active_providers:
            issues.append("No LLM providers are configured and enabled")
        
        # Check if default provider is available
        if self.default_provider not in providers:
            issues.append(f"Default provider '{self.default_provider}' is not configured")
        
        # Check vector DB settings
        if not os.path.exists(os.path.dirname(self.vector_db.persist_directory)):
            try:
                os.makedirs(os.path.dirname(self.vector_db.persist_directory), exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create vector DB directory: {e}")
        
        return issues


# Global settings instance
settings = RAGSettings()


def get_settings() -> RAGSettings:
    """Get application settings."""
    return settings


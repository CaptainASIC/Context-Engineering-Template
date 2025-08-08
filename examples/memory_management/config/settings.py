"""
Configuration settings for mem0 Memory Management System.

This module provides configuration management for conversation context,
user preferences, and memory storage operations.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum
import os


class MemoryType(str, Enum):
    """Types of memory storage."""
    CONVERSATION = "conversation"
    USER_PREFERENCE = "user_preference"
    ENTITY = "entity"
    FACT = "fact"
    RELATIONSHIP = "relationship"


class VectorStoreType(str, Enum):
    """Supported vector store backends."""
    CHROMA = "chroma"
    FAISS = "faiss"
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"
    PINECONE = "pinecone"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE_OPENAI = "azure_openai"
    OLLAMA = "ollama"


class MemoryStorageConfig(BaseModel):
    """Memory storage configuration."""
    
    # Storage backend
    backend: str = Field(default="sqlite", description="Storage backend")
    connection_string: str = Field(default="sqlite:///memory.db", description="Connection string")
    
    # Vector store settings
    vector_store: VectorStoreType = Field(default=VectorStoreType.CHROMA, description="Vector store type")
    vector_dimension: int = Field(default=384, description="Vector dimension")
    
    # Collection settings
    collection_name: str = Field(default="memories", description="Default collection name")
    max_memories_per_user: int = Field(default=10000, description="Maximum memories per user")
    
    # Retention settings
    default_ttl_days: int = Field(default=365, description="Default TTL in days")
    cleanup_interval_hours: int = Field(default=24, description="Cleanup interval in hours")
    
    @validator('backend')
    def validate_backend(cls, v):
        allowed_backends = ['sqlite', 'postgresql', 'redis', 'memory']
        if v not in allowed_backends:
            raise ValueError(f'Backend must be one of: {allowed_backends}')
        return v


class LLMConfig(BaseModel):
    """LLM configuration for memory processing."""
    
    # Primary LLM
    provider: LLMProvider = Field(default=LLMProvider.OPENAI, description="LLM provider")
    model: str = Field(default="gpt-3.5-turbo", description="Model name")
    
    # API settings
    api_key: Optional[str] = Field(None, description="API key")
    api_base: Optional[str] = Field(None, description="API base URL")
    organization: Optional[str] = Field(None, description="Organization ID")
    
    # Model parameters
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Temperature")
    max_tokens: int = Field(default=1000, description="Maximum tokens")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling")
    
    # Fallback configuration
    fallback_provider: Optional[LLMProvider] = Field(None, description="Fallback provider")
    fallback_model: Optional[str] = Field(None, description="Fallback model")
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return v


class EmbeddingConfig(BaseModel):
    """Embedding configuration for memory vectors."""
    
    # Embedding model
    provider: str = Field(default="sentence-transformers", description="Embedding provider")
    model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model")
    dimension: int = Field(default=384, description="Embedding dimension")
    
    # Processing settings
    batch_size: int = Field(default=32, description="Batch size")
    normalize: bool = Field(default=True, description="Normalize embeddings")
    
    # Caching
    cache_embeddings: bool = Field(default=True, description="Cache embeddings")
    cache_size: int = Field(default=10000, description="Cache size")


class ConversationConfig(BaseModel):
    """Conversation memory configuration."""
    
    # Context window
    max_context_length: int = Field(default=4000, description="Maximum context length")
    context_overlap: int = Field(default=200, description="Context overlap")
    
    # Memory extraction
    extract_entities: bool = Field(default=True, description="Extract entities")
    extract_relationships: bool = Field(default=True, description="Extract relationships")
    extract_facts: bool = Field(default=True, description="Extract facts")
    
    # Summarization
    enable_summarization: bool = Field(default=True, description="Enable conversation summarization")
    summarization_threshold: int = Field(default=2000, description="Tokens before summarization")
    
    # Relevance scoring
    relevance_threshold: float = Field(default=0.7, description="Relevance threshold")
    max_relevant_memories: int = Field(default=10, description="Maximum relevant memories")


class UserPreferenceConfig(BaseModel):
    """User preference memory configuration."""
    
    # Preference categories
    track_preferences: List[str] = Field(
        default_factory=lambda: [
            "communication_style",
            "topics_of_interest",
            "expertise_areas",
            "learning_preferences",
            "interaction_patterns"
        ],
        description="Preference categories to track"
    )
    
    # Learning settings
    learning_rate: float = Field(default=0.1, description="Preference learning rate")
    confidence_threshold: float = Field(default=0.8, description="Confidence threshold")
    
    # Update frequency
    update_frequency: str = Field(default="immediate", description="Update frequency")
    batch_update_size: int = Field(default=10, description="Batch update size")


class MemoryManagementSettings(BaseSettings):
    """Main memory management settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application settings
    app_name: str = Field(default="Memory Management Service", description="Application name")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Log level")
    environment: str = Field(default="development", description="Environment")
    
    # Core configurations
    storage: MemoryStorageConfig = Field(default_factory=MemoryStorageConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    user_preference: UserPreferenceConfig = Field(default_factory=UserPreferenceConfig)
    
    # API settings
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=1, description="API workers")
    
    # Security settings
    enable_auth: bool = Field(default=False, description="Enable authentication")
    secret_key: str = Field(default="your-secret-key", description="Secret key")
    access_token_expire_minutes: int = Field(default=30, description="Token expiration")
    
    # Performance settings
    max_concurrent_operations: int = Field(default=10, description="Max concurrent operations")
    request_timeout: int = Field(default=30, description="Request timeout")
    
    # Monitoring settings
    enable_metrics: bool = Field(default=True, description="Enable metrics")
    metrics_port: int = Field(default=9090, description="Metrics port")
    
    @validator('storage', pre=True)
    def validate_storage_config(cls, v):
        if isinstance(v, dict):
            return MemoryStorageConfig(**v)
        return v
    
    @validator('llm', pre=True)
    def validate_llm_config(cls, v):
        if isinstance(v, dict):
            return LLMConfig(**v)
        return v
    
    @validator('embedding', pre=True)
    def validate_embedding_config(cls, v):
        if isinstance(v, dict):
            return EmbeddingConfig(**v)
        return v
    
    @validator('conversation', pre=True)
    def validate_conversation_config(cls, v):
        if isinstance(v, dict):
            return ConversationConfig(**v)
        return v
    
    @validator('user_preference', pre=True)
    def validate_user_preference_config(cls, v):
        if isinstance(v, dict):
            return UserPreferenceConfig(**v)
        return v
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration dictionary."""
        config = {
            "provider": self.llm.provider,
            "model": self.llm.model,
            "temperature": self.llm.temperature,
            "max_tokens": self.llm.max_tokens,
            "top_p": self.llm.top_p
        }
        
        if self.llm.api_key:
            config["api_key"] = self.llm.api_key
        if self.llm.api_base:
            config["api_base"] = self.llm.api_base
        if self.llm.organization:
            config["organization"] = self.llm.organization
        
        return config
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration dictionary."""
        return {
            "provider": self.embedding.provider,
            "model": self.embedding.model,
            "dimension": self.embedding.dimension,
            "batch_size": self.embedding.batch_size,
            "normalize": self.embedding.normalize
        }
    
    def get_vector_store_config(self) -> Dict[str, Any]:
        """Get vector store configuration."""
        config = {
            "type": self.storage.vector_store,
            "dimension": self.storage.vector_dimension,
            "collection_name": self.storage.collection_name
        }
        
        # Add provider-specific configurations
        if self.storage.vector_store == VectorStoreType.CHROMA:
            config["persist_directory"] = "./chroma_db"
        elif self.storage.vector_store == VectorStoreType.FAISS:
            config["index_path"] = "./faiss_index"
        elif self.storage.vector_store == VectorStoreType.QDRANT:
            config["host"] = os.getenv("QDRANT_HOST", "localhost")
            config["port"] = int(os.getenv("QDRANT_PORT", "6333"))
        
        return config
    
    def get_memory_types(self) -> List[str]:
        """Get list of supported memory types."""
        return [memory_type.value for memory_type in MemoryType]
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    def get_supported_llm_providers(self) -> List[str]:
        """Get list of supported LLM providers."""
        return [provider.value for provider in LLMProvider]
    
    def get_supported_vector_stores(self) -> List[str]:
        """Get list of supported vector stores."""
        return [store.value for store in VectorStoreType]
    
    def validate_memory_type(self, memory_type: str) -> bool:
        """Validate memory type."""
        return memory_type in self.get_memory_types()
    
    def get_retention_policy(self, memory_type: str) -> Dict[str, Any]:
        """Get retention policy for memory type."""
        base_policy = {
            "ttl_days": self.storage.default_ttl_days,
            "cleanup_interval_hours": self.storage.cleanup_interval_hours
        }
        
        # Customize based on memory type
        if memory_type == MemoryType.CONVERSATION:
            base_policy["ttl_days"] = 90  # Shorter retention for conversations
        elif memory_type == MemoryType.USER_PREFERENCE:
            base_policy["ttl_days"] = 730  # Longer retention for preferences
        elif memory_type == MemoryType.ENTITY:
            base_policy["ttl_days"] = 365  # Standard retention for entities
        
        return base_policy
    
    def get_privacy_settings(self) -> Dict[str, Any]:
        """Get privacy and data protection settings."""
        return {
            "anonymize_data": True,
            "encrypt_sensitive_data": self.is_production(),
            "data_retention_days": self.storage.default_ttl_days,
            "allow_data_export": True,
            "allow_data_deletion": True,
            "audit_access": self.is_production()
        }


# Global settings instance
settings = MemoryManagementSettings()


def get_settings() -> MemoryManagementSettings:
    """Get application settings."""
    return settings


def create_test_settings() -> MemoryManagementSettings:
    """Create test settings with in-memory storage."""
    test_settings = MemoryManagementSettings()
    test_settings.storage.backend = "memory"
    test_settings.storage.connection_string = "sqlite:///:memory:"
    test_settings.debug = True
    test_settings.log_level = "DEBUG"
    test_settings.environment = "test"
    return test_settings


def load_settings_from_file(file_path: str) -> MemoryManagementSettings:
    """Load settings from configuration file."""
    import yaml
    
    with open(file_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return MemoryManagementSettings(**config_data)


def save_settings_to_file(settings: MemoryManagementSettings, file_path: str):
    """Save settings to configuration file."""
    import yaml
    
    config_data = settings.dict()
    
    with open(file_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, indent=2)


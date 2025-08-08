"""
Configuration settings for Neon/PostgreSQL Vector Database with pgvector.

This module provides configuration management for database connections,
embedding models, and semantic search operations.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    
    # Connection settings
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    database: str = Field(default="vectordb", description="Database name")
    username: str = Field(default="postgres", description="Database username")
    password: str = Field(..., description="Database password")
    
    # SSL and security
    sslmode: str = Field(default="prefer", description="SSL mode")
    sslcert: Optional[str] = Field(None, description="SSL certificate path")
    sslkey: Optional[str] = Field(None, description="SSL key path")
    sslrootcert: Optional[str] = Field(None, description="SSL root certificate path")
    
    # Connection pool settings
    min_connections: int = Field(default=5, description="Minimum connections in pool")
    max_connections: int = Field(default=20, description="Maximum connections in pool")
    connection_timeout: int = Field(default=30, description="Connection timeout in seconds")
    command_timeout: int = Field(default=60, description="Command timeout in seconds")
    
    # Performance settings
    statement_cache_size: int = Field(default=1024, description="Statement cache size")
    max_cached_statement_lifetime: int = Field(default=300, description="Max cached statement lifetime")
    max_cacheable_statement_size: int = Field(default=1024, description="Max cacheable statement size")
    
    @validator('port')
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v
    
    @validator('sslmode')
    def validate_sslmode(cls, v):
        allowed_modes = ['disable', 'allow', 'prefer', 'require', 'verify-ca', 'verify-full']
        if v not in allowed_modes:
            raise ValueError(f'SSL mode must be one of: {allowed_modes}')
        return v
    
    def get_connection_url(self) -> str:
        """Get database connection URL."""
        url = f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        
        params = []
        if self.sslmode != "prefer":
            params.append(f"sslmode={self.sslmode}")
        if self.sslcert:
            params.append(f"sslcert={self.sslcert}")
        if self.sslkey:
            params.append(f"sslkey={self.sslkey}")
        if self.sslrootcert:
            params.append(f"sslrootcert={self.sslrootcert}")
        
        if params:
            url += "?" + "&".join(params)
        
        return url
    
    def get_async_connection_url(self) -> str:
        """Get async database connection URL."""
        return self.get_connection_url().replace("postgresql://", "postgresql+asyncpg://")


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    
    # Model settings
    model_name: str = Field(default="all-MiniLM-L6-v2", description="Embedding model name")
    model_provider: str = Field(default="sentence-transformers", description="Model provider")
    dimension: int = Field(default=384, description="Embedding dimension")
    max_sequence_length: int = Field(default=512, description="Maximum sequence length")
    
    # Processing settings
    batch_size: int = Field(default=32, description="Batch size for embedding generation")
    normalize_embeddings: bool = Field(default=True, description="Normalize embeddings")
    device: str = Field(default="cpu", description="Device for model inference")
    
    # Caching settings
    enable_cache: bool = Field(default=True, description="Enable embedding cache")
    cache_size: int = Field(default=10000, description="Cache size")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    
    @validator('dimension')
    def validate_dimension(cls, v):
        if v <= 0:
            raise ValueError('Dimension must be positive')
        return v
    
    @validator('device')
    def validate_device(cls, v):
        allowed_devices = ['cpu', 'cuda', 'mps', 'auto']
        if v not in allowed_devices:
            raise ValueError(f'Device must be one of: {allowed_devices}')
        return v


class SearchConfig(BaseModel):
    """Search configuration."""
    
    # Search parameters
    default_limit: int = Field(default=10, description="Default search result limit")
    max_limit: int = Field(default=100, description="Maximum search result limit")
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold")
    
    # Distance metrics
    distance_metric: str = Field(default="cosine", description="Distance metric for similarity")
    
    # Indexing settings
    index_type: str = Field(default="ivfflat", description="Vector index type")
    index_lists: int = Field(default=100, description="Number of lists for IVF index")
    index_probes: int = Field(default=10, description="Number of probes for search")
    
    # Performance settings
    enable_parallel_search: bool = Field(default=True, description="Enable parallel search")
    search_timeout: int = Field(default=30, description="Search timeout in seconds")
    
    @validator('similarity_threshold')
    def validate_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Similarity threshold must be between 0 and 1')
        return v
    
    @validator('distance_metric')
    def validate_distance_metric(cls, v):
        allowed_metrics = ['cosine', 'l2', 'inner_product']
        if v not in allowed_metrics:
            raise ValueError(f'Distance metric must be one of: {allowed_metrics}')
        return v
    
    @validator('index_type')
    def validate_index_type(cls, v):
        allowed_types = ['ivfflat', 'hnsw']
        if v not in allowed_types:
            raise ValueError(f'Index type must be one of: {allowed_types}')
        return v


class VectorDBSettings(BaseSettings):
    """Main vector database settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application settings
    app_name: str = Field(default="Vector Database Service", description="Application name")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Log level")
    environment: str = Field(default="development", description="Environment")
    
    # Database configuration
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    
    # Embedding configuration
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    
    # Search configuration
    search: SearchConfig = Field(default_factory=SearchConfig)
    
    # Data processing settings
    chunk_size: int = Field(default=1000, description="Default chunk size for text processing")
    chunk_overlap: int = Field(default=200, description="Chunk overlap size")
    max_file_size_mb: int = Field(default=50, description="Maximum file size in MB")
    
    # Batch processing settings
    batch_size: int = Field(default=100, description="Batch size for bulk operations")
    max_concurrent_operations: int = Field(default=10, description="Max concurrent operations")
    
    # Monitoring settings
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=8080, description="Metrics server port")
    
    # Cache settings
    redis_url: Optional[str] = Field(None, description="Redis URL for caching")
    enable_query_cache: bool = Field(default=True, description="Enable query result caching")
    query_cache_ttl: int = Field(default=300, description="Query cache TTL in seconds")
    
    @validator('database', pre=True)
    def validate_database_config(cls, v):
        if isinstance(v, dict):
            return DatabaseConfig(**v)
        return v
    
    @validator('embedding', pre=True)
    def validate_embedding_config(cls, v):
        if isinstance(v, dict):
            return EmbeddingConfig(**v)
        return v
    
    @validator('search', pre=True)
    def validate_search_config(cls, v):
        if isinstance(v, dict):
            return SearchConfig(**v)
        return v
    
    def get_database_url(self) -> str:
        """Get database connection URL."""
        return self.database.get_connection_url()
    
    def get_async_database_url(self) -> str:
        """Get async database connection URL."""
        return self.database.get_async_connection_url()
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    def get_supported_file_types(self) -> List[str]:
        """Get list of supported file types."""
        return ["txt", "md", "pdf", "docx", "html", "json"]
    
    def validate_file_size(self, size_bytes: int) -> bool:
        """Validate file size against limits."""
        max_size_bytes = self.max_file_size_mb * 1024 * 1024
        return size_bytes <= max_size_bytes
    
    def get_embedding_cache_config(self) -> Dict[str, Any]:
        """Get embedding cache configuration."""
        return {
            "enabled": self.embedding.enable_cache,
            "size": self.embedding.cache_size,
            "ttl": self.embedding.cache_ttl
        }
    
    def get_search_parameters(self) -> Dict[str, Any]:
        """Get default search parameters."""
        return {
            "limit": self.search.default_limit,
            "threshold": self.search.similarity_threshold,
            "distance_metric": self.search.distance_metric,
            "timeout": self.search.search_timeout
        }
    
    def get_index_parameters(self) -> Dict[str, Any]:
        """Get vector index parameters."""
        return {
            "type": self.search.index_type,
            "lists": self.search.index_lists,
            "probes": self.search.index_probes
        }


class NeonConfig(BaseModel):
    """Neon-specific configuration."""
    
    # Neon connection settings
    project_id: Optional[str] = Field(None, description="Neon project ID")
    branch_name: str = Field(default="main", description="Neon branch name")
    region: Optional[str] = Field(None, description="Neon region")
    
    # API settings
    api_key: Optional[str] = Field(None, description="Neon API key")
    api_base_url: str = Field(default="https://console.neon.tech/api/v2", description="Neon API base URL")
    
    # Connection pooling (Neon-specific)
    enable_connection_pooling: bool = Field(default=True, description="Enable connection pooling")
    pooler_mode: str = Field(default="transaction", description="Pooler mode")
    
    @validator('pooler_mode')
    def validate_pooler_mode(cls, v):
        allowed_modes = ['session', 'transaction', 'statement']
        if v not in allowed_modes:
            raise ValueError(f'Pooler mode must be one of: {allowed_modes}')
        return v
    
    def get_neon_connection_string(self, database_config: DatabaseConfig) -> str:
        """Get Neon-specific connection string."""
        if not self.project_id:
            return database_config.get_connection_url()
        
        # Construct Neon connection string
        host = f"{self.project_id}.{self.region}.aws.neon.tech" if self.region else f"{self.project_id}.aws.neon.tech"
        
        url = f"postgresql://{database_config.username}:{database_config.password}@{host}:{database_config.port}/{database_config.database}"
        
        params = ["sslmode=require"]
        if self.enable_connection_pooling:
            params.append(f"pooler={self.pooler_mode}")
        
        return url + "?" + "&".join(params)


# Enhanced settings with Neon support
class EnhancedVectorDBSettings(VectorDBSettings):
    """Enhanced vector database settings with Neon support."""
    
    # Neon configuration
    neon: Optional[NeonConfig] = Field(None, description="Neon-specific configuration")
    
    @validator('neon', pre=True)
    def validate_neon_config(cls, v):
        if v and isinstance(v, dict):
            return NeonConfig(**v)
        return v
    
    def get_connection_url(self) -> str:
        """Get connection URL with Neon support."""
        if self.neon and self.neon.project_id:
            return self.neon.get_neon_connection_string(self.database)
        return self.database.get_connection_url()
    
    def get_async_connection_url(self) -> str:
        """Get async connection URL with Neon support."""
        url = self.get_connection_url()
        return url.replace("postgresql://", "postgresql+asyncpg://")
    
    def is_neon_database(self) -> bool:
        """Check if using Neon database."""
        return self.neon is not None and self.neon.project_id is not None


# Global settings instance
settings = EnhancedVectorDBSettings()


def get_settings() -> EnhancedVectorDBSettings:
    """Get application settings."""
    return settings


def create_test_settings() -> EnhancedVectorDBSettings:
    """Create test settings with in-memory/test database."""
    test_settings = EnhancedVectorDBSettings()
    test_settings.database.database = "test_vectordb"
    test_settings.database.host = "localhost"
    test_settings.debug = True
    test_settings.log_level = "DEBUG"
    test_settings.environment = "test"
    return test_settings


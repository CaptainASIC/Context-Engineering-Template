"""
Configuration settings for Neo4j Knowledge Graph integration.

This module provides configuration management for Neo4j connections,
graph schemas, and knowledge graph operations.
"""

from typing import Dict, List, Optional, Any, Set
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class Neo4jConfig(BaseModel):
    """Neo4j database configuration."""
    uri: str = Field(default="bolt://localhost:7687", description="Neo4j URI")
    username: str = Field(default="neo4j", description="Neo4j username")
    password: str = Field(..., description="Neo4j password")
    database: str = Field(default="neo4j", description="Neo4j database name")
    max_connection_lifetime: int = Field(default=30, description="Max connection lifetime in seconds")
    max_connection_pool_size: int = Field(default=50, description="Max connection pool size")
    connection_acquisition_timeout: int = Field(default=60, description="Connection timeout in seconds")
    encrypted: bool = Field(default=False, description="Use encrypted connection")
    trust: str = Field(default="TRUST_ALL_CERTIFICATES", description="Trust strategy")


class GraphSchemaConfig(BaseModel):
    """Graph schema configuration."""
    # Node types and their properties
    node_types: Dict[str, Dict[str, str]] = Field(
        default={
            "Person": {
                "name": "string",
                "email": "string",
                "title": "string",
                "department": "string",
                "skills": "list",
                "created_at": "datetime",
                "updated_at": "datetime"
            },
            "Policy": {
                "name": "string",
                "description": "string",
                "version": "string",
                "status": "string",
                "category": "string",
                "tags": "list",
                "created_at": "datetime",
                "updated_at": "datetime"
            },
            "UseCase": {
                "name": "string",
                "description": "string",
                "priority": "string",
                "status": "string",
                "business_value": "string",
                "complexity": "string",
                "tags": "list",
                "created_at": "datetime",
                "updated_at": "datetime"
            },
            "Document": {
                "title": "string",
                "content": "string",
                "type": "string",
                "source": "string",
                "author": "string",
                "tags": "list",
                "created_at": "datetime",
                "updated_at": "datetime"
            },
            "Organization": {
                "name": "string",
                "type": "string",
                "industry": "string",
                "size": "string",
                "location": "string",
                "created_at": "datetime",
                "updated_at": "datetime"
            },
            "Concept": {
                "name": "string",
                "definition": "string",
                "category": "string",
                "synonyms": "list",
                "related_terms": "list",
                "created_at": "datetime",
                "updated_at": "datetime"
            }
        },
        description="Node types and their property schemas"
    )
    
    # Relationship types and their properties
    relationship_types: Dict[str, Dict[str, str]] = Field(
        default={
            "WORKS_FOR": {
                "role": "string",
                "start_date": "datetime",
                "end_date": "datetime",
                "is_current": "boolean"
            },
            "MANAGES": {
                "start_date": "datetime",
                "end_date": "datetime",
                "is_current": "boolean"
            },
            "IMPLEMENTS": {
                "implementation_date": "datetime",
                "status": "string",
                "notes": "string"
            },
            "RELATES_TO": {
                "relationship_type": "string",
                "strength": "float",
                "confidence": "float",
                "source": "string"
            },
            "AUTHORED": {
                "authored_date": "datetime",
                "role": "string"
            },
            "REFERENCES": {
                "reference_type": "string",
                "page_number": "integer",
                "section": "string"
            },
            "PART_OF": {
                "hierarchy_level": "integer",
                "percentage": "float"
            },
            "DEPENDS_ON": {
                "dependency_type": "string",
                "criticality": "string",
                "notes": "string"
            },
            "SIMILAR_TO": {
                "similarity_score": "float",
                "similarity_type": "string",
                "algorithm": "string"
            }
        },
        description="Relationship types and their property schemas"
    )
    
    # Required indexes for performance
    indexes: List[Dict[str, Any]] = Field(
        default=[
            {"label": "Person", "properties": ["name", "email"]},
            {"label": "Policy", "properties": ["name", "version"]},
            {"label": "UseCase", "properties": ["name", "status"]},
            {"label": "Document", "properties": ["title", "type"]},
            {"label": "Organization", "properties": ["name"]},
            {"label": "Concept", "properties": ["name", "category"]}
        ],
        description="Database indexes to create"
    )
    
    # Constraints for data integrity
    constraints: List[Dict[str, Any]] = Field(
        default=[
            {"label": "Person", "property": "email", "type": "unique"},
            {"label": "Policy", "property": "name", "type": "unique"},
            {"label": "Organization", "property": "name", "type": "unique"},
            {"label": "Document", "property": "title", "type": "unique"}
        ],
        description="Database constraints to create"
    )


class KnowledgeGraphSettings(BaseSettings):
    """Main knowledge graph settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application settings
    app_name: str = Field(default="Neo4j Knowledge Graph", description="Application name")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Log level")
    
    # Neo4j configuration
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    
    # Graph schema
    schema: GraphSchemaConfig = Field(default_factory=GraphSchemaConfig)
    
    # Data processing settings
    batch_size: int = Field(default=1000, description="Batch size for bulk operations")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries in seconds")
    
    # Text processing
    enable_nlp: bool = Field(default=True, description="Enable NLP processing")
    nlp_model: str = Field(default="en_core_web_sm", description="SpaCy model for NLP")
    similarity_threshold: float = Field(default=0.8, description="Similarity threshold for relationships")
    
    # Graph analysis
    enable_graph_algorithms: bool = Field(default=True, description="Enable graph algorithms")
    max_path_length: int = Field(default=5, description="Maximum path length for traversals")
    
    # Import/Export settings
    export_format: str = Field(default="json", description="Default export format")
    import_batch_size: int = Field(default=500, description="Import batch size")
    
    @validator('neo4j', pre=True)
    def validate_neo4j_config(cls, v):
        if isinstance(v, dict):
            return Neo4jConfig(**v)
        return v
    
    def get_neo4j_uri(self) -> str:
        """Get Neo4j connection URI."""
        return self.neo4j.uri
    
    def get_neo4j_auth(self) -> tuple:
        """Get Neo4j authentication tuple."""
        return (self.neo4j.username, self.neo4j.password)
    
    def get_supported_node_types(self) -> Set[str]:
        """Get set of supported node types."""
        return set(self.schema.node_types.keys())
    
    def get_supported_relationship_types(self) -> Set[str]:
        """Get set of supported relationship types."""
        return set(self.schema.relationship_types.keys())
    
    def get_node_schema(self, node_type: str) -> Optional[Dict[str, str]]:
        """Get schema for a specific node type."""
        return self.schema.node_types.get(node_type)
    
    def get_relationship_schema(self, rel_type: str) -> Optional[Dict[str, str]]:
        """Get schema for a specific relationship type."""
        return self.schema.relationship_types.get(rel_type)
    
    def validate_node_properties(self, node_type: str, properties: Dict[str, Any]) -> List[str]:
        """
        Validate node properties against schema.
        
        Args:
            node_type: Type of node
            properties: Properties to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        schema = self.get_node_schema(node_type)
        
        if not schema:
            errors.append(f"Unknown node type: {node_type}")
            return errors
        
        # Check required properties (name is typically required)
        if 'name' in schema and 'name' not in properties:
            errors.append(f"Missing required property 'name' for {node_type}")
        
        # Validate property types (simplified validation)
        for prop, value in properties.items():
            if prop in schema:
                expected_type = schema[prop]
                if expected_type == "string" and not isinstance(value, str):
                    errors.append(f"Property '{prop}' should be string, got {type(value).__name__}")
                elif expected_type == "integer" and not isinstance(value, int):
                    errors.append(f"Property '{prop}' should be integer, got {type(value).__name__}")
                elif expected_type == "float" and not isinstance(value, (int, float)):
                    errors.append(f"Property '{prop}' should be float, got {type(value).__name__}")
                elif expected_type == "boolean" and not isinstance(value, bool):
                    errors.append(f"Property '{prop}' should be boolean, got {type(value).__name__}")
                elif expected_type == "list" and not isinstance(value, list):
                    errors.append(f"Property '{prop}' should be list, got {type(value).__name__}")
        
        return errors
    
    def validate_relationship_properties(self, rel_type: str, properties: Dict[str, Any]) -> List[str]:
        """
        Validate relationship properties against schema.
        
        Args:
            rel_type: Type of relationship
            properties: Properties to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        schema = self.get_relationship_schema(rel_type)
        
        if not schema:
            errors.append(f"Unknown relationship type: {rel_type}")
            return errors
        
        # Validate property types (similar to node validation)
        for prop, value in properties.items():
            if prop in schema:
                expected_type = schema[prop]
                if expected_type == "string" and not isinstance(value, str):
                    errors.append(f"Property '{prop}' should be string, got {type(value).__name__}")
                elif expected_type == "float" and not isinstance(value, (int, float)):
                    errors.append(f"Property '{prop}' should be float, got {type(value).__name__}")
                elif expected_type == "boolean" and not isinstance(value, bool):
                    errors.append(f"Property '{prop}' should be boolean, got {type(value).__name__}")
        
        return errors


# Global settings instance
settings = KnowledgeGraphSettings()


def get_settings() -> KnowledgeGraphSettings:
    """Get application settings."""
    return settings


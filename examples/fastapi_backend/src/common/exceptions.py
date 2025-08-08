"""
Custom exceptions for the Context Engineering Platform.

This module defines application-specific exceptions with proper error codes,
HTTP status codes, and detailed error messages.
"""

from typing import Any, Dict, Optional


class ContextEngineeringException(Exception):
    """Base exception for Context Engineering Platform."""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


# Authentication Exceptions
class AuthenticationException(ContextEngineeringException):
    """Authentication related exceptions."""
    
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_FAILED",
            status_code=401,
            details=details
        )


class AuthorizationException(ContextEngineeringException):
    """Authorization related exceptions."""
    
    def __init__(self, message: str = "Access denied", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="ACCESS_DENIED",
            status_code=403,
            details=details
        )


class InvalidTokenException(ContextEngineeringException):
    """Invalid or expired token exceptions."""
    
    def __init__(self, message: str = "Invalid or expired token", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="INVALID_TOKEN",
            status_code=401,
            details=details
        )


# Agent Exceptions
class AgentException(ContextEngineeringException):
    """AI Agent related exceptions."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AGENT_ERROR",
            status_code=500,
            details=details
        )


class AgentNotFoundException(ContextEngineeringException):
    """Agent not found exception."""
    
    def __init__(self, agent_id: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Agent with ID '{agent_id}' not found",
            error_code="AGENT_NOT_FOUND",
            status_code=404,
            details=details
        )


class ModelProviderException(ContextEngineeringException):
    """LLM model provider exceptions."""
    
    def __init__(self, provider: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Error with {provider}: {message}",
            error_code="MODEL_PROVIDER_ERROR",
            status_code=502,
            details=details
        )


# Knowledge Graph Exceptions
class KnowledgeGraphException(ContextEngineeringException):
    """Knowledge graph related exceptions."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="KNOWLEDGE_GRAPH_ERROR",
            status_code=500,
            details=details
        )


class EntityNotFoundException(ContextEngineeringException):
    """Entity not found in knowledge graph."""
    
    def __init__(self, entity_id: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Entity with ID '{entity_id}' not found",
            error_code="ENTITY_NOT_FOUND",
            status_code=404,
            details=details
        )


class RelationshipException(ContextEngineeringException):
    """Relationship creation/modification exceptions."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="RELATIONSHIP_ERROR",
            status_code=400,
            details=details
        )


# Vector Database Exceptions
class VectorDatabaseException(ContextEngineeringException):
    """Vector database related exceptions."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VECTOR_DATABASE_ERROR",
            status_code=500,
            details=details
        )


class EmbeddingException(ContextEngineeringException):
    """Embedding generation exceptions."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="EMBEDDING_ERROR",
            status_code=500,
            details=details
        )


class SimilaritySearchException(ContextEngineeringException):
    """Vector similarity search exceptions."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="SIMILARITY_SEARCH_ERROR",
            status_code=500,
            details=details
        )


# Memory Management Exceptions
class MemoryException(ContextEngineeringException):
    """Memory management related exceptions."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="MEMORY_ERROR",
            status_code=500,
            details=details
        )


class MemoryNotFoundException(ContextEngineeringException):
    """Memory not found exception."""
    
    def __init__(self, memory_id: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Memory with ID '{memory_id}' not found",
            error_code="MEMORY_NOT_FOUND",
            status_code=404,
            details=details
        )


class ContextException(ContextEngineeringException):
    """Context processing exceptions."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="CONTEXT_ERROR",
            status_code=400,
            details=details
        )


# Validation Exceptions
class ValidationException(ContextEngineeringException):
    """Data validation exceptions."""
    
    def __init__(self, message: str, field: str = None, details: Optional[Dict[str, Any]] = None):
        if field:
            message = f"Validation error for field '{field}': {message}"
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=422,
            details=details
        )


# Rate Limiting Exceptions
class RateLimitException(ContextEngineeringException):
    """Rate limiting exceptions."""
    
    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details=details
        )


# External Service Exceptions
class ExternalServiceException(ContextEngineeringException):
    """External service integration exceptions."""
    
    def __init__(self, service: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Error with external service '{service}': {message}",
            error_code="EXTERNAL_SERVICE_ERROR",
            status_code=502,
            details=details
        )


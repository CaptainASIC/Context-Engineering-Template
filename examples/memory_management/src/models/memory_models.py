"""
Memory Models for mem0 Memory Management System.

This module defines Pydantic models for different types of memories
including conversations, user preferences, entities, and facts.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from uuid import uuid4, UUID
from enum import Enum

from pydantic import BaseModel, Field, validator


class MemoryType(str, Enum):
    """Types of memory storage."""
    CONVERSATION = "conversation"
    USER_PREFERENCE = "user_preference"
    ENTITY = "entity"
    FACT = "fact"
    RELATIONSHIP = "relationship"


class MemorySource(str, Enum):
    """Sources of memory information."""
    EXPLICIT = "explicit"  # Directly provided by user
    INFERRED = "inferred"  # Inferred from conversation
    LEARNED = "learned"    # Learned from patterns
    SYSTEM = "system"      # System generated


class BaseMemoryRecord(BaseModel):
    """Base class for all memory records."""
    
    memory_id: Optional[str] = Field(None, description="Memory identifier")
    user_id: str = Field(..., description="User identifier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Memory metadata
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    source: MemorySource = Field(default=MemorySource.EXPLICIT, description="Memory source")
    tags: List[str] = Field(default_factory=list, description="Memory tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.dict(exclude_none=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create instance from dictionary."""
        return cls(**data)


class ConversationMemory(BaseMemoryRecord):
    """Conversation memory record."""
    
    conversation_id: str = Field(..., description="Conversation identifier")
    message: str = Field(..., description="Message content")
    role: str = Field(..., description="Message role (user, assistant, system)")
    
    # Message metadata
    message_index: Optional[int] = Field(None, description="Message index in conversation")
    tokens: Optional[int] = Field(None, description="Token count")
    
    # Context information
    context_summary: Optional[str] = Field(None, description="Context summary")
    extracted_entities: List[str] = Field(default_factory=list, description="Extracted entities")
    extracted_facts: List[str] = Field(default_factory=list, description="Extracted facts")
    
    @validator('role')
    def validate_role(cls, v):
        allowed_roles = ['user', 'assistant', 'system', 'function']
        if v not in allowed_roles:
            raise ValueError(f'Role must be one of: {allowed_roles}')
        return v
    
    def get_memory_type(self) -> str:
        return MemoryType.CONVERSATION.value


class UserPreference(BaseMemoryRecord):
    """User preference memory record."""
    
    preference_type: str = Field(..., description="Type of preference")
    preference_value: Any = Field(..., description="Preference value")
    
    # Preference metadata
    category: Optional[str] = Field(None, description="Preference category")
    priority: int = Field(default=1, ge=1, le=10, description="Preference priority")
    
    # Learning information
    learning_source: Optional[str] = Field(None, description="How preference was learned")
    reinforcement_count: int = Field(default=1, description="Number of times reinforced")
    last_reinforced: Optional[datetime] = Field(None, description="Last reinforcement time")
    
    @validator('preference_type')
    def validate_preference_type(cls, v):
        # Common preference types
        common_types = [
            'communication_style', 'topics_of_interest', 'expertise_areas',
            'learning_preferences', 'interaction_patterns', 'language_preference',
            'response_length', 'formality_level', 'technical_depth'
        ]
        # Allow custom types but log them
        if v not in common_types:
            import logging
            logging.getLogger(__name__).debug(f"Custom preference type: {v}")
        return v
    
    def get_memory_type(self) -> str:
        return MemoryType.USER_PREFERENCE.value
    
    def reinforce(self):
        """Reinforce this preference."""
        self.reinforcement_count += 1
        self.last_reinforced = datetime.now(timezone.utc)
        self.confidence = min(1.0, self.confidence + 0.1)  # Increase confidence


class EntityMemory(BaseMemoryRecord):
    """Entity memory record."""
    
    entity_name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type")
    
    # Entity properties
    properties: Dict[str, Any] = Field(default_factory=dict, description="Entity properties")
    aliases: List[str] = Field(default_factory=list, description="Entity aliases")
    
    # Relationships
    relationships: List[Dict[str, Any]] = Field(default_factory=list, description="Entity relationships")
    
    # Context information
    first_mentioned: Optional[datetime] = Field(None, description="First mention time")
    last_mentioned: Optional[datetime] = Field(None, description="Last mention time")
    mention_count: int = Field(default=1, description="Number of mentions")
    
    @validator('entity_type')
    def validate_entity_type(cls, v):
        # Common entity types
        common_types = [
            'person', 'organization', 'location', 'product', 'concept',
            'event', 'document', 'technology', 'skill', 'topic'
        ]
        # Allow custom types
        return v.lower()
    
    def get_memory_type(self) -> str:
        return MemoryType.ENTITY.value
    
    def add_relationship(self, target_entity: str, relationship_type: str, properties: Optional[Dict[str, Any]] = None):
        """Add relationship to another entity."""
        relationship = {
            "target_entity": target_entity,
            "relationship_type": relationship_type,
            "properties": properties or {},
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        self.relationships.append(relationship)
    
    def update_mention(self):
        """Update mention statistics."""
        now = datetime.now(timezone.utc)
        if not self.first_mentioned:
            self.first_mentioned = now
        self.last_mentioned = now
        self.mention_count += 1


class FactMemory(BaseMemoryRecord):
    """Fact memory record."""
    
    fact: str = Field(..., description="Fact statement")
    fact_type: str = Field(default="general", description="Type of fact")
    
    # Fact validation
    verified: bool = Field(default=False, description="Whether fact is verified")
    verification_source: Optional[str] = Field(None, description="Verification source")
    
    # Fact relationships
    related_entities: List[str] = Field(default_factory=list, description="Related entities")
    related_facts: List[str] = Field(default_factory=list, description="Related facts")
    
    # Temporal information
    fact_date: Optional[datetime] = Field(None, description="Date the fact occurred")
    expiry_date: Optional[datetime] = Field(None, description="Fact expiry date")
    
    @validator('fact_type')
    def validate_fact_type(cls, v):
        common_types = [
            'general', 'personal', 'professional', 'technical', 'historical',
            'preference', 'capability', 'limitation', 'goal', 'achievement'
        ]
        return v.lower()
    
    def get_memory_type(self) -> str:
        return MemoryType.FACT.value
    
    def is_expired(self) -> bool:
        """Check if fact is expired."""
        if not self.expiry_date:
            return False
        return datetime.now(timezone.utc) > self.expiry_date


class RelationshipMemory(BaseMemoryRecord):
    """Relationship memory record."""
    
    source_entity: str = Field(..., description="Source entity")
    target_entity: str = Field(..., description="Target entity")
    relationship_type: str = Field(..., description="Type of relationship")
    
    # Relationship properties
    properties: Dict[str, Any] = Field(default_factory=dict, description="Relationship properties")
    strength: float = Field(default=1.0, ge=0.0, le=1.0, description="Relationship strength")
    bidirectional: bool = Field(default=False, description="Is relationship bidirectional")
    
    # Temporal information
    start_date: Optional[datetime] = Field(None, description="Relationship start date")
    end_date: Optional[datetime] = Field(None, description="Relationship end date")
    
    def get_memory_type(self) -> str:
        return MemoryType.RELATIONSHIP.value
    
    def is_active(self) -> bool:
        """Check if relationship is currently active."""
        now = datetime.now(timezone.utc)
        
        if self.start_date and self.start_date > now:
            return False
        
        if self.end_date and self.end_date < now:
            return False
        
        return True


class MemoryQuery(BaseModel):
    """Memory query parameters."""
    
    user_id: str = Field(..., description="User identifier")
    query: str = Field(..., description="Search query")
    
    # Query filters
    memory_types: Optional[List[str]] = Field(None, description="Memory types to search")
    tags: Optional[List[str]] = Field(None, description="Tags to filter by")
    date_range: Optional[Dict[str, datetime]] = Field(None, description="Date range filter")
    
    # Search parameters
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results")
    threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")
    
    # Additional filters
    filters: Dict[str, Any] = Field(default_factory=dict, description="Additional filters")
    
    def to_search_params(self) -> Dict[str, Any]:
        """Convert to search parameters."""
        params = {
            "user_id": self.user_id,
            "query": self.query,
            "limit": self.limit,
            "threshold": self.threshold
        }
        
        if self.memory_types:
            params["memory_types"] = self.memory_types
        
        if self.tags:
            params["tags"] = self.tags
        
        if self.date_range:
            params["date_range"] = self.date_range
        
        if self.filters:
            params["filters"] = self.filters
        
        return params


class MemorySearchResult(BaseModel):
    """Memory search result."""
    
    memory_id: str = Field(..., description="Memory identifier")
    content: str = Field(..., description="Memory content")
    memory_type: str = Field(..., description="Type of memory")
    
    # Relevance scoring
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    relevance_score: Optional[float] = Field(None, description="Relevance score")
    
    # Memory metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Memory metadata")
    tags: List[str] = Field(default_factory=list, description="Memory tags")
    
    # Temporal information
    timestamp: Optional[str] = Field(None, description="Memory timestamp")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.dict(exclude_none=True)


class MemoryAnalytics(BaseModel):
    """Memory analytics and insights."""
    
    user_id: str = Field(..., description="User identifier")
    analysis_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Memory statistics
    total_memories: int = Field(default=0, description="Total number of memories")
    memory_type_counts: Dict[str, int] = Field(default_factory=dict, description="Count by memory type")
    
    # Activity patterns
    daily_activity: Dict[str, int] = Field(default_factory=dict, description="Daily memory activity")
    weekly_activity: Dict[str, int] = Field(default_factory=dict, description="Weekly memory activity")
    
    # Content analysis
    top_entities: List[Dict[str, Any]] = Field(default_factory=list, description="Most mentioned entities")
    top_topics: List[Dict[str, Any]] = Field(default_factory=list, description="Most discussed topics")
    
    # Preference insights
    preference_categories: Dict[str, int] = Field(default_factory=dict, description="Preference categories")
    confidence_distribution: Dict[str, int] = Field(default_factory=dict, description="Confidence score distribution")
    
    # Relationship insights
    entity_relationships: List[Dict[str, Any]] = Field(default_factory=list, description="Entity relationships")
    relationship_types: Dict[str, int] = Field(default_factory=dict, description="Relationship type counts")
    
    def add_memory_stats(self, memories: List[BaseMemoryRecord]):
        """Add statistics from memory list."""
        self.total_memories = len(memories)
        
        # Count by type
        for memory in memories:
            memory_type = memory.get_memory_type()
            self.memory_type_counts[memory_type] = self.memory_type_counts.get(memory_type, 0) + 1
        
        # Analyze activity patterns
        for memory in memories:
            date_str = memory.created_at.strftime("%Y-%m-%d")
            self.daily_activity[date_str] = self.daily_activity.get(date_str, 0) + 1
            
            week_str = memory.created_at.strftime("%Y-W%U")
            self.weekly_activity[week_str] = self.weekly_activity.get(week_str, 0) + 1


# Factory functions for creating memory records
def create_conversation_memory(
    user_id: str,
    conversation_id: str,
    message: str,
    role: str,
    **kwargs
) -> ConversationMemory:
    """Create conversation memory record."""
    return ConversationMemory(
        user_id=user_id,
        conversation_id=conversation_id,
        message=message,
        role=role,
        **kwargs
    )


def create_user_preference(
    user_id: str,
    preference_type: str,
    preference_value: Any,
    **kwargs
) -> UserPreference:
    """Create user preference record."""
    return UserPreference(
        user_id=user_id,
        preference_type=preference_type,
        preference_value=preference_value,
        **kwargs
    )


def create_entity_memory(
    user_id: str,
    entity_name: str,
    entity_type: str,
    properties: Optional[Dict[str, Any]] = None,
    **kwargs
) -> EntityMemory:
    """Create entity memory record."""
    return EntityMemory(
        user_id=user_id,
        entity_name=entity_name,
        entity_type=entity_type,
        properties=properties or {},
        **kwargs
    )


def create_fact_memory(
    user_id: str,
    fact: str,
    fact_type: str = "general",
    **kwargs
) -> FactMemory:
    """Create fact memory record."""
    return FactMemory(
        user_id=user_id,
        fact=fact,
        fact_type=fact_type,
        **kwargs
    )


def create_relationship_memory(
    user_id: str,
    source_entity: str,
    target_entity: str,
    relationship_type: str,
    **kwargs
) -> RelationshipMemory:
    """Create relationship memory record."""
    return RelationshipMemory(
        user_id=user_id,
        source_entity=source_entity,
        target_entity=target_entity,
        relationship_type=relationship_type,
        **kwargs
    )


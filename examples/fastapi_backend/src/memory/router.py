"""
Memory Management router with mem0 integration patterns.

This module provides endpoints for managing conversation memory, user preferences,
and context storage using mem0 memory management system.
"""

from datetime import datetime, timedelta
from typing import Annotated, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Query
from loguru import logger
from pydantic import BaseModel, Field

from src.auth.router import get_current_active_user
from src.common.exceptions import (
    MemoryException,
    MemoryNotFoundException,
    ContextException,
    ValidationException
)

router = APIRouter()

# Memory models
class MemoryItem(BaseModel):
    """Individual memory item model."""
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID = Field(..., description="User ID")
    content: str = Field(..., description="Memory content")
    memory_type: str = Field(..., description="Memory type (preference, fact, context)")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Memory importance")
    tags: List[str] = Field(default=[], description="Memory tags")
    metadata: Dict = Field(default={}, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = Field(default=0, description="Access frequency")
    
    class Config:
        from_attributes = True


class MemoryCreate(BaseModel):
    """Memory creation model."""
    content: str = Field(..., description="Memory content")
    memory_type: str = Field(..., description="Memory type")
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: List[str] = Field(default=[])
    metadata: Dict = Field(default={})


class MemoryUpdate(BaseModel):
    """Memory update model."""
    content: Optional[str] = None
    importance: Optional[float] = Field(None, ge=0.0, le=1.0)
    tags: Optional[List[str]] = None
    metadata: Optional[Dict] = None


class MemorySearch(BaseModel):
    """Memory search model."""
    query: str = Field(..., description="Search query")
    memory_types: Optional[List[str]] = Field(None, description="Filter by memory types")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    min_importance: float = Field(default=0.0, ge=0.0, le=1.0)
    limit: int = Field(default=10, ge=1, le=100)


class MemorySearchResult(BaseModel):
    """Memory search result model."""
    memories: List[MemoryItem]
    total_count: int
    search_metadata: Dict = Field(default={})


class ConversationContext(BaseModel):
    """Conversation context model."""
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    conversation_id: UUID
    context_data: Dict = Field(..., description="Context information")
    summary: str = Field(..., description="Context summary")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        from_attributes = True


class UserPreferences(BaseModel):
    """User preferences model."""
    user_id: UUID
    preferences: Dict = Field(..., description="User preferences")
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        from_attributes = True


class MemoryStats(BaseModel):
    """Memory statistics model."""
    total_memories: int
    memory_type_distribution: Dict[str, int]
    average_importance: float
    most_accessed_memories: List[MemoryItem]
    recent_memories: List[MemoryItem]


# Mock data storage (in production, use mem0 and database)
memories_db: Dict[UUID, MemoryItem] = {}
contexts_db: Dict[UUID, ConversationContext] = {}
preferences_db: Dict[UUID, UserPreferences] = {}

# Supported memory types
MEMORY_TYPES = ["preference", "fact", "context", "skill", "goal", "feedback"]


def validate_memory_type(memory_type: str) -> bool:
    """Validate memory type."""
    return memory_type in MEMORY_TYPES


def calculate_memory_relevance(memory: MemoryItem, query: str) -> float:
    """Calculate memory relevance to query (simplified)."""
    query_lower = query.lower()
    content_lower = memory.content.lower()
    
    # Simple relevance calculation
    relevance = 0.0
    
    # Exact match bonus
    if query_lower in content_lower:
        relevance += 0.5
    
    # Word overlap
    query_words = set(query_lower.split())
    content_words = set(content_lower.split())
    overlap = len(query_words.intersection(content_words))
    if len(query_words) > 0:
        relevance += (overlap / len(query_words)) * 0.3
    
    # Tag match bonus
    for tag in memory.tags:
        if tag.lower() in query_lower:
            relevance += 0.2
    
    # Importance and recency factors
    relevance *= memory.importance
    
    # Recent access bonus
    days_since_access = (datetime.utcnow() - memory.last_accessed).days
    if days_since_access < 7:
        relevance *= 1.1
    
    return min(relevance, 1.0)


def consolidate_memories(user_id: UUID) -> int:
    """Consolidate similar memories for a user."""
    user_memories = [m for m in memories_db.values() if m.user_id == user_id]
    consolidated_count = 0
    
    # Simple consolidation logic (in production, use more sophisticated algorithms)
    for i, memory1 in enumerate(user_memories):
        for memory2 in user_memories[i+1:]:
            similarity = calculate_memory_relevance(memory1, memory2.content)
            if similarity > 0.8:  # High similarity threshold
                # Merge memories (simplified)
                memory1.importance = max(memory1.importance, memory2.importance)
                memory1.access_count += memory2.access_count
                memory1.tags = list(set(memory1.tags + memory2.tags))
                
                # Remove the duplicate
                if memory2.id in memories_db:
                    del memories_db[memory2.id]
                    consolidated_count += 1
    
    return consolidated_count


@router.post("/memories", response_model=MemoryItem)
async def create_memory(
    memory_data: MemoryCreate,
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> MemoryItem:
    """
    Create a new memory item.
    
    - **content**: Memory content
    - **memory_type**: Type of memory (preference, fact, context, skill, goal, feedback)
    - **importance**: Memory importance (0.0 to 1.0)
    - **tags**: Memory tags for categorization
    - **metadata**: Additional metadata
    """
    if not validate_memory_type(memory_data.memory_type):
        raise ValidationException(f"Invalid memory type: {memory_data.memory_type}")
    
    memory = MemoryItem(
        user_id=UUID(current_user["id"]),
        content=memory_data.content,
        memory_type=memory_data.memory_type,
        importance=memory_data.importance,
        tags=memory_data.tags,
        metadata=memory_data.metadata
    )
    
    memories_db[memory.id] = memory
    
    logger.info(f"Created memory {memory.id} for user {current_user['email']}")
    
    return memory


@router.get("/memories", response_model=List[MemoryItem])
async def list_memories(
    current_user: Annotated[dict, Depends(get_current_active_user)],
    memory_type: Optional[str] = Query(None, description="Filter by memory type"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    min_importance: float = Query(0.0, ge=0.0, le=1.0, description="Minimum importance"),
    limit: int = Query(50, ge=1, le=200, description="Result limit")
) -> List[MemoryItem]:
    """
    List user memories with optional filtering.
    
    - **memory_type**: Filter by memory type
    - **tags**: Filter by tags (comma-separated)
    - **min_importance**: Minimum importance threshold
    - **limit**: Maximum number of results
    """
    user_memories = [
        m for m in memories_db.values()
        if m.user_id == UUID(current_user["id"])
    ]
    
    # Filter by type
    if memory_type:
        if not validate_memory_type(memory_type):
            raise ValidationException(f"Invalid memory type: {memory_type}")
        user_memories = [m for m in user_memories if m.memory_type == memory_type]
    
    # Filter by tags
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",")]
        user_memories = [
            m for m in user_memories
            if any(tag in m.tags for tag in tag_list)
        ]
    
    # Filter by importance
    user_memories = [m for m in user_memories if m.importance >= min_importance]
    
    # Sort by importance and recency
    user_memories.sort(
        key=lambda x: (x.importance, x.last_accessed),
        reverse=True
    )
    
    return user_memories[:limit]


@router.get("/memories/{memory_id}", response_model=MemoryItem)
async def get_memory(
    memory_id: UUID,
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> MemoryItem:
    """Get a specific memory by ID."""
    memory = memories_db.get(memory_id)
    if not memory:
        raise MemoryNotFoundException(str(memory_id))
    
    if memory.user_id != UUID(current_user["id"]):
        raise MemoryNotFoundException(str(memory_id))
    
    # Update access tracking
    memory.last_accessed = datetime.utcnow()
    memory.access_count += 1
    
    return memory


@router.put("/memories/{memory_id}", response_model=MemoryItem)
async def update_memory(
    memory_id: UUID,
    memory_data: MemoryUpdate,
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> MemoryItem:
    """Update an existing memory."""
    memory = memories_db.get(memory_id)
    if not memory:
        raise MemoryNotFoundException(str(memory_id))
    
    if memory.user_id != UUID(current_user["id"]):
        raise MemoryNotFoundException(str(memory_id))
    
    # Update fields
    if memory_data.content is not None:
        memory.content = memory_data.content
    if memory_data.importance is not None:
        memory.importance = memory_data.importance
    if memory_data.tags is not None:
        memory.tags = memory_data.tags
    if memory_data.metadata is not None:
        memory.metadata.update(memory_data.metadata)
    
    logger.info(f"Updated memory {memory_id}")
    
    return memory


@router.delete("/memories/{memory_id}")
async def delete_memory(
    memory_id: UUID,
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> dict:
    """Delete a memory."""
    memory = memories_db.get(memory_id)
    if not memory:
        raise MemoryNotFoundException(str(memory_id))
    
    if memory.user_id != UUID(current_user["id"]):
        raise MemoryNotFoundException(str(memory_id))
    
    del memories_db[memory_id]
    
    logger.info(f"Deleted memory {memory_id}")
    
    return {"message": f"Memory {memory_id} deleted successfully"}


@router.post("/memories/search", response_model=MemorySearchResult)
async def search_memories(
    search_data: MemorySearch,
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> MemorySearchResult:
    """
    Search memories using semantic similarity.
    
    - **query**: Search query
    - **memory_types**: Filter by memory types
    - **tags**: Filter by tags
    - **min_importance**: Minimum importance threshold
    - **limit**: Maximum results
    """
    user_memories = [
        m for m in memories_db.values()
        if m.user_id == UUID(current_user["id"])
    ]
    
    # Apply filters
    if search_data.memory_types:
        user_memories = [
            m for m in user_memories
            if m.memory_type in search_data.memory_types
        ]
    
    if search_data.tags:
        user_memories = [
            m for m in user_memories
            if any(tag in m.tags for tag in search_data.tags)
        ]
    
    user_memories = [
        m for m in user_memories
        if m.importance >= search_data.min_importance
    ]
    
    # Calculate relevance scores
    memory_scores = []
    for memory in user_memories:
        relevance = calculate_memory_relevance(memory, search_data.query)
        if relevance > 0.1:  # Minimum relevance threshold
            memory_scores.append((memory, relevance))
    
    # Sort by relevance
    memory_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top results
    top_memories = [memory for memory, score in memory_scores[:search_data.limit]]
    
    return MemorySearchResult(
        memories=top_memories,
        total_count=len(memory_scores),
        search_metadata={
            "query": search_data.query,
            "filters_applied": {
                "memory_types": search_data.memory_types,
                "tags": search_data.tags,
                "min_importance": search_data.min_importance
            }
        }
    )


@router.post("/consolidate")
async def consolidate_user_memories(
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> dict:
    """
    Consolidate similar memories to reduce redundancy.
    
    Merges similar memories and removes duplicates.
    """
    try:
        consolidated_count = consolidate_memories(UUID(current_user["id"]))
        
        logger.info(f"Consolidated {consolidated_count} memories for user {current_user['email']}")
        
        return {
            "message": f"Successfully consolidated {consolidated_count} memories",
            "consolidated_count": consolidated_count
        }
    
    except Exception as e:
        logger.error(f"Memory consolidation error: {e}")
        raise MemoryException(f"Consolidation failed: {str(e)}")


@router.get("/preferences", response_model=UserPreferences)
async def get_user_preferences(
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> UserPreferences:
    """Get user preferences."""
    user_id = UUID(current_user["id"])
    preferences = preferences_db.get(user_id)
    
    if not preferences:
        # Create default preferences
        preferences = UserPreferences(
            user_id=user_id,
            preferences={}
        )
        preferences_db[user_id] = preferences
    
    return preferences


@router.put("/preferences", response_model=UserPreferences)
async def update_user_preferences(
    preferences_data: Dict,
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> UserPreferences:
    """Update user preferences."""
    user_id = UUID(current_user["id"])
    preferences = preferences_db.get(user_id)
    
    if not preferences:
        preferences = UserPreferences(
            user_id=user_id,
            preferences=preferences_data
        )
    else:
        preferences.preferences.update(preferences_data)
        preferences.updated_at = datetime.utcnow()
    
    preferences_db[user_id] = preferences
    
    logger.info(f"Updated preferences for user {current_user['email']}")
    
    return preferences


@router.get("/stats", response_model=MemoryStats)
async def get_memory_stats(
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> MemoryStats:
    """Get memory statistics for the user."""
    user_memories = [
        m for m in memories_db.values()
        if m.user_id == UUID(current_user["id"])
    ]
    
    # Calculate statistics
    memory_type_dist = {}
    total_importance = 0
    
    for memory in user_memories:
        memory_type_dist[memory.memory_type] = memory_type_dist.get(memory.memory_type, 0) + 1
        total_importance += memory.importance
    
    avg_importance = total_importance / len(user_memories) if user_memories else 0
    
    # Most accessed memories
    most_accessed = sorted(user_memories, key=lambda x: x.access_count, reverse=True)[:5]
    
    # Recent memories
    recent = sorted(user_memories, key=lambda x: x.created_at, reverse=True)[:5]
    
    return MemoryStats(
        total_memories=len(user_memories),
        memory_type_distribution=memory_type_dist,
        average_importance=avg_importance,
        most_accessed_memories=most_accessed,
        recent_memories=recent
    )


@router.get("/types")
async def get_memory_types() -> Dict[str, List[str]]:
    """Get supported memory types."""
    return {
        "supported_types": MEMORY_TYPES,
        "descriptions": {
            "preference": "User preferences and settings",
            "fact": "Factual information about the user",
            "context": "Conversation context and history",
            "skill": "User skills and capabilities",
            "goal": "User goals and objectives",
            "feedback": "User feedback and evaluations"
        }
    }


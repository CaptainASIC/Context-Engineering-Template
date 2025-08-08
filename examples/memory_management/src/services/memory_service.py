"""
Memory Service using mem0 for conversation context and user preferences.

This module provides a comprehensive memory management service that handles
conversation context, user preferences, entity relationships, and fact storage
using the mem0 framework with multiple LLM support.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timezone, timedelta
from uuid import uuid4, UUID
import json
import hashlib
from contextlib import asynccontextmanager

from mem0 import Memory
from loguru import logger
import numpy as np

from config.settings import MemoryManagementSettings, get_settings, MemoryType
from src.models.memory_models import (
    MemoryRecord, ConversationMemory, UserPreference, 
    EntityMemory, FactMemory, MemoryQuery, MemorySearchResult
)


class MemoryService:
    """
    Comprehensive memory service using mem0 for intelligent memory management.
    
    Provides conversation context, user preferences, entity relationships,
    and fact storage with advanced retrieval and learning capabilities.
    """
    
    def __init__(self, settings: Optional[MemoryManagementSettings] = None):
        """Initialize memory service."""
        self.settings = settings or get_settings()
        
        # mem0 Memory instance
        self.memory = None
        
        # Memory collections by type
        self._collections = {}
        
        # Cache for frequently accessed memories
        self._memory_cache = {}
        self._cache_ttl = 300  # 5 minutes
        
        # Statistics tracking
        self._stats = {
            "memories_created": 0,
            "memories_retrieved": 0,
            "memories_updated": 0,
            "memories_deleted": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info("Initialized MemoryService")
    
    async def initialize(self) -> bool:
        """
        Initialize memory service and mem0 components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Configure mem0 with settings
            config = self._build_mem0_config()
            
            # Initialize mem0 Memory
            self.memory = Memory(config)
            
            # Initialize collections
            await self._initialize_collections()
            
            # Setup cleanup tasks
            asyncio.create_task(self._periodic_cleanup())
            
            logger.info("Memory service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize memory service: {e}")
            return False
    
    def _build_mem0_config(self) -> Dict[str, Any]:
        """Build mem0 configuration from settings."""
        config = {
            "llm": self.settings.get_llm_config(),
            "embedder": self.settings.get_embedding_config(),
            "vector_store": self.settings.get_vector_store_config(),
            "version": "v1.1"
        }
        
        # Add custom configurations
        if self.settings.storage.backend == "sqlite":
            config["history_db_path"] = self.settings.storage.connection_string.replace("sqlite:///", "")
        
        return config
    
    async def _initialize_collections(self):
        """Initialize memory collections for different types."""
        memory_types = self.settings.get_memory_types()
        
        for memory_type in memory_types:
            collection_name = f"{memory_type}_memories"
            self._collections[memory_type] = collection_name
            logger.debug(f"Initialized collection: {collection_name}")
    
    async def add_conversation_memory(
        self,
        user_id: str,
        conversation_id: str,
        message: str,
        role: str = "user",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add conversation memory.
        
        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            message: Message content
            role: Message role (user, assistant, system)
            metadata: Additional metadata
            
        Returns:
            Memory ID
        """
        try:
            # Prepare memory data
            memory_data = {
                "message": message,
                "role": role,
                "conversation_id": conversation_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **(metadata or {})
            }
            
            # Add to mem0
            result = self.memory.add(
                messages=[{"role": role, "content": message}],
                user_id=user_id,
                metadata=memory_data
            )
            
            # Update statistics
            self._stats["memories_created"] += 1
            
            # Create memory record
            memory_record = ConversationMemory(
                user_id=user_id,
                conversation_id=conversation_id,
                message=message,
                role=role,
                metadata=memory_data
            )
            
            # Cache the memory
            memory_id = str(result.get("id", uuid4()))
            self._cache_memory(memory_id, memory_record)
            
            logger.debug(f"Added conversation memory: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to add conversation memory: {e}")
            raise
    
    async def add_user_preference(
        self,
        user_id: str,
        preference_type: str,
        preference_value: Any,
        confidence: float = 1.0,
        source: str = "explicit",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add or update user preference.
        
        Args:
            user_id: User identifier
            preference_type: Type of preference
            preference_value: Preference value
            confidence: Confidence score (0.0 to 1.0)
            source: Source of preference (explicit, inferred, learned)
            metadata: Additional metadata
            
        Returns:
            Memory ID
        """
        try:
            # Prepare preference data
            preference_data = {
                "preference_type": preference_type,
                "preference_value": preference_value,
                "confidence": confidence,
                "source": source,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **(metadata or {})
            }
            
            # Create preference message for mem0
            preference_message = f"User prefers {preference_type}: {preference_value}"
            
            # Add to mem0
            result = self.memory.add(
                messages=[{"role": "system", "content": preference_message}],
                user_id=user_id,
                metadata=preference_data
            )
            
            # Update statistics
            self._stats["memories_created"] += 1
            
            # Create preference record
            preference_record = UserPreference(
                user_id=user_id,
                preference_type=preference_type,
                preference_value=preference_value,
                confidence=confidence,
                source=source,
                metadata=preference_data
            )
            
            # Cache the preference
            memory_id = str(result.get("id", uuid4()))
            self._cache_memory(memory_id, preference_record)
            
            logger.debug(f"Added user preference: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to add user preference: {e}")
            raise
    
    async def add_entity_memory(
        self,
        user_id: str,
        entity_name: str,
        entity_type: str,
        properties: Dict[str, Any],
        relationships: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add entity memory with properties and relationships.
        
        Args:
            user_id: User identifier
            entity_name: Entity name
            entity_type: Entity type
            properties: Entity properties
            relationships: Entity relationships
            metadata: Additional metadata
            
        Returns:
            Memory ID
        """
        try:
            # Prepare entity data
            entity_data = {
                "entity_name": entity_name,
                "entity_type": entity_type,
                "properties": properties,
                "relationships": relationships or [],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **(metadata or {})
            }
            
            # Create entity description for mem0
            entity_description = f"Entity {entity_name} is a {entity_type} with properties: {json.dumps(properties)}"
            
            # Add to mem0
            result = self.memory.add(
                messages=[{"role": "system", "content": entity_description}],
                user_id=user_id,
                metadata=entity_data
            )
            
            # Update statistics
            self._stats["memories_created"] += 1
            
            # Create entity record
            entity_record = EntityMemory(
                user_id=user_id,
                entity_name=entity_name,
                entity_type=entity_type,
                properties=properties,
                relationships=relationships or [],
                metadata=entity_data
            )
            
            # Cache the entity
            memory_id = str(result.get("id", uuid4()))
            self._cache_memory(memory_id, entity_record)
            
            logger.debug(f"Added entity memory: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to add entity memory: {e}")
            raise
    
    async def add_fact_memory(
        self,
        user_id: str,
        fact: str,
        fact_type: str = "general",
        confidence: float = 1.0,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add fact memory.
        
        Args:
            user_id: User identifier
            fact: Fact statement
            fact_type: Type of fact
            confidence: Confidence score
            source: Source of fact
            metadata: Additional metadata
            
        Returns:
            Memory ID
        """
        try:
            # Prepare fact data
            fact_data = {
                "fact": fact,
                "fact_type": fact_type,
                "confidence": confidence,
                "source": source,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **(metadata or {})
            }
            
            # Add to mem0
            result = self.memory.add(
                messages=[{"role": "system", "content": f"Fact: {fact}"}],
                user_id=user_id,
                metadata=fact_data
            )
            
            # Update statistics
            self._stats["memories_created"] += 1
            
            # Create fact record
            fact_record = FactMemory(
                user_id=user_id,
                fact=fact,
                fact_type=fact_type,
                confidence=confidence,
                source=source,
                metadata=fact_data
            )
            
            # Cache the fact
            memory_id = str(result.get("id", uuid4()))
            self._cache_memory(memory_id, fact_record)
            
            logger.debug(f"Added fact memory: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to add fact memory: {e}")
            raise
    
    async def search_memories(
        self,
        user_id: str,
        query: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 10,
        threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MemorySearchResult]:
        """
        Search memories using semantic similarity.
        
        Args:
            user_id: User identifier
            query: Search query
            memory_types: Types of memories to search
            limit: Maximum number of results
            threshold: Similarity threshold
            filters: Additional filters
            
        Returns:
            List of search results
        """
        try:
            # Search using mem0
            search_results = self.memory.search(
                query=query,
                user_id=user_id,
                limit=limit
            )
            
            # Update statistics
            self._stats["memories_retrieved"] += len(search_results)
            
            # Process and filter results
            processed_results = []
            
            for result in search_results:
                # Extract memory data
                memory_data = result.get("metadata", {})
                memory_content = result.get("memory", "")
                similarity_score = result.get("score", 0.0)
                
                # Apply threshold filter
                if similarity_score < threshold:
                    continue
                
                # Apply memory type filter
                if memory_types:
                    memory_type = self._infer_memory_type(memory_data)
                    if memory_type not in memory_types:
                        continue
                
                # Apply custom filters
                if filters and not self._apply_filters(memory_data, filters):
                    continue
                
                # Create search result
                search_result = MemorySearchResult(
                    memory_id=str(result.get("id", "")),
                    content=memory_content,
                    memory_type=self._infer_memory_type(memory_data),
                    similarity_score=similarity_score,
                    metadata=memory_data,
                    timestamp=memory_data.get("timestamp")
                )
                
                processed_results.append(search_result)
            
            # Sort by similarity score
            processed_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            logger.debug(f"Found {len(processed_results)} memories for query: {query}")
            return processed_results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            raise
    
    async def get_conversation_context(
        self,
        user_id: str,
        conversation_id: str,
        max_messages: int = 20,
        max_tokens: int = 4000
    ) -> List[Dict[str, Any]]:
        """
        Get conversation context for a specific conversation.
        
        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            max_messages: Maximum number of messages
            max_tokens: Maximum token count
            
        Returns:
            List of conversation messages
        """
        try:
            # Search for conversation memories
            query = f"conversation_id:{conversation_id}"
            
            memories = await self.search_memories(
                user_id=user_id,
                query=query,
                memory_types=[MemoryType.CONVERSATION],
                limit=max_messages * 2  # Get more to filter
            )
            
            # Filter and sort conversation messages
            conversation_messages = []
            total_tokens = 0
            
            for memory in memories:
                metadata = memory.metadata
                
                if metadata.get("conversation_id") != conversation_id:
                    continue
                
                # Estimate token count (rough approximation)
                message_tokens = len(memory.content.split()) * 1.3
                
                if total_tokens + message_tokens > max_tokens:
                    break
                
                message = {
                    "role": metadata.get("role", "user"),
                    "content": memory.content,
                    "timestamp": metadata.get("timestamp"),
                    "memory_id": memory.memory_id
                }
                
                conversation_messages.append(message)
                total_tokens += message_tokens
                
                if len(conversation_messages) >= max_messages:
                    break
            
            # Sort by timestamp
            conversation_messages.sort(
                key=lambda x: x.get("timestamp", ""),
                reverse=False
            )
            
            logger.debug(f"Retrieved {len(conversation_messages)} messages for conversation {conversation_id}")
            return conversation_messages
            
        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            raise
    
    async def get_user_preferences(
        self,
        user_id: str,
        preference_types: Optional[List[str]] = None,
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Get user preferences.
        
        Args:
            user_id: User identifier
            preference_types: Specific preference types to retrieve
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary of user preferences
        """
        try:
            # Search for preference memories
            query = "user preferences"
            
            memories = await self.search_memories(
                user_id=user_id,
                query=query,
                memory_types=[MemoryType.USER_PREFERENCE],
                limit=100
            )
            
            # Process preferences
            preferences = {}
            
            for memory in memories:
                metadata = memory.metadata
                
                preference_type = metadata.get("preference_type")
                preference_value = metadata.get("preference_value")
                confidence = metadata.get("confidence", 0.0)
                
                if not preference_type or confidence < min_confidence:
                    continue
                
                if preference_types and preference_type not in preference_types:
                    continue
                
                # Keep highest confidence preference for each type
                if preference_type not in preferences or confidence > preferences[preference_type]["confidence"]:
                    preferences[preference_type] = {
                        "value": preference_value,
                        "confidence": confidence,
                        "source": metadata.get("source", "unknown"),
                        "timestamp": metadata.get("timestamp"),
                        "memory_id": memory.memory_id
                    }
            
            logger.debug(f"Retrieved {len(preferences)} preferences for user {user_id}")
            return preferences
            
        except Exception as e:
            logger.error(f"Failed to get user preferences: {e}")
            raise
    
    async def update_memory(
        self,
        memory_id: str,
        user_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update existing memory.
        
        Args:
            memory_id: Memory identifier
            user_id: User identifier
            updates: Updates to apply
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update using mem0
            result = self.memory.update(
                memory_id=memory_id,
                data=updates
            )
            
            # Update statistics
            self._stats["memories_updated"] += 1
            
            # Clear from cache
            self._clear_cache(memory_id)
            
            logger.debug(f"Updated memory: {memory_id}")
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            return False
    
    async def delete_memory(
        self,
        memory_id: str,
        user_id: str
    ) -> bool:
        """
        Delete memory.
        
        Args:
            memory_id: Memory identifier
            user_id: User identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete using mem0
            result = self.memory.delete(memory_id=memory_id)
            
            # Update statistics
            self._stats["memories_deleted"] += 1
            
            # Clear from cache
            self._clear_cache(memory_id)
            
            logger.debug(f"Deleted memory: {memory_id}")
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    async def get_user_memory_summary(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Get summary of user's memories.
        
        Args:
            user_id: User identifier
            
        Returns:
            Memory summary statistics
        """
        try:
            # Get all memories for user
            all_memories = self.memory.get_all(user_id=user_id)
            
            # Analyze memories
            summary = {
                "total_memories": len(all_memories),
                "memory_types": {},
                "recent_activity": {},
                "top_entities": {},
                "preference_count": 0,
                "conversation_count": 0,
                "fact_count": 0
            }
            
            # Process each memory
            for memory in all_memories:
                metadata = memory.get("metadata", {})
                memory_type = self._infer_memory_type(metadata)
                
                # Count by type
                summary["memory_types"][memory_type] = summary["memory_types"].get(memory_type, 0) + 1
                
                # Count specific types
                if memory_type == MemoryType.USER_PREFERENCE:
                    summary["preference_count"] += 1
                elif memory_type == MemoryType.CONVERSATION:
                    summary["conversation_count"] += 1
                elif memory_type == MemoryType.FACT:
                    summary["fact_count"] += 1
                
                # Track recent activity
                timestamp = metadata.get("timestamp", "")
                if timestamp:
                    date = timestamp.split("T")[0]
                    summary["recent_activity"][date] = summary["recent_activity"].get(date, 0) + 1
                
                # Track entities
                entity_name = metadata.get("entity_name")
                if entity_name:
                    summary["top_entities"][entity_name] = summary["top_entities"].get(entity_name, 0) + 1
            
            # Sort recent activity and top entities
            summary["recent_activity"] = dict(sorted(
                summary["recent_activity"].items(),
                key=lambda x: x[0],
                reverse=True
            )[:7])  # Last 7 days
            
            summary["top_entities"] = dict(sorted(
                summary["top_entities"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])  # Top 10 entities
            
            logger.debug(f"Generated memory summary for user {user_id}")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get memory summary for user {user_id}: {e}")
            raise
    
    def _infer_memory_type(self, metadata: Dict[str, Any]) -> str:
        """Infer memory type from metadata."""
        if "preference_type" in metadata:
            return MemoryType.USER_PREFERENCE
        elif "conversation_id" in metadata:
            return MemoryType.CONVERSATION
        elif "entity_name" in metadata:
            return MemoryType.ENTITY
        elif "fact" in metadata:
            return MemoryType.FACT
        else:
            return "unknown"
    
    def _apply_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Apply custom filters to memory metadata."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        
        return True
    
    def _cache_memory(self, memory_id: str, memory_record: Any):
        """Cache memory record."""
        self._memory_cache[memory_id] = {
            "record": memory_record,
            "timestamp": datetime.now(timezone.utc),
            "ttl": self._cache_ttl
        }
    
    def _get_cached_memory(self, memory_id: str) -> Optional[Any]:
        """Get memory from cache."""
        if memory_id not in self._memory_cache:
            self._stats["cache_misses"] += 1
            return None
        
        cache_entry = self._memory_cache[memory_id]
        
        # Check TTL
        if (datetime.now(timezone.utc) - cache_entry["timestamp"]).seconds > cache_entry["ttl"]:
            del self._memory_cache[memory_id]
            self._stats["cache_misses"] += 1
            return None
        
        self._stats["cache_hits"] += 1
        return cache_entry["record"]
    
    def _clear_cache(self, memory_id: str):
        """Clear memory from cache."""
        if memory_id in self._memory_cache:
            del self._memory_cache[memory_id]
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of expired memories and cache."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean expired cache entries
                current_time = datetime.now(timezone.utc)
                expired_keys = []
                
                for key, entry in self._memory_cache.items():
                    if (current_time - entry["timestamp"]).seconds > entry["ttl"]:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self._memory_cache[key]
                
                logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")
                
                # TODO: Implement memory retention policy cleanup
                
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            **self._stats,
            "cache_size": len(self._memory_cache),
            "cache_hit_rate": (
                self._stats["cache_hits"] / 
                (self._stats["cache_hits"] + self._stats["cache_misses"])
                if (self._stats["cache_hits"] + self._stats["cache_misses"]) > 0
                else 0.0
            ),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        health = {
            "status": "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": {}
        }
        
        try:
            # Test mem0 connection
            if self.memory:
                # Try a simple operation
                test_result = self.memory.get_all(user_id="health_check_user", limit=1)
                health["status"] = "healthy"
                health["details"]["mem0"] = "ok"
            else:
                health["status"] = "unhealthy"
                health["details"]["mem0"] = "not initialized"
            
            # Add statistics
            health["details"]["statistics"] = await self.get_statistics()
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["details"]["error"] = str(e)
            logger.error(f"Health check failed: {e}")
        
        return health


# Global memory service instance
_memory_service = None


async def get_memory_service() -> MemoryService:
    """Get global memory service instance."""
    global _memory_service
    
    if _memory_service is None:
        _memory_service = MemoryService()
        await _memory_service.initialize()
    
    return _memory_service


async def close_memory_service():
    """Close global memory service."""
    global _memory_service
    
    if _memory_service:
        # Cleanup if needed
        _memory_service = None


"""
PydanticAI RAG Agent implementation with multi-LLM support.

This module implements a Retrieval-Augmented Generation agent using PydanticAI
with support for multiple LLM providers and advanced retrieval strategies.
"""

from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
import asyncio

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model, KnownModelName
from loguru import logger

from ..models.document import Document, DocumentChunk
from ..retrieval.vector_store import VectorStore
from ..llm_providers.factory import LLMProviderFactory
from config.settings import RAGSettings, get_settings


@dataclass
class RAGContext:
    """Context for RAG operations."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_history: List[Dict[str, str]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.metadata is None:
            self.metadata = {}


class RAGQuery(BaseModel):
    """RAG query model."""
    question: str = Field(..., description="User question")
    context: Optional[RAGContext] = Field(None, description="Query context")
    retrieval_params: Optional[Dict[str, Any]] = Field(None, description="Retrieval parameters")
    llm_params: Optional[Dict[str, Any]] = Field(None, description="LLM parameters")


class RAGResponse(BaseModel):
    """RAG response model."""
    answer: str = Field(..., description="Generated answer")
    sources: List[DocumentChunk] = Field(default=[], description="Source documents")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
    metadata: Dict[str, Any] = Field(default={}, description="Response metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PydanticRAGAgent:
    """
    PydanticAI-based RAG agent with multi-LLM support.
    
    This agent combines document retrieval with language generation
    to provide contextually relevant answers to user questions.
    """
    
    def __init__(self, settings: Optional[RAGSettings] = None):
        """Initialize the RAG agent."""
        self.settings = settings or get_settings()
        self.vector_store = VectorStore(self.settings.vector_db)
        self.llm_factory = LLMProviderFactory(self.settings)
        
        # Initialize PydanticAI agent
        self._init_agent()
        
        logger.info(f"Initialized RAG agent with provider: {self.settings.default_provider}")
    
    def _init_agent(self):
        """Initialize the PydanticAI agent."""
        # Get the active LLM model
        model = self.llm_factory.get_model(self.settings.default_provider)
        
        # Create the agent with system prompt
        self.agent = Agent(
            model=model,
            system_prompt=self.settings.system_prompt,
            deps_type=RAGContext
        )
        
        # Register tools
        self._register_tools()
    
    def _register_tools(self):
        """Register tools for the agent."""
        
        @self.agent.tool
        async def search_documents(
            ctx: RunContext[RAGContext], 
            query: str, 
            top_k: int = None
        ) -> List[DocumentChunk]:
            """
            Search for relevant documents in the knowledge base.
            
            Args:
                query: Search query
                top_k: Number of documents to retrieve
                
            Returns:
                List of relevant document chunks
            """
            top_k = top_k or self.settings.retrieval.top_k
            
            try:
                results = await self.vector_store.search(
                    query=query,
                    top_k=top_k,
                    threshold=self.settings.retrieval.similarity_threshold
                )
                
                logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")
                return results
                
            except Exception as e:
                logger.error(f"Error searching documents: {e}")
                return []
        
        @self.agent.tool
        async def get_conversation_context(
            ctx: RunContext[RAGContext]
        ) -> str:
            """
            Get conversation context from the current session.
            
            Returns:
                Formatted conversation history
            """
            if not ctx.deps or not ctx.deps.conversation_history:
                return "No previous conversation context."
            
            context_lines = []
            for msg in ctx.deps.conversation_history[-5:]:  # Last 5 messages
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                context_lines.append(f"{role.title()}: {content}")
            
            return "\n".join(context_lines)
        
        @self.agent.tool
        async def analyze_query_intent(
            ctx: RunContext[RAGContext], 
            query: str
        ) -> Dict[str, Any]:
            """
            Analyze the intent and complexity of the user query.
            
            Args:
                query: User query to analyze
                
            Returns:
                Query analysis results
            """
            analysis = {
                "query_type": "factual",  # factual, conversational, analytical
                "complexity": "medium",   # low, medium, high
                "requires_context": True,
                "keywords": [],
                "entities": []
            }
            
            # Simple keyword extraction
            words = query.lower().split()
            question_words = ["what", "how", "why", "when", "where", "who"]
            
            if any(word in words for word in question_words):
                analysis["query_type"] = "factual"
            elif any(word in words for word in ["compare", "analyze", "evaluate"]):
                analysis["query_type"] = "analytical"
                analysis["complexity"] = "high"
            else:
                analysis["query_type"] = "conversational"
                analysis["complexity"] = "low"
            
            # Extract potential keywords (simple approach)
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
            analysis["keywords"] = [word for word in words if len(word) > 3 and word not in stop_words]
            
            return analysis
    
    async def query(self, query: RAGQuery) -> RAGResponse:
        """
        Process a RAG query and generate a response.
        
        Args:
            query: RAG query object
            
        Returns:
            RAG response with answer and sources
        """
        try:
            # Prepare context
            context = query.context or RAGContext()
            
            # Run the agent
            result = await self.agent.run(
                query.question,
                deps=context
            )
            
            # Extract sources from tool calls (if any)
            sources = []
            if hasattr(result, 'all_messages'):
                for message in result.all_messages():
                    if hasattr(message, 'tool_calls'):
                        for tool_call in message.tool_calls:
                            if tool_call.tool_name == 'search_documents' and tool_call.result:
                                sources.extend(tool_call.result)
            
            # Calculate confidence (simplified)
            confidence = self._calculate_confidence(query.question, result.data, sources)
            
            # Prepare metadata
            metadata = {
                "model_used": self.settings.default_provider,
                "retrieval_count": len(sources),
                "query_length": len(query.question),
                "response_length": len(result.data),
                "processing_time": 0.0  # Would be calculated in practice
            }
            
            response = RAGResponse(
                answer=result.data,
                sources=sources,
                confidence=confidence,
                metadata=metadata
            )
            
            logger.info(f"Generated response with {len(sources)} sources")
            return response
            
        except Exception as e:
            logger.error(f"Error processing RAG query: {e}")
            return RAGResponse(
                answer=f"I apologize, but I encountered an error processing your question: {str(e)}",
                sources=[],
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    async def stream_query(self, query: RAGQuery) -> AsyncGenerator[str, None]:
        """
        Stream a RAG query response.
        
        Args:
            query: RAG query object
            
        Yields:
            Response chunks as they are generated
        """
        try:
            context = query.context or RAGContext()
            
            # Stream the response
            async with self.agent.run_stream(query.question, deps=context) as stream:
                async for chunk in stream:
                    yield chunk
                    
        except Exception as e:
            logger.error(f"Error streaming RAG query: {e}")
            yield f"Error: {str(e)}"
    
    def _calculate_confidence(self, question: str, answer: str, sources: List[DocumentChunk]) -> float:
        """
        Calculate confidence score for the response.
        
        Args:
            question: Original question
            answer: Generated answer
            sources: Retrieved sources
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.5  # Base confidence
        
        # Boost confidence if we have relevant sources
        if sources:
            confidence += min(len(sources) * 0.1, 0.3)
        
        # Boost confidence if answer is substantial
        if len(answer) > 100:
            confidence += 0.1
        
        # Reduce confidence if answer contains uncertainty phrases
        uncertainty_phrases = ["i'm not sure", "i don't know", "unclear", "uncertain"]
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
    
    async def add_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of documents to add
            
        Returns:
            Addition results
        """
        try:
            result = await self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to knowledge base")
            return result
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return {"error": str(e), "added_count": 0}
    
    async def delete_documents(self, document_ids: List[str]) -> Dict[str, Any]:
        """
        Delete documents from the knowledge base.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            Deletion results
        """
        try:
            result = await self.vector_store.delete_documents(document_ids)
            logger.info(f"Deleted {len(document_ids)} documents from knowledge base")
            return result
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return {"error": str(e), "deleted_count": 0}
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics.
        
        Returns:
            Agent statistics
        """
        try:
            vector_stats = await self.vector_store.get_stats()
            
            return {
                "agent_name": self.settings.agent_name,
                "provider": self.settings.default_provider,
                "vector_store": vector_stats,
                "settings": {
                    "top_k": self.settings.retrieval.top_k,
                    "similarity_threshold": self.settings.retrieval.similarity_threshold,
                    "chunk_size": self.settings.retrieval.chunk_size
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}
    
    async def switch_provider(self, provider_name: str) -> bool:
        """
        Switch to a different LLM provider.
        
        Args:
            provider_name: Name of the provider to switch to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if provider is available
            providers = self.settings.get_llm_providers()
            if provider_name not in providers:
                logger.error(f"Provider {provider_name} not configured")
                return False
            
            # Get new model
            model = self.llm_factory.get_model(provider_name)
            
            # Update agent
            self.agent = Agent(
                model=model,
                system_prompt=self.settings.system_prompt,
                deps_type=RAGContext
            )
            
            # Re-register tools
            self._register_tools()
            
            # Update settings
            self.settings.default_provider = provider_name
            
            logger.info(f"Switched to provider: {provider_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error switching provider: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the agent.
        
        Returns:
            Health check results
        """
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        try:
            # Check LLM provider
            try:
                test_query = RAGQuery(question="Hello, are you working?")
                test_response = await self.query(test_query)
                health["components"]["llm"] = {
                    "status": "healthy" if test_response.answer else "unhealthy",
                    "provider": self.settings.default_provider
                }
            except Exception as e:
                health["components"]["llm"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # Check vector store
            try:
                vector_stats = await self.vector_store.get_stats()
                health["components"]["vector_store"] = {
                    "status": "healthy",
                    "document_count": vector_stats.get("document_count", 0)
                }
            except Exception as e:
                health["components"]["vector_store"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # Overall status
            component_statuses = [comp["status"] for comp in health["components"].values()]
            if all(status == "healthy" for status in component_statuses):
                health["status"] = "healthy"
            elif any(status == "healthy" for status in component_statuses):
                health["status"] = "degraded"
            else:
                health["status"] = "unhealthy"
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
        
        return health


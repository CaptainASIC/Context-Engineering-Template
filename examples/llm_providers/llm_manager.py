"""
LIXIL AI Hub Multi-LLM Provider Integration Manager

This module provides comprehensive integration with multiple Large Language Model
providers, including intelligent routing, fallback mechanisms, cost optimization,
and performance monitoring for the LIXIL AI Hub Platform.

Key Features:
- Multi-provider support (OpenAI, Anthropic, Google Gemini, OpenRouter, Ollama)
- Intelligent model selection based on task type and language
- Automatic failover and retry mechanisms
- Cost tracking and optimization
- Response caching and rate limiting
- Performance monitoring and analytics

Author: LIXIL AI Hub Platform Team
Version: 1.0.0
"""

import asyncio
import hashlib
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from abc import ABC, abstractmethod

import aiohttp
import openai
import anthropic
import google.generativeai as genai
from pydantic import BaseModel, Field, validator
import tiktoken
import asyncpg

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"


class TaskType(str, Enum):
    """Types of tasks for intelligent model selection."""
    GENERAL_CHAT = "general_chat"
    POLICY_QA = "policy_qa"
    DOCUMENT_ANALYSIS = "document_analysis"
    MULTILINGUAL = "multilingual"
    CODE_GENERATION = "code_generation"
    CREATIVE_WRITING = "creative_writing"
    TECHNICAL_WRITING = "technical_writing"
    SUMMARIZATION = "summarization"


class ModelCapability(str, Enum):
    """Model capabilities for selection criteria."""
    TEXT_GENERATION = "text_generation"
    MULTILINGUAL = "multilingual"
    LONG_CONTEXT = "long_context"
    FAST_RESPONSE = "fast_response"
    HIGH_QUALITY = "high_quality"
    COST_EFFECTIVE = "cost_effective"
    CODE_UNDERSTANDING = "code_understanding"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    provider: LLMProvider
    model_name: str
    max_tokens: int
    context_window: int
    cost_per_1k_tokens: float
    capabilities: List[ModelCapability]
    supported_languages: List[str]
    is_available: bool = True
    priority: int = 1  # Lower number = higher priority


@dataclass
class LLMRequest:
    """Request for LLM processing."""
    prompt: str
    task_type: TaskType
    language: str = "en"
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    preferred_provider: Optional[LLMProvider] = None
    required_capabilities: List[ModelCapability] = field(default_factory=list)
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class LLMResponse:
    """Response from LLM processing."""
    content: str
    provider: LLMProvider
    model: str
    tokens_used: int
    cost: float
    response_time: float
    cached: bool = False
    error: Optional[str] = None


class LLMUsageStats(BaseModel):
    """Usage statistics for monitoring."""
    provider: LLMProvider
    model: str
    total_requests: int
    total_tokens: int
    total_cost: float
    average_response_time: float
    success_rate: float
    last_used: datetime


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Defines the interface that all LLM providers must implement
    for consistent integration across different services.
    """

    def __init__(self, api_key: str, config: Dict[str, Any]):
        """
        Initialize the LLM provider.
        
        Args:
            api_key: API key for the provider
            config: Provider-specific configuration
        """
        self.api_key = api_key
        self.config = config
        self.rate_limiter = {}
        self.last_request_time = {}

    @abstractmethod
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response from the LLM.
        
        Args:
            request: LLM request
            
        Returns:
            LLM response
        """
        pass

    @abstractmethod
    def get_available_models(self) -> List[ModelConfig]:
        """Get list of available models for this provider."""
        pass

    @abstractmethod
    def estimate_cost(self, prompt: str, max_tokens: int, model: str) -> float:
        """Estimate cost for a request."""
        pass

    async def _check_rate_limit(self, model: str) -> bool:
        """Check if request is within rate limits."""
        now = time.time()
        if model not in self.last_request_time:
            self.last_request_time[model] = now
            return True
        
        # Simple rate limiting - can be enhanced
        time_since_last = now - self.last_request_time[model]
        if time_since_last < 1.0:  # 1 second minimum between requests
            return False
        
        self.last_request_time[model] = now
        return True


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation."""

    def __init__(self, api_key: str, config: Dict[str, Any]):
        """Initialize OpenAI provider."""
        super().__init__(api_key, config)
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.encoding = tiktoken.get_encoding("cl100k_base")

    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using OpenAI API."""
        start_time = time.time()
        
        try:
            # Select appropriate model
            model = self._select_model(request)
            
            # Check rate limits
            if not await self._check_rate_limit(model):
                await asyncio.sleep(1)
            
            # Prepare messages
            messages = [{"role": "user", "content": request.prompt}]
            
            # Make API call
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=request.max_tokens or 1000,
                temperature=request.temperature
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            cost = self.estimate_cost(request.prompt, tokens_used, model)
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=content,
                provider=LLMProvider.OPENAI,
                model=model,
                tokens_used=tokens_used,
                cost=cost,
                response_time=response_time
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return LLMResponse(
                content="",
                provider=LLMProvider.OPENAI,
                model="",
                tokens_used=0,
                cost=0.0,
                response_time=time.time() - start_time,
                error=str(e)
            )

    def _select_model(self, request: LLMRequest) -> str:
        """Select appropriate OpenAI model based on request."""
        if request.task_type == TaskType.CODE_GENERATION:
            return "gpt-4"
        elif request.task_type in [TaskType.GENERAL_CHAT, TaskType.POLICY_QA]:
            return "gpt-3.5-turbo"
        elif ModelCapability.LONG_CONTEXT in request.required_capabilities:
            return "gpt-4-turbo"
        else:
            return "gpt-3.5-turbo"

    def get_available_models(self) -> List[ModelConfig]:
        """Get available OpenAI models."""
        return [
            ModelConfig(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4",
                max_tokens=4096,
                context_window=8192,
                cost_per_1k_tokens=0.03,
                capabilities=[ModelCapability.HIGH_QUALITY, ModelCapability.CODE_UNDERSTANDING],
                supported_languages=["en", "es", "fr", "de", "ja", "zh", "ko"]
            ),
            ModelConfig(
                provider=LLMProvider.OPENAI,
                model_name="gpt-3.5-turbo",
                max_tokens=4096,
                context_window=4096,
                cost_per_1k_tokens=0.002,
                capabilities=[ModelCapability.FAST_RESPONSE, ModelCapability.COST_EFFECTIVE],
                supported_languages=["en", "es", "fr", "de", "ja", "zh", "ko"]
            ),
            ModelConfig(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4-turbo",
                max_tokens=4096,
                context_window=128000,
                cost_per_1k_tokens=0.01,
                capabilities=[ModelCapability.LONG_CONTEXT, ModelCapability.HIGH_QUALITY],
                supported_languages=["en", "es", "fr", "de", "ja", "zh", "ko"]
            )
        ]

    def estimate_cost(self, prompt: str, tokens_used: int, model: str) -> float:
        """Estimate cost for OpenAI request."""
        model_costs = {
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.002,
            "gpt-4-turbo": 0.01
        }
        cost_per_1k = model_costs.get(model, 0.002)
        return (tokens_used / 1000) * cost_per_1k


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider implementation."""

    def __init__(self, api_key: str, config: Dict[str, Any]):
        """Initialize Anthropic provider."""
        super().__init__(api_key, config)
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Anthropic API."""
        start_time = time.time()
        
        try:
            model = self._select_model(request)
            
            if not await self._check_rate_limit(model):
                await asyncio.sleep(1)
            
            response = await self.client.messages.create(
                model=model,
                max_tokens=request.max_tokens or 1000,
                temperature=request.temperature,
                messages=[{"role": "user", "content": request.prompt}]
            )
            
            content = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            cost = self.estimate_cost(request.prompt, tokens_used, model)
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=content,
                provider=LLMProvider.ANTHROPIC,
                model=model,
                tokens_used=tokens_used,
                cost=cost,
                response_time=response_time
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return LLMResponse(
                content="",
                provider=LLMProvider.ANTHROPIC,
                model="",
                tokens_used=0,
                cost=0.0,
                response_time=time.time() - start_time,
                error=str(e)
            )

    def _select_model(self, request: LLMRequest) -> str:
        """Select appropriate Anthropic model."""
        if ModelCapability.LONG_CONTEXT in request.required_capabilities:
            return "claude-3-opus-20240229"
        elif request.task_type in [TaskType.TECHNICAL_WRITING, TaskType.POLICY_QA]:
            return "claude-3-sonnet-20240229"
        else:
            return "claude-3-haiku-20240307"

    def get_available_models(self) -> List[ModelConfig]:
        """Get available Anthropic models."""
        return [
            ModelConfig(
                provider=LLMProvider.ANTHROPIC,
                model_name="claude-3-opus-20240229",
                max_tokens=4096,
                context_window=200000,
                cost_per_1k_tokens=0.015,
                capabilities=[ModelCapability.HIGH_QUALITY, ModelCapability.LONG_CONTEXT],
                supported_languages=["en", "es", "fr", "de", "ja", "zh"]
            ),
            ModelConfig(
                provider=LLMProvider.ANTHROPIC,
                model_name="claude-3-sonnet-20240229",
                max_tokens=4096,
                context_window=200000,
                cost_per_1k_tokens=0.003,
                capabilities=[ModelCapability.HIGH_QUALITY, ModelCapability.LONG_CONTEXT],
                supported_languages=["en", "es", "fr", "de", "ja", "zh"]
            ),
            ModelConfig(
                provider=LLMProvider.ANTHROPIC,
                model_name="claude-3-haiku-20240307",
                max_tokens=4096,
                context_window=200000,
                cost_per_1k_tokens=0.00025,
                capabilities=[ModelCapability.FAST_RESPONSE, ModelCapability.COST_EFFECTIVE],
                supported_languages=["en", "es", "fr", "de", "ja", "zh"]
            )
        ]

    def estimate_cost(self, prompt: str, tokens_used: int, model: str) -> float:
        """Estimate cost for Anthropic request."""
        model_costs = {
            "claude-3-opus-20240229": 0.015,
            "claude-3-sonnet-20240229": 0.003,
            "claude-3-haiku-20240307": 0.00025
        }
        cost_per_1k = model_costs.get(model, 0.003)
        return (tokens_used / 1000) * cost_per_1k


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider implementation."""

    def __init__(self, api_key: str, config: Dict[str, Any]):
        """Initialize Gemini provider."""
        super().__init__(api_key, config)
        genai.configure(api_key=api_key)

    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Gemini API."""
        start_time = time.time()
        
        try:
            model_name = self._select_model(request)
            model = genai.GenerativeModel(model_name)
            
            if not await self._check_rate_limit(model_name):
                await asyncio.sleep(1)
            
            # Configure generation
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=request.max_tokens or 1000,
                temperature=request.temperature
            )
            
            response = await asyncio.to_thread(
                model.generate_content,
                request.prompt,
                generation_config=generation_config
            )
            
            content = response.text
            # Gemini doesn't provide token usage in response, estimate it
            tokens_used = len(content.split()) * 1.3  # Rough estimation
            cost = self.estimate_cost(request.prompt, int(tokens_used), model_name)
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=content,
                provider=LLMProvider.GEMINI,
                model=model_name,
                tokens_used=int(tokens_used),
                cost=cost,
                response_time=response_time
            )
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return LLMResponse(
                content="",
                provider=LLMProvider.GEMINI,
                model="",
                tokens_used=0,
                cost=0.0,
                response_time=time.time() - start_time,
                error=str(e)
            )

    def _select_model(self, request: LLMRequest) -> str:
        """Select appropriate Gemini model."""
        if request.language in ["ja", "zh", "ko"]:
            return "gemini-pro"
        elif request.task_type == TaskType.MULTILINGUAL:
            return "gemini-pro"
        else:
            return "gemini-pro"

    def get_available_models(self) -> List[ModelConfig]:
        """Get available Gemini models."""
        return [
            ModelConfig(
                provider=LLMProvider.GEMINI,
                model_name="gemini-pro",
                max_tokens=2048,
                context_window=30720,
                cost_per_1k_tokens=0.0005,
                capabilities=[ModelCapability.MULTILINGUAL, ModelCapability.COST_EFFECTIVE],
                supported_languages=["en", "es", "fr", "de", "ja", "zh", "ko", "pt", "it", "nl"]
            )
        ]

    def estimate_cost(self, prompt: str, tokens_used: int, model: str) -> float:
        """Estimate cost for Gemini request."""
        return (tokens_used / 1000) * 0.0005


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider implementation."""

    def __init__(self, api_key: str, config: Dict[str, Any]):
        """Initialize Ollama provider."""
        super().__init__(api_key, config)
        self.base_url = config.get("base_url", "http://localhost:11434")

    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Ollama API."""
        start_time = time.time()
        
        try:
            model = self._select_model(request)
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model,
                    "prompt": request.prompt,
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens or 1000
                    }
                }
                
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result.get("response", "")
                        tokens_used = len(content.split()) * 1.3  # Estimation
                        cost = 0.0  # Local models are free
                        response_time = time.time() - start_time
                        
                        return LLMResponse(
                            content=content,
                            provider=LLMProvider.OLLAMA,
                            model=model,
                            tokens_used=int(tokens_used),
                            cost=cost,
                            response_time=response_time
                        )
                    else:
                        raise Exception(f"Ollama API error: {response.status}")
                        
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return LLMResponse(
                content="",
                provider=LLMProvider.OLLAMA,
                model="",
                tokens_used=0,
                cost=0.0,
                response_time=time.time() - start_time,
                error=str(e)
            )

    def _select_model(self, request: LLMRequest) -> str:
        """Select appropriate Ollama model."""
        if request.task_type == TaskType.CODE_GENERATION:
            return "codellama"
        elif request.task_type in [TaskType.GENERAL_CHAT, TaskType.POLICY_QA]:
            return "llama2"
        else:
            return "llama2"

    def get_available_models(self) -> List[ModelConfig]:
        """Get available Ollama models."""
        return [
            ModelConfig(
                provider=LLMProvider.OLLAMA,
                model_name="llama2",
                max_tokens=2048,
                context_window=4096,
                cost_per_1k_tokens=0.0,
                capabilities=[ModelCapability.COST_EFFECTIVE, ModelCapability.FAST_RESPONSE],
                supported_languages=["en"]
            ),
            ModelConfig(
                provider=LLMProvider.OLLAMA,
                model_name="codellama",
                max_tokens=2048,
                context_window=4096,
                cost_per_1k_tokens=0.0,
                capabilities=[ModelCapability.CODE_UNDERSTANDING, ModelCapability.COST_EFFECTIVE],
                supported_languages=["en"]
            )
        ]

    def estimate_cost(self, prompt: str, tokens_used: int, model: str) -> float:
        """Estimate cost for Ollama request (always free)."""
        return 0.0


class ResponseCache:
    """
    Response caching system for LLM requests.
    
    Provides intelligent caching of LLM responses to reduce costs
    and improve response times for repeated queries.
    """

    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        """
        Initialize response cache.
        
        Args:
            max_size: Maximum number of cached responses
            ttl_hours: Time-to-live for cached responses in hours
        """
        self.cache: Dict[str, Tuple[LLMResponse, datetime]] = {}
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)

    def _generate_cache_key(self, request: LLMRequest) -> str:
        """Generate cache key for request."""
        key_data = f"{request.prompt}_{request.task_type}_{request.language}_{request.temperature}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Get cached response if available and valid."""
        cache_key = self._generate_cache_key(request)
        
        if cache_key in self.cache:
            response, timestamp = self.cache[cache_key]
            
            # Check if cache entry is still valid
            if datetime.now() - timestamp < self.ttl:
                response.cached = True
                return response
            else:
                # Remove expired entry
                del self.cache[cache_key]
        
        return None

    def set(self, request: LLMRequest, response: LLMResponse):
        """Cache response for request."""
        cache_key = self._generate_cache_key(request)
        
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[cache_key] = (response, datetime.now())

    def clear(self):
        """Clear all cached responses."""
        self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = datetime.now()
        valid_entries = sum(1 for _, timestamp in self.cache.values() 
                          if now - timestamp < self.ttl)
        
        return {
            "total_entries": len(self.cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self.cache) - valid_entries,
            "cache_size": len(self.cache),
            "max_size": self.max_size
        }


class LLMManager:
    """
    Main LLM manager for the LIXIL AI Hub Platform.
    
    Orchestrates multiple LLM providers with intelligent routing,
    fallback mechanisms, cost optimization, and performance monitoring.
    """

    def __init__(self, database_url: str, provider_configs: Dict[str, Dict[str, Any]]):
        """
        Initialize LLM manager.
        
        Args:
            database_url: PostgreSQL database connection URL
            provider_configs: Configuration for each provider
        """
        self.database_url = database_url
        self.db_pool = None
        self.providers: Dict[LLMProvider, BaseLLMProvider] = {}
        self.cache = ResponseCache()
        self.usage_stats: Dict[str, LLMUsageStats] = {}
        
        # Initialize providers
        self._initialize_providers(provider_configs)

    def _initialize_providers(self, configs: Dict[str, Dict[str, Any]]):
        """Initialize LLM providers based on configuration."""
        for provider_name, config in configs.items():
            try:
                provider_enum = LLMProvider(provider_name)
                api_key = config.get("api_key")
                
                if not api_key:
                    logger.warning(f"No API key provided for {provider_name}")
                    continue
                
                if provider_enum == LLMProvider.OPENAI:
                    self.providers[provider_enum] = OpenAIProvider(api_key, config)
                elif provider_enum == LLMProvider.ANTHROPIC:
                    self.providers[provider_enum] = AnthropicProvider(api_key, config)
                elif provider_enum == LLMProvider.GEMINI:
                    self.providers[provider_enum] = GeminiProvider(api_key, config)
                elif provider_enum == LLMProvider.OLLAMA:
                    self.providers[provider_enum] = OllamaProvider(api_key, config)
                
                logger.info(f"Initialized {provider_name} provider")
                
            except Exception as e:
                logger.error(f"Failed to initialize {provider_name} provider: {e}")

    async def initialize(self):
        """Initialize database connection and schema."""
        self.db_pool = await asyncpg.create_pool(
            self.database_url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        
        await self._create_llm_schema()
        logger.info("LLM manager initialized")

    async def close(self):
        """Close database connections."""
        if self.db_pool:
            await self.db_pool.close()

    async def _create_llm_schema(self):
        """Create LLM usage tracking schema."""
        schema_sql = """
        -- LLM usage tracking table
        CREATE TABLE IF NOT EXISTS llm_usage (
            usage_id SERIAL PRIMARY KEY,
            user_id VARCHAR(64),
            session_id VARCHAR(64),
            provider VARCHAR(50) NOT NULL,
            model VARCHAR(100) NOT NULL,
            task_type VARCHAR(50) NOT NULL,
            language VARCHAR(10) NOT NULL,
            prompt_tokens INTEGER NOT NULL,
            completion_tokens INTEGER NOT NULL,
            total_tokens INTEGER NOT NULL,
            cost DECIMAL(10, 6) NOT NULL,
            response_time DECIMAL(8, 3) NOT NULL,
            cached BOOLEAN DEFAULT FALSE,
            success BOOLEAN NOT NULL,
            error_message TEXT,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL
        );

        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_llm_usage_user_id ON llm_usage(user_id);
        CREATE INDEX IF NOT EXISTS idx_llm_usage_provider ON llm_usage(provider);
        CREATE INDEX IF NOT EXISTS idx_llm_usage_timestamp ON llm_usage(timestamp);
        CREATE INDEX IF NOT EXISTS idx_llm_usage_task_type ON llm_usage(task_type);
        """
        
        async with self.db_pool.acquire() as conn:
            await conn.execute(schema_sql)

    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response using the best available provider.
        
        Args:
            request: LLM request
            
        Returns:
            LLM response
        """
        # Check cache first
        cached_response = self.cache.get(request)
        if cached_response:
            await self._log_usage(request, cached_response)
            return cached_response
        
        # Select best provider
        selected_provider = self._select_provider(request)
        
        if not selected_provider:
            return LLMResponse(
                content="",
                provider=LLMProvider.OPENAI,
                model="",
                tokens_used=0,
                cost=0.0,
                response_time=0.0,
                error="No available providers"
            )
        
        # Try primary provider
        response = await self._try_provider(selected_provider, request)
        
        # Try fallback providers if primary fails
        if response.error:
            for fallback_provider in self._get_fallback_providers(selected_provider, request):
                response = await self._try_provider(fallback_provider, request)
                if not response.error:
                    break
        
        # Cache successful responses
        if not response.error:
            self.cache.set(request, response)
        
        # Log usage
        await self._log_usage(request, response)
        
        return response

    def _select_provider(self, request: LLMRequest) -> Optional[LLMProvider]:
        """Select the best provider for the request."""
        # Use preferred provider if specified and available
        if request.preferred_provider and request.preferred_provider in self.providers:
            return request.preferred_provider
        
        # Intelligent selection based on task type and language
        if request.language in ["ja", "zh", "ko"] and LLMProvider.GEMINI in self.providers:
            return LLMProvider.GEMINI
        
        if request.task_type == TaskType.CODE_GENERATION and LLMProvider.OPENAI in self.providers:
            return LLMProvider.OPENAI
        
        if request.task_type in [TaskType.TECHNICAL_WRITING, TaskType.POLICY_QA] and LLMProvider.ANTHROPIC in self.providers:
            return LLMProvider.ANTHROPIC
        
        # Default to first available provider
        return next(iter(self.providers.keys()), None)

    def _get_fallback_providers(self, primary: LLMProvider, request: LLMRequest) -> List[LLMProvider]:
        """Get fallback providers in order of preference."""
        all_providers = list(self.providers.keys())
        fallbacks = [p for p in all_providers if p != primary]
        
        # Prioritize based on task type
        if request.task_type == TaskType.MULTILINGUAL:
            fallbacks.sort(key=lambda p: 0 if p == LLMProvider.GEMINI else 1)
        elif request.task_type == TaskType.CODE_GENERATION:
            fallbacks.sort(key=lambda p: 0 if p == LLMProvider.OPENAI else 1)
        
        return fallbacks

    async def _try_provider(self, provider: LLMProvider, request: LLMRequest) -> LLMResponse:
        """Try to get response from a specific provider."""
        if provider not in self.providers:
            return LLMResponse(
                content="",
                provider=provider,
                model="",
                tokens_used=0,
                cost=0.0,
                response_time=0.0,
                error=f"Provider {provider} not available"
            )
        
        try:
            return await self.providers[provider].generate_response(request)
        except Exception as e:
            logger.error(f"Provider {provider} failed: {e}")
            return LLMResponse(
                content="",
                provider=provider,
                model="",
                tokens_used=0,
                cost=0.0,
                response_time=0.0,
                error=str(e)
            )

    async def _log_usage(self, request: LLMRequest, response: LLMResponse):
        """Log LLM usage for monitoring and billing."""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO llm_usage (
                    user_id, session_id, provider, model, task_type, language,
                    prompt_tokens, completion_tokens, total_tokens, cost,
                    response_time, cached, success, error_message, timestamp
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
            """,
                request.user_id, request.session_id, response.provider.value,
                response.model, request.task_type.value, request.language,
                0, 0, response.tokens_used, response.cost, response.response_time,
                response.cached, response.error is None, response.error, datetime.now()
            )

    async def get_usage_stats(self, user_id: Optional[str] = None, 
                            days: int = 30) -> Dict[str, Any]:
        """Get usage statistics."""
        since_date = datetime.now() - timedelta(days=days)
        
        conditions = ["timestamp >= $1"]
        params = [since_date]
        
        if user_id:
            conditions.append("user_id = $2")
            params.append(user_id)
        
        where_clause = " AND ".join(conditions)
        
        async with self.db_pool.acquire() as conn:
            stats = await conn.fetchrow(f"""
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(total_tokens) as total_tokens,
                    SUM(cost) as total_cost,
                    AVG(response_time) as avg_response_time,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as success_rate,
                    SUM(CASE WHEN cached THEN 1 ELSE 0 END) as cached_requests
                FROM llm_usage 
                WHERE {where_clause}
            """, *params)
            
            provider_stats = await conn.fetch(f"""
                SELECT 
                    provider,
                    COUNT(*) as requests,
                    SUM(total_tokens) as tokens,
                    SUM(cost) as cost,
                    AVG(response_time) as avg_time
                FROM llm_usage 
                WHERE {where_clause}
                GROUP BY provider
                ORDER BY requests DESC
            """, *params)
        
        return {
            "total_requests": stats["total_requests"],
            "total_tokens": stats["total_tokens"],
            "total_cost": float(stats["total_cost"]) if stats["total_cost"] else 0.0,
            "average_response_time": float(stats["avg_response_time"]) if stats["avg_response_time"] else 0.0,
            "success_rate": float(stats["success_rate"]) if stats["success_rate"] else 0.0,
            "cached_requests": stats["cached_requests"],
            "cache_hit_rate": stats["cached_requests"] / max(stats["total_requests"], 1),
            "provider_breakdown": [
                {
                    "provider": row["provider"],
                    "requests": row["requests"],
                    "tokens": row["tokens"],
                    "cost": float(row["cost"]),
                    "avg_response_time": float(row["avg_time"])
                }
                for row in provider_stats
            ]
        }

    def get_available_models(self) -> Dict[LLMProvider, List[ModelConfig]]:
        """Get all available models from all providers."""
        models = {}
        for provider_enum, provider in self.providers.items():
            models[provider_enum] = provider.get_available_models()
        return models


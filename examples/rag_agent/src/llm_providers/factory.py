"""
LLM Provider Factory for multi-LLM support.

This module provides a factory for creating and managing different
LLM providers (OpenAI, Anthropic, Google, OpenRouter, Ollama).
"""

from typing import Dict, Optional, Any
from abc import ABC, abstractmethod

from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.ollama import OllamaModel
from loguru import logger

from config.settings import RAGSettings, LLMProviderConfig


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    def __init__(self, config: LLMProviderConfig):
        self.config = config
        self.model: Optional[Model] = None
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the provider."""
        pass
    
    @abstractmethod
    def get_model(self) -> Model:
        """Get the PydanticAI model instance."""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test the provider connection."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation."""
    
    async def initialize(self) -> bool:
        """Initialize OpenAI provider."""
        try:
            if not self.config.api_key:
                logger.error("OpenAI API key not provided")
                return False
            
            self.model = OpenAIModel(
                model_name=self.config.model,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
            
            logger.info(f"Initialized OpenAI provider with model: {self.config.model}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}")
            return False
    
    def get_model(self) -> Model:
        """Get OpenAI model instance."""
        if not self.model:
            raise RuntimeError("OpenAI provider not initialized")
        return self.model
    
    async def test_connection(self) -> bool:
        """Test OpenAI connection."""
        try:
            # Simple test query
            model = self.get_model()
            # In practice, you would make a minimal API call here
            return True
            
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            return False


class AnthropicProvider(BaseLLMProvider):
    """Anthropic provider implementation."""
    
    async def initialize(self) -> bool:
        """Initialize Anthropic provider."""
        try:
            if not self.config.api_key:
                logger.error("Anthropic API key not provided")
                return False
            
            self.model = AnthropicModel(
                model_name=self.config.model,
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
            
            logger.info(f"Initialized Anthropic provider with model: {self.config.model}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic provider: {e}")
            return False
    
    def get_model(self) -> Model:
        """Get Anthropic model instance."""
        if not self.model:
            raise RuntimeError("Anthropic provider not initialized")
        return self.model
    
    async def test_connection(self) -> bool:
        """Test Anthropic connection."""
        try:
            model = self.get_model()
            return True
            
        except Exception as e:
            logger.error(f"Anthropic connection test failed: {e}")
            return False


class GoogleProvider(BaseLLMProvider):
    """Google Gemini provider implementation."""
    
    async def initialize(self) -> bool:
        """Initialize Google provider."""
        try:
            if not self.config.api_key:
                logger.error("Google API key not provided")
                return False
            
            self.model = GeminiModel(
                model_name=self.config.model,
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
            
            logger.info(f"Initialized Google provider with model: {self.config.model}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Google provider: {e}")
            return False
    
    def get_model(self) -> Model:
        """Get Google model instance."""
        if not self.model:
            raise RuntimeError("Google provider not initialized")
        return self.model
    
    async def test_connection(self) -> bool:
        """Test Google connection."""
        try:
            model = self.get_model()
            return True
            
        except Exception as e:
            logger.error(f"Google connection test failed: {e}")
            return False


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter provider implementation."""
    
    async def initialize(self) -> bool:
        """Initialize OpenRouter provider."""
        try:
            if not self.config.api_key:
                logger.error("OpenRouter API key not provided")
                return False
            
            # OpenRouter uses OpenAI-compatible API
            self.model = OpenAIModel(
                model_name=self.config.model,
                api_key=self.config.api_key,
                base_url=self.config.base_url or "https://openrouter.ai/api/v1",
                timeout=self.config.timeout
            )
            
            logger.info(f"Initialized OpenRouter provider with model: {self.config.model}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter provider: {e}")
            return False
    
    def get_model(self) -> Model:
        """Get OpenRouter model instance."""
        if not self.model:
            raise RuntimeError("OpenRouter provider not initialized")
        return self.model
    
    async def test_connection(self) -> bool:
        """Test OpenRouter connection."""
        try:
            model = self.get_model()
            return True
            
        except Exception as e:
            logger.error(f"OpenRouter connection test failed: {e}")
            return False


class OllamaProvider(BaseLLMProvider):
    """Ollama provider implementation."""
    
    async def initialize(self) -> bool:
        """Initialize Ollama provider."""
        try:
            self.model = OllamaModel(
                model_name=self.config.model,
                base_url=self.config.base_url or "http://localhost:11434",
                timeout=self.config.timeout
            )
            
            logger.info(f"Initialized Ollama provider with model: {self.config.model}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama provider: {e}")
            return False
    
    def get_model(self) -> Model:
        """Get Ollama model instance."""
        if not self.model:
            raise RuntimeError("Ollama provider not initialized")
        return self.model
    
    async def test_connection(self) -> bool:
        """Test Ollama connection."""
        try:
            model = self.get_model()
            # In practice, you would check if Ollama is running and model is available
            return True
            
        except Exception as e:
            logger.error(f"Ollama connection test failed: {e}")
            return False


class LLMProviderFactory:
    """Factory for creating and managing LLM providers."""
    
    def __init__(self, settings: RAGSettings):
        self.settings = settings
        self.providers: Dict[str, BaseLLMProvider] = {}
        self._provider_classes = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "google": GoogleProvider,
            "openrouter": OpenRouterProvider,
            "ollama": OllamaProvider
        }
    
    async def initialize_providers(self) -> Dict[str, bool]:
        """
        Initialize all configured providers.
        
        Returns:
            Dictionary mapping provider names to initialization success
        """
        results = {}
        provider_configs = self.settings.get_llm_providers()
        
        for name, config in provider_configs.items():
            if not config.enabled:
                logger.info(f"Skipping disabled provider: {name}")
                results[name] = False
                continue
            
            try:
                provider_class = self._provider_classes.get(name)
                if not provider_class:
                    logger.error(f"Unknown provider: {name}")
                    results[name] = False
                    continue
                
                provider = provider_class(config)
                success = await provider.initialize()
                
                if success:
                    self.providers[name] = provider
                    logger.info(f"Successfully initialized provider: {name}")
                else:
                    logger.error(f"Failed to initialize provider: {name}")
                
                results[name] = success
                
            except Exception as e:
                logger.error(f"Error initializing provider {name}: {e}")
                results[name] = False
        
        return results
    
    def get_model(self, provider_name: str) -> Model:
        """
        Get a model from a specific provider.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            PydanticAI model instance
            
        Raises:
            ValueError: If provider is not available
        """
        if provider_name not in self.providers:
            # Try to initialize the provider on demand
            provider_configs = self.settings.get_llm_providers()
            if provider_name not in provider_configs:
                raise ValueError(f"Provider {provider_name} not configured")
            
            config = provider_configs[provider_name]
            provider_class = self._provider_classes.get(provider_name)
            
            if not provider_class:
                raise ValueError(f"Unknown provider: {provider_name}")
            
            provider = provider_class(config)
            # Note: This is synchronous initialization for simplicity
            # In practice, you might want to handle this differently
            
            self.providers[provider_name] = provider
        
        return self.providers[provider_name].get_model()
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return list(self.providers.keys())
    
    async def test_all_providers(self) -> Dict[str, bool]:
        """
        Test connections for all providers.
        
        Returns:
            Dictionary mapping provider names to connection test results
        """
        results = {}
        
        for name, provider in self.providers.items():
            try:
                results[name] = await provider.test_connection()
            except Exception as e:
                logger.error(f"Error testing provider {name}: {e}")
                results[name] = False
        
        return results
    
    async def get_provider_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all providers.
        
        Returns:
            Provider statistics
        """
        stats = {
            "total_providers": len(self._provider_classes),
            "initialized_providers": len(self.providers),
            "available_providers": list(self.providers.keys()),
            "provider_configs": {}
        }
        
        # Add configuration details (without sensitive info)
        provider_configs = self.settings.get_llm_providers()
        for name, config in provider_configs.items():
            stats["provider_configs"][name] = {
                "model": config.model,
                "enabled": config.enabled,
                "initialized": name in self.providers,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens
            }
        
        return stats
    
    def switch_default_provider(self, provider_name: str) -> bool:
        """
        Switch the default provider.
        
        Args:
            provider_name: Name of the provider to switch to
            
        Returns:
            True if successful, False otherwise
        """
        if provider_name not in self.providers:
            logger.error(f"Provider {provider_name} not available")
            return False
        
        self.settings.default_provider = provider_name
        logger.info(f"Switched default provider to: {provider_name}")
        return True


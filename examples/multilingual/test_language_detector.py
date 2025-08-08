"""
Comprehensive test suite for the multilingual language detection and response generation module.

This test suite covers:
- Language detection accuracy and confidence scoring
- Multilingual response generation with various LLM providers
- Cultural context application and regional variations
- Error handling and fallback mechanisms
- Edge cases and boundary conditions

Author: LIXIL AI Hub Platform Team
Version: 1.0.0
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List

from language_detector import (
    LanguageDetector,
    MultilingualResponseGenerator,
    MultilingualProcessor,
    MultilingualRequest,
    MultilingualResponse,
    SupportedLanguage,
    LanguageDetectionResult
)


class TestLanguageDetector:
    """Test cases for the LanguageDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a LanguageDetector instance for testing."""
        return LanguageDetector()

    @pytest.mark.asyncio
    async def test_detect_english_text(self, detector):
        """Test detection of English text with high confidence."""
        text = "Hello, I need help with LIXIL AI policies and procedures."
        result = await detector.detect_language(text)
        
        assert result.language == SupportedLanguage.ENGLISH
        assert result.confidence > 0.7
        assert result.detection_method == "langdetect"
        assert isinstance(result.alternative_languages, list)

    @pytest.mark.asyncio
    async def test_detect_japanese_text(self, detector):
        """Test detection of Japanese text with cultural characters."""
        text = "こんにちは、LIXILのAIポリシーについて質問があります。"
        result = await detector.detect_language(text)
        
        assert result.language == SupportedLanguage.JAPANESE
        assert result.confidence > 0.8  # Should be higher due to unique characters
        assert result.detection_method == "langdetect"

    @pytest.mark.asyncio
    async def test_detect_french_text(self, detector):
        """Test detection of French text."""
        text = "Bonjour, j'ai besoin d'aide avec les politiques IA de LIXIL."
        result = await detector.detect_language(text)
        
        assert result.language == SupportedLanguage.FRENCH
        assert result.confidence > 0.6
        assert result.detection_method == "langdetect"

    @pytest.mark.asyncio
    async def test_detect_short_text_lower_confidence(self, detector):
        """Test that short text results in lower confidence scores."""
        short_text = "Hello"
        long_text = "Hello, I need comprehensive help with LIXIL AI policies and procedures for our organization."
        
        short_result = await detector.detect_language(short_text)
        long_result = await detector.detect_language(long_text)
        
        assert short_result.confidence < long_result.confidence
        assert both results detect English
        assert short_result.language == SupportedLanguage.ENGLISH
        assert long_result.language == SupportedLanguage.ENGLISH

    @pytest.mark.asyncio
    async def test_empty_text_raises_error(self, detector):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await detector.detect_language("")
            
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await detector.detect_language("   ")

    @pytest.mark.asyncio
    async def test_unsupported_language_fallback(self, detector):
        """Test fallback to English for unsupported languages."""
        with patch('langdetect.detect', return_value='fi'):  # Finnish - unsupported
            result = await detector.detect_language("Hei, tarvitsen apua")
            
            assert result.language == SupportedLanguage.ENGLISH
            assert result.confidence == 0.5
            assert result.detection_method == "langdetect"

    @pytest.mark.asyncio
    async def test_detection_exception_fallback(self, detector):
        """Test fallback when language detection fails."""
        with patch('langdetect.detect', side_effect=Exception("Detection failed")):
            result = await detector.detect_language("Some text")
            
            assert result.language == SupportedLanguage.ENGLISH
            assert result.confidence == 0.3
            assert result.detection_method == "fallback"

    def test_confidence_calculation_text_length(self, detector):
        """Test confidence calculation based on text length."""
        # Test short text
        short_confidence = detector._calculate_confidence("Hi", "en")
        
        # Test medium text
        medium_confidence = detector._calculate_confidence("Hello there, how are you?", "en")
        
        # Test long text
        long_text = "This is a very long text that should result in higher confidence scores " * 3
        long_confidence = detector._calculate_confidence(long_text, "en")
        
        assert short_confidence < medium_confidence
        assert medium_confidence <= long_confidence

    def test_confidence_calculation_language_patterns(self, detector):
        """Test confidence boost for language-specific patterns."""
        # English with common words
        english_confidence = detector._calculate_confidence("The cat is on the mat and it is sleeping", "en")
        
        # English without common words
        generic_confidence = detector._calculate_confidence("Cat mat sleeping", "en")
        
        assert english_confidence > generic_confidence


class TestMultilingualRequest:
    """Test cases for the MultilingualRequest model."""

    def test_valid_request_creation(self):
        """Test creation of valid multilingual request."""
        request = MultilingualRequest(
            text="Hello, I need help",
            preferred_language=SupportedLanguage.ENGLISH,
            context="AI policy question",
            user_region="US"
        )
        
        assert request.text == "Hello, I need help"
        assert request.preferred_language == SupportedLanguage.ENGLISH
        assert request.context == "AI policy question"
        assert request.user_region == "US"

    def test_text_validation_empty_string(self):
        """Test validation of empty text strings."""
        with pytest.raises(ValueError):
            MultilingualRequest(text="")
            
        with pytest.raises(ValueError):
            MultilingualRequest(text="   ")

    def test_text_validation_whitespace_trimming(self):
        """Test that whitespace is properly trimmed."""
        request = MultilingualRequest(text="  Hello world  ")
        assert request.text == "Hello world"

    def test_optional_fields_default_none(self):
        """Test that optional fields default to None."""
        request = MultilingualRequest(text="Hello")
        
        assert request.preferred_language is None
        assert request.context is None
        assert request.user_region is None

    def test_text_length_validation(self):
        """Test text length validation limits."""
        # Test maximum length
        long_text = "a" * 10001
        with pytest.raises(ValueError):
            MultilingualRequest(text=long_text)
            
        # Test acceptable length
        acceptable_text = "a" * 9999
        request = MultilingualRequest(text=acceptable_text)
        assert len(request.text) == 9999


class TestMultilingualResponseGenerator:
    """Test cases for the MultilingualResponseGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a MultilingualResponseGenerator for testing."""
        return MultilingualResponseGenerator(
            openai_api_key="test_openai_key",
            anthropic_api_key="test_anthropic_key", 
            gemini_api_key="test_gemini_key"
        )

    @pytest.fixture
    def sample_request(self):
        """Create a sample multilingual request."""
        return MultilingualRequest(
            text="What is LIXIL's AI policy?",
            preferred_language=SupportedLanguage.ENGLISH,
            user_region="US"
        )

    @pytest.fixture
    def sample_detection_result(self):
        """Create a sample language detection result."""
        return LanguageDetectionResult(
            language=SupportedLanguage.ENGLISH,
            confidence=0.9,
            alternative_languages=[(SupportedLanguage.FRENCH, 0.1)],
            detection_method="langdetect"
        )

    def test_cultural_context_mapping(self, generator):
        """Test that cultural contexts are properly mapped."""
        assert SupportedLanguage.JAPANESE in generator.cultural_contexts
        assert SupportedLanguage.GERMAN in generator.cultural_contexts
        assert SupportedLanguage.ENGLISH in generator.cultural_contexts
        
        japanese_context = generator.cultural_contexts[SupportedLanguage.JAPANESE]
        assert japanese_context["formality"] == "polite"
        assert japanese_context["honorifics"] is True

    @pytest.mark.asyncio
    async def test_generate_response_basic(self, generator, sample_request, sample_detection_result):
        """Test basic response generation."""
        with patch.object(generator, '_generate_with_llm', return_value="Test response"):
            response = await generator.generate_response(
                sample_request, sample_detection_result
            )
            
            assert isinstance(response, MultilingualResponse)
            assert response.detected_language == SupportedLanguage.ENGLISH
            assert response.response_language == SupportedLanguage.ENGLISH
            assert response.response_text == "Test response"
            assert response.confidence_score == 0.9

    @pytest.mark.asyncio
    async def test_preferred_language_override(self, generator, sample_detection_result):
        """Test that preferred language overrides detected language."""
        request = MultilingualRequest(
            text="Hello",
            preferred_language=SupportedLanguage.FRENCH
        )
        
        with patch.object(generator, '_generate_with_llm', return_value="Bonjour"):
            response = await generator.generate_response(request, sample_detection_result)
            
            assert response.detected_language == SupportedLanguage.ENGLISH
            assert response.response_language == SupportedLanguage.FRENCH

    def test_build_system_prompt_basic(self, generator):
        """Test system prompt building with basic parameters."""
        prompt = generator._build_system_prompt(
            SupportedLanguage.ENGLISH,
            {"formality": "professional", "business_style": "friendly"}
        )
        
        assert "English" in prompt
        assert "professional" in prompt
        assert "friendly" in prompt
        assert "LIXIL" in prompt

    def test_build_system_prompt_with_policy_context(self, generator):
        """Test system prompt building with policy context."""
        policy_context = "LIXIL AI Policy Section 3.2: Data Privacy"
        prompt = generator._build_system_prompt(
            SupportedLanguage.ENGLISH,
            {},
            policy_context
        )
        
        assert policy_context in prompt
        assert "Policy Context:" in prompt

    def test_fallback_response_languages(self, generator):
        """Test fallback responses in different languages."""
        english_fallback = generator._fallback_response(SupportedLanguage.ENGLISH)
        japanese_fallback = generator._fallback_response(SupportedLanguage.JAPANESE)
        french_fallback = generator._fallback_response(SupportedLanguage.FRENCH)
        
        assert "technical difficulties" in english_fallback.lower()
        assert "技術的な問題" in japanese_fallback
        assert "difficultés techniques" in french_fallback.lower()

    def test_fallback_response_unsupported_language(self, generator):
        """Test fallback response for unsupported language defaults to English."""
        # Test with a language not in fallback_messages
        fallback = generator._fallback_response(SupportedLanguage.KOREAN)
        english_fallback = generator._fallback_response(SupportedLanguage.ENGLISH)
        
        assert fallback == english_fallback

    @pytest.mark.asyncio
    async def test_llm_selection_japanese(self, generator):
        """Test that Japanese text uses Gemini LLM."""
        with patch.object(generator, '_generate_with_gemini', return_value="Japanese response") as mock_gemini:
            with patch.object(generator, '_generate_with_openai') as mock_openai:
                with patch.object(generator, '_generate_with_anthropic') as mock_anthropic:
                    
                    await generator._generate_with_llm(
                        "Test", SupportedLanguage.JAPANESE, {}, None
                    )
                    
                    mock_gemini.assert_called_once()
                    mock_openai.assert_not_called()
                    mock_anthropic.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_selection_english(self, generator):
        """Test that English text uses Anthropic LLM."""
        with patch.object(generator, '_generate_with_anthropic', return_value="English response") as mock_anthropic:
            with patch.object(generator, '_generate_with_openai') as mock_openai:
                with patch.object(generator, '_generate_with_gemini') as mock_gemini:
                    
                    await generator._generate_with_llm(
                        "Test", SupportedLanguage.ENGLISH, {}, None
                    )
                    
                    mock_anthropic.assert_called_once()
                    mock_openai.assert_not_called()
                    mock_gemini.assert_not_called()


class TestMultilingualProcessor:
    """Test cases for the main MultilingualProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create a MultilingualProcessor for testing."""
        return MultilingualProcessor(
            openai_api_key="test_openai",
            anthropic_api_key="test_anthropic",
            gemini_api_key="test_gemini"
        )

    @pytest.mark.asyncio
    async def test_process_request_end_to_end(self, processor):
        """Test end-to-end request processing."""
        request = MultilingualRequest(text="Hello, I need help with AI policies")
        
        with patch.object(processor.detector, 'detect_language') as mock_detect:
            with patch.object(processor.generator, 'generate_response') as mock_generate:
                
                # Mock detection result
                mock_detect.return_value = LanguageDetectionResult(
                    language=SupportedLanguage.ENGLISH,
                    confidence=0.9,
                    alternative_languages=[],
                    detection_method="langdetect"
                )
                
                # Mock response
                mock_generate.return_value = MultilingualResponse(
                    detected_language=SupportedLanguage.ENGLISH,
                    response_language=SupportedLanguage.ENGLISH,
                    response_text="Here's information about LIXIL AI policies...",
                    confidence_score=0.9,
                    cultural_context_applied=True,
                    fallback_used=False
                )
                
                result = await processor.process_request(request)
                
                assert isinstance(result, MultilingualResponse)
                assert result.detected_language == SupportedLanguage.ENGLISH
                assert result.response_language == SupportedLanguage.ENGLISH
                assert "LIXIL AI policies" in result.response_text
                
                mock_detect.assert_called_once_with(request.text)
                mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_request_with_policy_context(self, processor):
        """Test request processing with policy context."""
        request = MultilingualRequest(text="What are the data privacy rules?")
        policy_context = "LIXIL Data Privacy Policy: Section 2.1"
        
        with patch.object(processor.detector, 'detect_language') as mock_detect:
            with patch.object(processor.generator, 'generate_response') as mock_generate:
                
                mock_detect.return_value = LanguageDetectionResult(
                    language=SupportedLanguage.ENGLISH,
                    confidence=0.8,
                    alternative_languages=[],
                    detection_method="langdetect"
                )
                
                await processor.process_request(request, policy_context)
                
                # Verify policy context was passed to generator
                mock_generate.assert_called_once()
                call_args = mock_generate.call_args
                assert call_args[0][2] == policy_context  # Third argument should be policy_context


# Integration tests
class TestMultilingualIntegration:
    """Integration tests for the complete multilingual system."""

    @pytest.mark.asyncio
    async def test_multilingual_workflow_english_to_japanese(self):
        """Test complete workflow from English input to Japanese response."""
        processor = MultilingualProcessor(
            openai_api_key="test_openai",
            anthropic_api_key="test_anthropic", 
            gemini_api_key="test_gemini"
        )
        
        request = MultilingualRequest(
            text="What is LIXIL's AI policy?",
            preferred_language=SupportedLanguage.JAPANESE,
            user_region="JP"
        )
        
        with patch.object(processor.generator, '_generate_with_gemini', 
                         return_value="LIXILのAIポリシーについて説明いたします。"):
            
            result = await processor.process_request(request)
            
            assert result.detected_language == SupportedLanguage.ENGLISH
            assert result.response_language == SupportedLanguage.JAPANESE
            assert "LIXIL" in result.response_text
            assert result.cultural_context_applied is True

    @pytest.mark.asyncio
    async def test_error_handling_chain(self):
        """Test error handling through the entire processing chain."""
        processor = MultilingualProcessor(
            openai_api_key="invalid_key",
            anthropic_api_key="invalid_key",
            gemini_api_key="invalid_key"
        )
        
        request = MultilingualRequest(text="Test error handling")
        
        # Mock LLM failures to trigger fallback
        with patch.object(processor.generator, '_generate_with_openai', 
                         side_effect=Exception("API Error")):
            with patch.object(processor.generator, '_fallback_response', 
                             return_value="Fallback response"):
                
                result = await processor.process_request(request)
                
                assert result.fallback_used is True
                assert "Fallback response" in result.response_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


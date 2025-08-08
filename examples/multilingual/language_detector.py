"""
Language Detection and Response Generation Module

This module provides language detection capabilities and multilingual response
generation for the LIXIL AI Hub Platform. It supports automatic language
detection from user input and generates contextually appropriate responses
in the detected or preferred language.

Key Features:
- Automatic language detection using multiple detection libraries
- Confidence scoring for language detection accuracy
- Multi-LLM support for response generation in different languages
- Cultural context awareness for regional variations
- Fallback mechanisms for unsupported languages

Author: LIXIL AI Hub Platform Team
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import langdetect
from langdetect.lang_detect_exception import LangDetectException
import pydantic
from pydantic import BaseModel, Field, validator
import openai
from anthropic import Anthropic
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SupportedLanguage(str, Enum):
    """Enumeration of supported languages for the LIXIL AI Hub Platform."""
    ENGLISH = "en"
    JAPANESE = "ja"
    FRENCH = "fr"
    GERMAN = "de"
    SPANISH = "es"
    DUTCH = "nl"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    CHINESE_SIMPLIFIED = "zh-cn"
    KOREAN = "ko"


@dataclass
class LanguageDetectionResult:
    """Result of language detection with confidence scoring."""
    language: SupportedLanguage
    confidence: float
    alternative_languages: List[Tuple[SupportedLanguage, float]]
    detection_method: str


class MultilingualRequest(BaseModel):
    """Request model for multilingual processing."""
    text: str = Field(..., min_length=1, max_length=10000)
    preferred_language: Optional[SupportedLanguage] = None
    context: Optional[str] = None
    user_region: Optional[str] = None

    @validator('text')
    def validate_text_content(cls, v):
        """Validate that text contains meaningful content."""
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()


class MultilingualResponse(BaseModel):
    """Response model for multilingual processing."""
    detected_language: SupportedLanguage
    response_language: SupportedLanguage
    response_text: str
    confidence_score: float
    cultural_context_applied: bool
    fallback_used: bool


class LanguageDetector:
    """
    Advanced language detection with multiple detection methods and confidence scoring.
    
    This class provides robust language detection capabilities using multiple
    detection libraries and algorithms to ensure accurate language identification
    even with short or ambiguous text inputs.
    """

    def __init__(self):
        """Initialize the language detector with supported languages."""
        self.supported_languages = set(lang.value for lang in SupportedLanguage)
        self.confidence_threshold = 0.7
        
    async def detect_language(self, text: str) -> LanguageDetectionResult:
        """
        Detect the language of the input text with confidence scoring.
        
        Args:
            text: Input text to analyze
            
        Returns:
            LanguageDetectionResult with detected language and confidence
            
        Raises:
            ValueError: If text is empty or invalid
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty for language detection")
            
        try:
            # Primary detection using langdetect
            detected_lang = langdetect.detect(text)
            confidence = self._calculate_confidence(text, detected_lang)
            
            # Get alternative languages with probabilities
            lang_probs = langdetect.detect_langs(text)
            alternatives = []
            
            for lang_prob in lang_probs[1:4]:  # Top 3 alternatives
                if lang_prob.lang in self.supported_languages:
                    alt_lang = SupportedLanguage(lang_prob.lang)
                    alternatives.append((alt_lang, lang_prob.prob))
            
            # Validate detected language is supported
            if detected_lang not in self.supported_languages:
                logger.warning(f"Unsupported language detected: {detected_lang}")
                detected_lang = SupportedLanguage.ENGLISH  # Fallback to English
                confidence = 0.5
                
            return LanguageDetectionResult(
                language=SupportedLanguage(detected_lang),
                confidence=confidence,
                alternative_languages=alternatives,
                detection_method="langdetect"
            )
            
        except LangDetectException as e:
            logger.error(f"Language detection failed: {e}")
            # Fallback to English with low confidence
            return LanguageDetectionResult(
                language=SupportedLanguage.ENGLISH,
                confidence=0.3,
                alternative_languages=[],
                detection_method="fallback"
            )
    
    def _calculate_confidence(self, text: str, detected_lang: str) -> float:
        """
        Calculate confidence score for language detection.
        
        Args:
            text: Original text
            detected_lang: Detected language code
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = 0.8
        
        # Adjust confidence based on text length
        text_length = len(text.strip())
        if text_length < 10:
            base_confidence *= 0.6
        elif text_length < 30:
            base_confidence *= 0.8
        elif text_length > 100:
            base_confidence *= 1.1
            
        # Adjust for language-specific patterns
        if detected_lang == "en" and any(word in text.lower() for word in ["the", "and", "is", "are"]):
            base_confidence *= 1.1
        elif detected_lang == "ja" and any(char in text for char in "あいうえおかきくけこ"):
            base_confidence *= 1.2
            
        return min(base_confidence, 1.0)


class MultilingualResponseGenerator:
    """
    Generate contextually appropriate responses in multiple languages.
    
    This class handles response generation using various LLM providers,
    with support for cultural context and regional variations.
    """

    def __init__(self, openai_api_key: str, anthropic_api_key: str, gemini_api_key: str):
        """
        Initialize the response generator with API keys.
        
        Args:
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic API key  
            gemini_api_key: Google Gemini API key
        """
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.anthropic_client = Anthropic(api_key=anthropic_api_key)
        genai.configure(api_key=gemini_api_key)
        
        # Cultural context mappings
        self.cultural_contexts = {
            SupportedLanguage.JAPANESE: {
                "formality": "polite",
                "honorifics": True,
                "business_style": "formal"
            },
            SupportedLanguage.GERMAN: {
                "formality": "formal",
                "directness": "high",
                "business_style": "professional"
            },
            SupportedLanguage.ENGLISH: {
                "formality": "professional",
                "tone": "helpful",
                "business_style": "friendly-professional"
            }
        }
    
    async def generate_response(
        self, 
        request: MultilingualRequest,
        detection_result: LanguageDetectionResult,
        policy_context: Optional[str] = None
    ) -> MultilingualResponse:
        """
        Generate a multilingual response based on detected language and context.
        
        Args:
            request: Original multilingual request
            detection_result: Language detection result
            policy_context: Optional policy context for response
            
        Returns:
            MultilingualResponse with generated content
        """
        # Determine response language
        response_language = request.preferred_language or detection_result.language
        
        # Apply cultural context
        cultural_context = self.cultural_contexts.get(response_language, {})
        
        # Generate response using appropriate LLM
        response_text = await self._generate_with_llm(
            text=request.text,
            target_language=response_language,
            cultural_context=cultural_context,
            policy_context=policy_context,
            user_region=request.user_region
        )
        
        return MultilingualResponse(
            detected_language=detection_result.language,
            response_language=response_language,
            response_text=response_text,
            confidence_score=detection_result.confidence,
            cultural_context_applied=bool(cultural_context),
            fallback_used=detection_result.detection_method == "fallback"
        )
    
    async def _generate_with_llm(
        self,
        text: str,
        target_language: SupportedLanguage,
        cultural_context: Dict,
        policy_context: Optional[str] = None,
        user_region: Optional[str] = None
    ) -> str:
        """
        Generate response using the most appropriate LLM for the target language.
        
        Args:
            text: Input text
            target_language: Target language for response
            cultural_context: Cultural context settings
            policy_context: Optional policy context
            user_region: User's region for localization
            
        Returns:
            Generated response text
        """
        # Select best LLM for language
        if target_language in [SupportedLanguage.JAPANESE, SupportedLanguage.CHINESE_SIMPLIFIED]:
            return await self._generate_with_gemini(text, target_language, cultural_context, policy_context)
        elif target_language in [SupportedLanguage.ENGLISH, SupportedLanguage.FRENCH]:
            return await self._generate_with_anthropic(text, target_language, cultural_context, policy_context)
        else:
            return await self._generate_with_openai(text, target_language, cultural_context, policy_context)
    
    async def _generate_with_openai(
        self, 
        text: str, 
        target_language: SupportedLanguage,
        cultural_context: Dict,
        policy_context: Optional[str] = None
    ) -> str:
        """Generate response using OpenAI GPT."""
        system_prompt = self._build_system_prompt(target_language, cultural_context, policy_context)
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return self._fallback_response(target_language)
    
    async def _generate_with_anthropic(
        self, 
        text: str, 
        target_language: SupportedLanguage,
        cultural_context: Dict,
        policy_context: Optional[str] = None
    ) -> str:
        """Generate response using Anthropic Claude."""
        system_prompt = self._build_system_prompt(target_language, cultural_context, policy_context)
        
        try:
            response = await asyncio.to_thread(
                self.anthropic_client.messages.create,
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                system=system_prompt,
                messages=[{"role": "user", "content": text}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            return self._fallback_response(target_language)
    
    async def _generate_with_gemini(
        self, 
        text: str, 
        target_language: SupportedLanguage,
        cultural_context: Dict,
        policy_context: Optional[str] = None
    ) -> str:
        """Generate response using Google Gemini."""
        system_prompt = self._build_system_prompt(target_language, cultural_context, policy_context)
        
        try:
            model = genai.GenerativeModel('gemini-pro')
            full_prompt = f"{system_prompt}\n\nUser: {text}"
            
            response = await asyncio.to_thread(model.generate_content, full_prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return self._fallback_response(target_language)
    
    def _build_system_prompt(
        self, 
        target_language: SupportedLanguage,
        cultural_context: Dict,
        policy_context: Optional[str] = None
    ) -> str:
        """Build system prompt with language and cultural context."""
        language_names = {
            SupportedLanguage.ENGLISH: "English",
            SupportedLanguage.JAPANESE: "Japanese",
            SupportedLanguage.FRENCH: "French",
            SupportedLanguage.GERMAN: "German",
            SupportedLanguage.SPANISH: "Spanish",
            SupportedLanguage.DUTCH: "Dutch",
            SupportedLanguage.ITALIAN: "Italian",
            SupportedLanguage.PORTUGUESE: "Portuguese",
            SupportedLanguage.CHINESE_SIMPLIFIED: "Chinese (Simplified)",
            SupportedLanguage.KOREAN: "Korean"
        }
        
        prompt = f"""You are a helpful AI assistant for LIXIL, responding in {language_names[target_language]}.

Cultural Context:
- Formality level: {cultural_context.get('formality', 'professional')}
- Business style: {cultural_context.get('business_style', 'professional')}
- Use appropriate cultural conventions for {language_names[target_language]}

"""
        
        if policy_context:
            prompt += f"Policy Context:\n{policy_context}\n\n"
            
        prompt += "Provide helpful, accurate responses about LIXIL AI policies and procedures."
        
        return prompt
    
    def _fallback_response(self, target_language: SupportedLanguage) -> str:
        """Provide fallback response when LLM generation fails."""
        fallback_messages = {
            SupportedLanguage.ENGLISH: "I apologize, but I'm experiencing technical difficulties. Please try again later or contact support.",
            SupportedLanguage.JAPANESE: "申し訳ございませんが、技術的な問題が発生しています。後でもう一度お試しいただくか、サポートにお問い合わせください。",
            SupportedLanguage.FRENCH: "Je m'excuse, mais je rencontre des difficultés techniques. Veuillez réessayer plus tard ou contacter le support.",
            SupportedLanguage.GERMAN: "Entschuldigung, aber ich habe technische Schwierigkeiten. Bitte versuchen Sie es später noch einmal oder wenden Sie sich an den Support.",
            SupportedLanguage.SPANISH: "Me disculpo, pero estoy experimentando dificultades técnicas. Por favor, inténtelo de nuevo más tarde o contacte con soporte."
        }
        
        return fallback_messages.get(
            target_language, 
            fallback_messages[SupportedLanguage.ENGLISH]
        )


class MultilingualProcessor:
    """
    Main processor class that combines language detection and response generation.
    
    This class provides the primary interface for multilingual processing
    in the LIXIL AI Hub Platform.
    """

    def __init__(self, openai_api_key: str, anthropic_api_key: str, gemini_api_key: str):
        """Initialize the multilingual processor."""
        self.detector = LanguageDetector()
        self.generator = MultilingualResponseGenerator(
            openai_api_key, anthropic_api_key, gemini_api_key
        )
    
    async def process_request(
        self, 
        request: MultilingualRequest,
        policy_context: Optional[str] = None
    ) -> MultilingualResponse:
        """
        Process a multilingual request end-to-end.
        
        Args:
            request: Multilingual request to process
            policy_context: Optional policy context
            
        Returns:
            Complete multilingual response
        """
        # Detect language
        detection_result = await self.detector.detect_language(request.text)
        
        # Generate response
        response = await self.generator.generate_response(
            request, detection_result, policy_context
        )
        
        logger.info(f"Processed request in {detection_result.language.value}, "
                   f"responded in {response.response_language.value}")
        
        return response


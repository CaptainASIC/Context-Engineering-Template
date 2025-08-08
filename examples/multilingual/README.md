# Multilingual Language Detection and Response Generation

This module provides comprehensive multilingual support for the LIXIL AI Hub Platform, including automatic language detection, cultural context awareness, and intelligent response generation using multiple LLM providers.

## Features

- **Automatic Language Detection**: Robust language detection with confidence scoring
- **Multi-LLM Support**: Intelligent routing between OpenAI, Anthropic, and Gemini based on language
- **Cultural Context**: Region-specific cultural adaptations and formality levels
- **Fallback Mechanisms**: Graceful degradation when detection or generation fails
- **10 Supported Languages**: English, Japanese, French, German, Spanish, Dutch, Italian, Portuguese, Chinese (Simplified), Korean

## Quick Start

```python
from multilingual.language_detector import MultilingualProcessor, MultilingualRequest, SupportedLanguage

# Initialize processor with API keys
processor = MultilingualProcessor(
    openai_api_key="your-openai-key",
    anthropic_api_key="your-anthropic-key", 
    gemini_api_key="your-gemini-key"
)

# Create request
request = MultilingualRequest(
    text="What is LIXIL's AI policy?",
    preferred_language=SupportedLanguage.JAPANESE,
    user_region="JP"
)

# Process request
response = await processor.process_request(request)
print(f"Detected: {response.detected_language}")
print(f"Response: {response.response_text}")
```

## Architecture

### Language Detection
- Primary detection using `langdetect` library
- Confidence scoring based on text length and language patterns
- Alternative language suggestions with probability scores
- Fallback to English for unsupported languages

### Response Generation
- **Japanese/Chinese**: Uses Google Gemini for better Asian language support
- **English/French**: Uses Anthropic Claude for nuanced responses
- **Other languages**: Uses OpenAI GPT for broad language coverage
- Cultural context application based on target language

### Cultural Context Mapping
- **Japanese**: Polite formality, honorifics, formal business style
- **German**: High formality, direct communication, professional tone
- **English**: Professional friendly tone, helpful approach

## Testing

Run the comprehensive test suite:

```bash
pip install -r requirements.txt
pytest test_language_detector.py -v
```

## Configuration

Set environment variables for API keys:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key" 
export GEMINI_API_KEY="your-gemini-key"
```

## Error Handling

The module includes robust error handling:
- Language detection failures fall back to English
- LLM API failures trigger fallback responses in target language
- Invalid input validation with clear error messages
- Graceful degradation maintains system availability

## Performance Considerations

- Async/await patterns for concurrent processing
- Intelligent LLM selection reduces latency
- Confidence thresholds prevent low-quality responses
- Caching opportunities for repeated language detection

## Integration with LIXIL AI Hub

This module integrates with:
- Policy parsing for context-aware responses
- Knowledge graphs for enhanced cultural context
- Admin portal for language preference management
- Analytics for language usage tracking


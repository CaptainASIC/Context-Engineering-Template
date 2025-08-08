# PydanticAI RAG Agent

A comprehensive Retrieval-Augmented Generation (RAG) agent implementation using PydanticAI with multi-LLM support and advanced document processing capabilities.

## Features

- **Multi-LLM Support**: OpenAI, Anthropic, Google Gemini, OpenRouter, and Ollama
- **Advanced Document Processing**: PDF, DOCX, Markdown, HTML, and text files
- **Vector Storage**: ChromaDB with sentence transformers for embeddings
- **Intelligent Chunking**: Overlapping chunks with metadata preservation
- **Streaming Responses**: Real-time response streaming
- **CLI Interface**: Rich command-line interface with progress indicators
- **Comprehensive Testing**: Full test suite with mocks and fixtures
- **Health Monitoring**: Built-in health checks and statistics

## Architecture

```
rag_agent/
├── src/
│   ├── agents/          # PydanticAI agent implementation
│   ├── models/          # Document and chunk models
│   ├── retrieval/       # Vector store and search
│   ├── llm_providers/   # Multi-LLM provider factory
│   └── utils/           # Utility functions
├── config/              # Configuration management
├── tests/               # Comprehensive test suite
├── data/                # Document storage
└── main.py              # CLI interface
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag_agent
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Configuration

Create a `.env` file with your API keys:

```env
# LLM Provider API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
OPENROUTER_API_KEY=your_openrouter_key

# Ollama Configuration (if using local Ollama)
OLLAMA_BASE_URL=http://localhost:11434

# Default Settings
DEFAULT_PROVIDER=openai
DEBUG=false
LOG_LEVEL=INFO

# Vector Database Settings
VECTOR_DB_PROVIDER=chroma
COLLECTION_NAME=rag_documents
EMBEDDING_MODEL=all-MiniLM-L6-v2
PERSIST_DIRECTORY=./data/vectordb

# Retrieval Settings
TOP_K=5
SIMILARITY_THRESHOLD=0.7
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

## Usage

### Command Line Interface

The RAG agent provides a rich CLI interface for all operations:

#### Query the Agent

```bash
# Basic query
python main.py query "What is artificial intelligence?"

# Query with specific provider
python main.py query "Explain machine learning" --provider anthropic

# Streaming response
python main.py query "Tell me about neural networks" --stream

# Query with additional context
python main.py query "How does this relate to deep learning?" --context "Previous discussion about AI"
```

#### Document Management

```bash
# Add single document
python main.py add-document path/to/document.pdf --title "AI Research Paper" --author "Dr. Smith"

# Add directory of documents
python main.py add-directory ./documents --recursive --types "pdf,txt,md"

# Add with tags
python main.py add-document report.txt --tags "research,ai,2024"
```

#### System Information

```bash
# View statistics
python main.py stats

# Check health
python main.py health

# List available providers
python main.py providers
```

### Programmatic Usage

```python
import asyncio
from src.agents.rag_agent import PydanticRAGAgent, RAGQuery, RAGContext
from src.models.document import Document, DocumentType, DocumentMetadata
from config.settings import get_settings

async def main():
    # Initialize agent
    settings = get_settings()
    agent = PydanticRAGAgent(settings)
    
    # Add documents
    document = Document(
        content="Artificial intelligence is the simulation of human intelligence...",
        document_type=DocumentType.TEXT,
        source_path="ai_intro.txt",
        metadata=DocumentMetadata(
            title="Introduction to AI",
            author="AI Researcher",
            tags=["ai", "introduction"]
        )
    )
    
    await agent.add_documents([document])
    
    # Query the agent
    context = RAGContext(
        user_id="user123",
        session_id="session456"
    )
    
    query = RAGQuery(
        question="What is artificial intelligence?",
        context=context
    )
    
    response = await agent.query(query)
    print(f"Answer: {response.answer}")
    print(f"Sources: {len(response.sources)}")
    print(f"Confidence: {response.confidence}")

# Run the example
asyncio.run(main())
```

### Streaming Responses

```python
async def stream_example():
    agent = PydanticRAGAgent()
    query = RAGQuery(question="Explain quantum computing")
    
    async for chunk in agent.stream_query(query):
        print(chunk, end="", flush=True)
    print()  # New line at end
```

## Multi-LLM Provider Support

The agent supports multiple LLM providers with automatic fallback:

### OpenAI
```python
settings.openai_api_key = "your-key"
settings.default_provider = "openai"
```

### Anthropic Claude
```python
settings.anthropic_api_key = "your-key"
settings.default_provider = "anthropic"
```

### Google Gemini
```python
settings.google_api_key = "your-key"
settings.default_provider = "google"
```

### OpenRouter
```python
settings.openrouter_api_key = "your-key"
settings.default_provider = "openrouter"
```

### Ollama (Local)
```python
settings.ollama_base_url = "http://localhost:11434"
settings.default_provider = "ollama"
```

### Dynamic Provider Switching

```python
# Switch providers at runtime
success = await agent.switch_provider("anthropic")
if success:
    print("Switched to Anthropic Claude")
```

## Document Processing

The agent supports various document types with intelligent processing:

### Supported Formats
- **PDF**: Extracted using pypdf with page metadata
- **DOCX**: Microsoft Word documents with formatting preservation
- **Markdown**: Full markdown parsing with section detection
- **HTML**: Web content with tag filtering
- **Text**: Plain text files with encoding detection

### Chunking Strategy
- **Intelligent Splitting**: Respects sentence and paragraph boundaries
- **Overlapping Chunks**: Configurable overlap for context preservation
- **Metadata Preservation**: Document metadata carried through to chunks
- **Feature Extraction**: Automatic text feature analysis

## Vector Storage

Uses ChromaDB for efficient vector storage and retrieval:

### Features
- **Persistent Storage**: Data persists between sessions
- **Similarity Search**: Cosine similarity with configurable thresholds
- **Metadata Filtering**: Filter by document type, author, tags, etc.
- **Batch Operations**: Efficient bulk document processing

### Configuration
```python
vector_db = VectorDBConfig(
    provider="chroma",
    collection_name="my_documents",
    embedding_model="all-MiniLM-L6-v2",
    dimension=384,
    distance_metric="cosine",
    persist_directory="./data/vectordb"
)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_rag_agent.py

# Run with verbose output
pytest -v tests/
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Mock Tests**: External service mocking
- **Error Handling**: Exception and edge case testing

## Performance Optimization

### Embedding Caching
```python
# Enable embedding caching for repeated queries
settings.cache_embeddings = True
settings.cache_directory = "./data/cache"
```

### Batch Processing
```python
# Process multiple documents efficiently
documents = [doc1, doc2, doc3, ...]
result = await agent.add_documents(documents)
```

### Memory Management
```python
# Configure chunk size for memory efficiency
settings.retrieval.chunk_size = 300  # Smaller chunks
settings.retrieval.top_k = 3         # Fewer results
```

## Monitoring and Observability

### Health Checks
```python
health = await agent.health_check()
print(f"Status: {health['status']}")
print(f"Components: {health['components']}")
```

### Statistics
```python
stats = await agent.get_stats()
print(f"Documents: {stats['vector_store']['total_documents']}")
print(f"Chunks: {stats['vector_store']['total_chunks']}")
```

### Logging
```python
from loguru import logger

# Configure logging
logger.add("rag_agent.log", rotation="10 MB", retention="7 days")
```

## Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py", "serve"]
```

### Environment Variables
```bash
# Production settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Security
SECRET_KEY=your-secret-key
ALLOWED_HOSTS=your-domain.com

# Database
VECTOR_DB_PERSIST_DIRECTORY=/data/vectordb
```

## Best Practices

### Document Preparation
1. **Clean Text**: Remove unnecessary formatting and artifacts
2. **Consistent Structure**: Use consistent headings and sections
3. **Metadata**: Include comprehensive metadata for better retrieval
4. **Size Management**: Split very large documents appropriately

### Query Optimization
1. **Specific Questions**: More specific queries yield better results
2. **Context Usage**: Provide conversation context for follow-up questions
3. **Provider Selection**: Choose appropriate LLM for query type
4. **Threshold Tuning**: Adjust similarity thresholds based on use case

### Performance Tuning
1. **Chunk Size**: Balance between context and precision
2. **Top-K Selection**: More results provide context but increase latency
3. **Model Selection**: Choose embedding models based on domain
4. **Caching**: Implement caching for frequently accessed content

## Troubleshooting

### Common Issues

**No API Key Configured**
```bash
Error: No LLM providers are configured and enabled
Solution: Set API keys in .env file
```

**Vector Store Connection Failed**
```bash
Error: Failed to initialize vector store
Solution: Check persist_directory permissions and disk space
```

**Document Processing Failed**
```bash
Error: Failed to add document
Solution: Check file format and encoding
```

### Debug Mode
```python
# Enable debug logging
settings.debug = True
settings.log_level = "DEBUG"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **PydanticAI**: Modern AI agent framework
- **ChromaDB**: Vector database for embeddings
- **Sentence Transformers**: High-quality text embeddings
- **Rich**: Beautiful terminal output
- **Typer**: Modern CLI framework

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed information
4. Join our community discussions

---

Built with ❤️ using PydanticAI and modern Python practices.


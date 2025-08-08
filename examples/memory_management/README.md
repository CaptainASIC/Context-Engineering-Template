# mem0 Memory Management Integration

A comprehensive memory management system using mem0 for conversation context, user preferences, entity relationships, and intelligent fact storage with multi-LLM support and advanced learning capabilities.

## Features

- **Conversation Memory**: Persistent conversation context with intelligent summarization
- **User Preferences**: Adaptive preference learning and storage
- **Entity Management**: Entity recognition, properties, and relationship mapping
- **Fact Storage**: Structured fact storage with verification and temporal tracking
- **Multi-LLM Support**: Compatible with OpenAI, Anthropic, Google Gemini, OpenRouter, and Ollama
- **Semantic Search**: Advanced memory retrieval using vector similarity
- **Memory Analytics**: Comprehensive insights and usage patterns
- **Privacy Controls**: User data management and retention policies
- **Real-time Learning**: Continuous adaptation from user interactions

## Architecture

```
memory_management/
├── src/
│   ├── services/        # Core memory service and business logic
│   ├── models/          # Pydantic models for memory types
│   ├── storage/         # Storage backends and adapters
│   └── utils/           # Utility functions and helpers
├── config/              # Configuration management
├── tests/               # Comprehensive test suite
├── examples/            # Usage examples and demos
├── data/                # Sample data and exports
└── scripts/             # Setup and maintenance scripts
```

## Installation

### 1. Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file:

```env
# LLM Configuration (choose your primary provider)
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
LLM_API_KEY=your-api-key
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=1000

# Fallback LLM (optional)
FALLBACK_LLM_PROVIDER=anthropic
FALLBACK_LLM_MODEL=claude-3-haiku-20240307
FALLBACK_LLM_API_KEY=your-fallback-api-key

# Embedding Configuration
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
EMBEDDING_BATCH_SIZE=32

# Vector Store Configuration
VECTOR_STORE_TYPE=chroma
VECTOR_STORE_COLLECTION=memories
VECTOR_DIMENSION=384

# Storage Configuration
STORAGE_BACKEND=sqlite
DATABASE_CONNECTION_STRING=sqlite:///memory.db
MAX_MEMORIES_PER_USER=10000
DEFAULT_TTL_DAYS=365

# Memory Processing
MAX_CONTEXT_LENGTH=4000
CONTEXT_OVERLAP=200
EXTRACT_ENTITIES=true
EXTRACT_RELATIONSHIPS=true
EXTRACT_FACTS=true
ENABLE_SUMMARIZATION=true
SUMMARIZATION_THRESHOLD=2000

# Search Configuration
RELEVANCE_THRESHOLD=0.7
MAX_RELEVANT_MEMORIES=10
DEFAULT_SEARCH_LIMIT=10
MAX_SEARCH_LIMIT=100

# Application Settings
DEBUG=false
LOG_LEVEL=INFO
ENVIRONMENT=production
API_HOST=0.0.0.0
API_PORT=8000

# Optional: Redis for caching
REDIS_URL=redis://localhost:6379
ENABLE_QUERY_CACHE=true
QUERY_CACHE_TTL=300

# Optional: Advanced vector stores
# QDRANT_HOST=localhost
# QDRANT_PORT=6333
# WEAVIATE_URL=http://localhost:8080
# PINECONE_API_KEY=your-pinecone-key
# PINECONE_ENVIRONMENT=us-west1-gcp
```

## Usage

### Basic Setup

```python
import asyncio
from src.services.memory_service import MemoryService
from config.settings import get_settings

async def main():
    # Initialize memory service
    settings = get_settings()
    memory_service = MemoryService(settings)
    await memory_service.initialize()
    
    print("Memory service initialized successfully!")
    
    # Check health
    health = await memory_service.health_check()
    print(f"Service status: {health['status']}")

asyncio.run(main())
```

### Conversation Memory Management

```python
from src.services.memory_service import get_memory_service

async def conversation_example():
    memory = await get_memory_service()
    
    user_id = "user_123"
    conversation_id = "conv_456"
    
    # Add conversation messages
    await memory.add_conversation_memory(
        user_id=user_id,
        conversation_id=conversation_id,
        message="I'm interested in learning about machine learning",
        role="user",
        metadata={"source": "web_chat"}
    )
    
    await memory.add_conversation_memory(
        user_id=user_id,
        conversation_id=conversation_id,
        message="I'd be happy to help you learn about machine learning! What specific area interests you most?",
        role="assistant",
        metadata={"response_time_ms": 1200}
    )
    
    # Get conversation context
    context = await memory.get_conversation_context(
        user_id=user_id,
        conversation_id=conversation_id,
        max_messages=20
    )
    
    print(f"Retrieved {len(context)} messages from conversation")
    for msg in context:
        print(f"{msg['role']}: {msg['content']}")

# Run the example
await conversation_example()
```

### User Preference Learning

```python
async def preference_example():
    memory = await get_memory_service()
    user_id = "user_123"
    
    # Add explicit preferences
    await memory.add_user_preference(
        user_id=user_id,
        preference_type="communication_style",
        preference_value="detailed_explanations",
        confidence=1.0,
        source="explicit",
        metadata={"set_by": "user_settings"}
    )
    
    # Add inferred preferences
    await memory.add_user_preference(
        user_id=user_id,
        preference_type="topics_of_interest",
        preference_value="machine_learning",
        confidence=0.8,
        source="inferred",
        metadata={"inferred_from": "conversation_analysis"}
    )
    
    # Add learned preferences
    await memory.add_user_preference(
        user_id=user_id,
        preference_type="expertise_level",
        preference_value="intermediate",
        confidence=0.7,
        source="learned",
        metadata={"learning_source": "interaction_patterns"}
    )
    
    # Retrieve user preferences
    preferences = await memory.get_user_preferences(
        user_id=user_id,
        min_confidence=0.6
    )
    
    print("User Preferences:")
    for pref_type, pref_data in preferences.items():
        print(f"  {pref_type}: {pref_data['value']} (confidence: {pref_data['confidence']})")

await preference_example()
```

### Entity and Relationship Management

```python
async def entity_example():
    memory = await get_memory_service()
    user_id = "user_123"
    
    # Add entity memories
    await memory.add_entity_memory(
        user_id=user_id,
        entity_name="Python",
        entity_type="programming_language",
        properties={
            "paradigm": "multi-paradigm",
            "typing": "dynamic",
            "first_appeared": 1991,
            "creator": "Guido van Rossum",
            "use_cases": ["web_development", "data_science", "automation", "ai_ml"]
        },
        relationships=[
            {
                "target_entity": "Django",
                "relationship_type": "has_framework",
                "properties": {"domain": "web_development", "popularity": "high"}
            },
            {
                "target_entity": "NumPy",
                "relationship_type": "has_library",
                "properties": {"domain": "scientific_computing", "importance": "critical"}
            }
        ],
        metadata={"source": "knowledge_base", "verified": True}
    )
    
    await memory.add_entity_memory(
        user_id=user_id,
        entity_name="Machine Learning",
        entity_type="field_of_study",
        properties={
            "parent_field": "artificial_intelligence",
            "key_concepts": ["supervised_learning", "unsupervised_learning", "reinforcement_learning"],
            "applications": ["prediction", "classification", "clustering", "recommendation"]
        },
        relationships=[
            {
                "target_entity": "Python",
                "relationship_type": "implemented_in",
                "properties": {"popularity": "very_high", "ecosystem": "rich"}
            }
        ]
    )
    
    # Search for entities
    results = await memory.search_memories(
        user_id=user_id,
        query="Python programming language",
        memory_types=["entity"],
        limit=5
    )
    
    print("Entity Search Results:")
    for result in results:
        print(f"  {result.content} (score: {result.similarity_score:.3f})")

await entity_example()
```

### Fact Storage and Verification

```python
async def fact_example():
    memory = await get_memory_service()
    user_id = "user_123"
    
    # Add various types of facts
    facts = [
        {
            "fact": "User has 5 years of experience in software development",
            "fact_type": "professional",
            "confidence": 1.0,
            "source": "user_profile",
            "metadata": {"verified": True, "verification_date": "2024-01-01"}
        },
        {
            "fact": "User prefers Python over JavaScript for backend development",
            "fact_type": "preference",
            "confidence": 0.9,
            "source": "conversation_analysis",
            "metadata": {"based_on": "multiple_conversations"}
        },
        {
            "fact": "Machine learning models require training data",
            "fact_type": "technical",
            "confidence": 1.0,
            "source": "knowledge_base",
            "metadata": {"domain": "machine_learning", "fundamental": True}
        },
        {
            "fact": "User completed a machine learning course in 2023",
            "fact_type": "achievement",
            "confidence": 0.8,
            "source": "inferred",
            "metadata": {"inferred_from": "conversation_context"}
        }
    ]
    
    for fact_data in facts:
        await memory.add_fact_memory(user_id=user_id, **fact_data)
    
    # Search for facts
    results = await memory.search_memories(
        user_id=user_id,
        query="software development experience",
        memory_types=["fact"],
        limit=10
    )
    
    print("Fact Search Results:")
    for result in results:
        metadata = result.metadata
        print(f"  Fact: {metadata.get('fact', result.content)}")
        print(f"    Type: {metadata.get('fact_type', 'unknown')}")
        print(f"    Confidence: {metadata.get('confidence', 0.0)}")
        print(f"    Source: {metadata.get('source', 'unknown')}")
        print()

await fact_example()
```

### Advanced Memory Search

```python
async def search_example():
    memory = await get_memory_service()
    user_id = "user_123"
    
    # Semantic search across all memory types
    results = await memory.search_memories(
        user_id=user_id,
        query="Python machine learning experience",
        limit=10,
        threshold=0.7
    )
    
    print("General Search Results:")
    for result in results:
        print(f"  Type: {result.memory_type}")
        print(f"  Content: {result.content[:100]}...")
        print(f"  Score: {result.similarity_score:.3f}")
        print()
    
    # Filtered search by memory type
    preference_results = await memory.search_memories(
        user_id=user_id,
        query="programming preferences",
        memory_types=["user_preference"],
        limit=5
    )
    
    print("Preference Search Results:")
    for result in preference_results:
        metadata = result.metadata
        print(f"  {metadata.get('preference_type')}: {metadata.get('preference_value')}")
    
    # Search with custom filters
    recent_results = await memory.search_memories(
        user_id=user_id,
        query="recent conversations",
        memory_types=["conversation"],
        filters={"role": "user"},
        limit=5
    )
    
    print("Recent User Messages:")
    for result in recent_results:
        print(f"  {result.content}")

await search_example()
```

### Memory Analytics and Insights

```python
async def analytics_example():
    memory = await get_memory_service()
    user_id = "user_123"
    
    # Get comprehensive memory summary
    summary = await memory.get_user_memory_summary(user_id=user_id)
    
    print("Memory Summary:")
    print(f"  Total memories: {summary['total_memories']}")
    print(f"  Conversations: {summary['conversation_count']}")
    print(f"  Preferences: {summary['preference_count']}")
    print(f"  Facts: {summary['fact_count']}")
    
    print("\nMemory Types:")
    for memory_type, count in summary['memory_types'].items():
        print(f"  {memory_type}: {count}")
    
    print("\nRecent Activity:")
    for date, count in summary['recent_activity'].items():
        print(f"  {date}: {count} memories")
    
    print("\nTop Entities:")
    for entity, count in summary['top_entities'].items():
        print(f"  {entity}: {count} mentions")
    
    # Get service statistics
    stats = await memory.get_statistics()
    
    print("\nService Statistics:")
    print(f"  Memories created: {stats['memories_created']}")
    print(f"  Memories retrieved: {stats['memories_retrieved']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")

await analytics_example()
```

## Configuration

### LLM Provider Configuration

#### OpenAI
```python
# settings.py or .env
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo  # or gpt-4, gpt-4-turbo
LLM_API_KEY=your-openai-api-key
LLM_ORGANIZATION=your-org-id  # optional
```

#### Anthropic
```python
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-haiku-20240307  # or claude-3-sonnet-20240229
LLM_API_KEY=your-anthropic-api-key
```

#### Google Gemini
```python
LLM_PROVIDER=google
LLM_MODEL=gemini-pro  # or gemini-pro-vision
LLM_API_KEY=your-google-api-key
```

#### OpenRouter (Multi-Model Access)
```python
LLM_PROVIDER=openai  # OpenRouter uses OpenAI-compatible API
LLM_API_BASE=https://openrouter.ai/api/v1
LLM_MODEL=anthropic/claude-3-haiku  # or any supported model
LLM_API_KEY=your-openrouter-api-key
```

#### Ollama (Local Models)
```python
LLM_PROVIDER=ollama
LLM_API_BASE=http://localhost:11434
LLM_MODEL=llama2  # or any installed model
# No API key needed for local Ollama
```

### Vector Store Configuration

#### ChromaDB (Default)
```python
VECTOR_STORE_TYPE=chroma
CHROMA_PERSIST_DIRECTORY=./chroma_db
VECTOR_DIMENSION=384
```

#### FAISS
```python
VECTOR_STORE_TYPE=faiss
FAISS_INDEX_PATH=./faiss_index
VECTOR_DIMENSION=384
```

#### Qdrant
```python
VECTOR_STORE_TYPE=qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=your-qdrant-api-key  # if using Qdrant Cloud
```

#### Weaviate
```python
VECTOR_STORE_TYPE=weaviate
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=your-weaviate-api-key  # if authentication enabled
```

#### Pinecone
```python
VECTOR_STORE_TYPE=pinecone
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX_NAME=memories
```

### Advanced Configuration

```python
# Memory Processing Settings
class ConversationConfig:
    max_context_length: int = 4000
    context_overlap: int = 200
    extract_entities: bool = True
    extract_relationships: bool = True
    extract_facts: bool = True
    enable_summarization: bool = True
    summarization_threshold: int = 2000
    relevance_threshold: float = 0.7
    max_relevant_memories: int = 10

# User Preference Settings
class UserPreferenceConfig:
    track_preferences: List[str] = [
        "communication_style",
        "topics_of_interest", 
        "expertise_areas",
        "learning_preferences",
        "interaction_patterns"
    ]
    learning_rate: float = 0.1
    confidence_threshold: float = 0.8
    update_frequency: str = "immediate"

# Storage Settings
class MemoryStorageConfig:
    backend: str = "sqlite"
    connection_string: str = "sqlite:///memory.db"
    max_memories_per_user: int = 10000
    default_ttl_days: int = 365
    cleanup_interval_hours: int = 24
```

## API Integration

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from src.services.memory_service import get_memory_service
from src.models.memory_models import MemoryQuery, MemorySearchResult

app = FastAPI(title="Memory Management API")

@app.post("/memories/conversation")
async def add_conversation_memory(
    user_id: str,
    conversation_id: str,
    message: str,
    role: str,
    metadata: dict = None
):
    memory = await get_memory_service()
    
    try:
        memory_id = await memory.add_conversation_memory(
            user_id=user_id,
            conversation_id=conversation_id,
            message=message,
            role=role,
            metadata=metadata
        )
        return {"memory_id": memory_id, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memories/search")
async def search_memories(query: MemoryQuery):
    memory = await get_memory_service()
    
    try:
        results = await memory.search_memories(**query.to_search_params())
        return {
            "results": [result.to_dict() for result in results],
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}/preferences")
async def get_user_preferences(
    user_id: str,
    preference_types: List[str] = None,
    min_confidence: float = 0.5
):
    memory = await get_memory_service()
    
    try:
        preferences = await memory.get_user_preferences(
            user_id=user_id,
            preference_types=preference_types,
            min_confidence=min_confidence
        )
        return {"preferences": preferences}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}/summary")
async def get_memory_summary(user_id: str):
    memory = await get_memory_service()
    
    try:
        summary = await memory.get_user_memory_summary(user_id=user_id)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    memory = await get_memory_service()
    return await memory.health_check()
```

### Streamlit Integration

```python
import streamlit as st
import asyncio
from src.services.memory_service import get_memory_service

st.title("Memory Management Dashboard")

# Initialize memory service
@st.cache_resource
def init_memory_service():
    return asyncio.run(get_memory_service())

memory = init_memory_service()

# User selection
user_id = st.text_input("User ID", value="demo_user")

# Memory search
st.header("Search Memories")
query = st.text_input("Search Query")
memory_types = st.multiselect(
    "Memory Types",
    ["conversation", "user_preference", "entity", "fact"],
    default=["conversation"]
)

if st.button("Search"):
    if query:
        results = asyncio.run(memory.search_memories(
            user_id=user_id,
            query=query,
            memory_types=memory_types,
            limit=10
        ))
        
        st.subheader(f"Found {len(results)} results")
        for result in results:
            with st.expander(f"{result.memory_type} - Score: {result.similarity_score:.3f}"):
                st.write(result.content)
                st.json(result.metadata)

# User preferences
st.header("User Preferences")
if st.button("Load Preferences"):
    preferences = asyncio.run(memory.get_user_preferences(user_id=user_id))
    
    for pref_type, pref_data in preferences.items():
        st.write(f"**{pref_type}**: {pref_data['value']} (confidence: {pref_data['confidence']:.2f})")

# Memory summary
st.header("Memory Summary")
if st.button("Generate Summary"):
    summary = asyncio.run(memory.get_user_memory_summary(user_id=user_id))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Memories", summary['total_memories'])
    with col2:
        st.metric("Conversations", summary['conversation_count'])
    with col3:
        st.metric("Preferences", summary['preference_count'])
    
    st.subheader("Memory Types")
    st.bar_chart(summary['memory_types'])
    
    st.subheader("Recent Activity")
    st.line_chart(summary['recent_activity'])
```

## Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test categories
pytest tests/test_memory_service.py
pytest tests/test_memory_models.py

# Run integration tests
pytest tests/integration/ -v

# Run performance tests
pytest tests/performance/ -v --benchmark-only
```

### Test Configuration

```python
# conftest.py
import pytest
import asyncio
from src.services.memory_service import MemoryService
from config.settings import create_test_settings

@pytest.fixture
async def memory_service():
    """Test memory service fixture."""
    settings = create_test_settings()
    service = MemoryService(settings)
    await service.initialize()
    
    yield service
    
    # Cleanup
    await service.close()

@pytest.fixture
def sample_memories():
    """Sample memory data for testing."""
    return [
        {
            "user_id": "test_user",
            "conversation_id": "conv_1",
            "message": "Hello, I need help with Python",
            "role": "user"
        },
        {
            "user_id": "test_user",
            "preference_type": "communication_style",
            "preference_value": "detailed",
            "confidence": 0.9
        }
    ]
```

## Performance Optimization

### Memory Caching

```python
# Configure memory caching
class MemoryService:
    def __init__(self, settings):
        self._memory_cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._max_cache_size = 1000
    
    def _cache_memory(self, memory_id: str, memory_record: Any):
        """Cache memory with TTL and size limits."""
        if len(self._memory_cache) >= self._max_cache_size:
            # Remove oldest entries
            oldest_keys = sorted(
                self._memory_cache.keys(),
                key=lambda k: self._memory_cache[k]["timestamp"]
            )[:100]
            
            for key in oldest_keys:
                del self._memory_cache[key]
        
        self._memory_cache[memory_id] = {
            "record": memory_record,
            "timestamp": datetime.now(timezone.utc),
            "ttl": self._cache_ttl
        }
```

### Batch Processing

```python
async def batch_add_memories(
    memory_service: MemoryService,
    memories: List[Dict[str, Any]],
    batch_size: int = 50
):
    """Add memories in batches for better performance."""
    results = []
    
    for i in range(0, len(memories), batch_size):
        batch = memories[i:i + batch_size]
        batch_tasks = []
        
        for memory_data in batch:
            memory_type = memory_data.pop("memory_type")
            
            if memory_type == "conversation":
                task = memory_service.add_conversation_memory(**memory_data)
            elif memory_type == "preference":
                task = memory_service.add_user_preference(**memory_data)
            elif memory_type == "entity":
                task = memory_service.add_entity_memory(**memory_data)
            elif memory_type == "fact":
                task = memory_service.add_fact_memory(**memory_data)
            
            batch_tasks.append(task)
        
        # Execute batch concurrently
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        results.extend(batch_results)
        
        # Small delay between batches to avoid overwhelming the system
        await asyncio.sleep(0.1)
    
    return results
```

### Vector Store Optimization

```python
# Optimize vector store performance
def optimize_vector_store_config(memory_count: int) -> Dict[str, Any]:
    """Optimize vector store configuration based on memory count."""
    config = {}
    
    if memory_count < 1000:
        # Small dataset - use FAISS
        config = {
            "type": "faiss",
            "index_type": "flat",  # Exact search
            "dimension": 384
        }
    elif memory_count < 100000:
        # Medium dataset - use ChromaDB
        config = {
            "type": "chroma",
            "collection_name": "memories",
            "dimension": 384,
            "distance_metric": "cosine"
        }
    else:
        # Large dataset - use Qdrant with optimization
        config = {
            "type": "qdrant",
            "collection_name": "memories",
            "dimension": 384,
            "distance_metric": "cosine",
            "hnsw_config": {
                "m": 16,
                "ef_construct": 100,
                "full_scan_threshold": 10000
            }
        }
    
    return config
```

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create data directory
RUN mkdir -p /app/data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import asyncio; from src.services.memory_service import get_memory_service; asyncio.run(get_memory_service().health_check())"

# Run application
CMD ["python", "main.py"]
```

### Production Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  memory-service:
    build: .
    environment:
      - LLM_API_KEY=${LLM_API_KEY}
      - DATABASE_CONNECTION_STRING=postgresql://user:pass@postgres:5432/memorydb
      - VECTOR_STORE_TYPE=qdrant
      - QDRANT_HOST=qdrant
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - qdrant
      - redis
    ports:
      - "8000:8000"
    volumes:
      - memory_data:/app/data
  
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: memorydb
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
  
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  memory_data:
  postgres_data:
  qdrant_data:
  redis_data:
```

## Best Practices

### Memory Management

1. **Conversation Context**: Keep conversations focused and summarize long contexts
2. **Preference Learning**: Use confidence scores and multiple sources for validation
3. **Entity Recognition**: Normalize entity names and maintain consistent relationships
4. **Fact Verification**: Implement verification workflows for critical facts

### Performance

1. **Caching**: Cache frequently accessed memories with appropriate TTL
2. **Batch Operations**: Process multiple memories in batches for efficiency
3. **Vector Optimization**: Choose appropriate vector store based on scale
4. **Memory Cleanup**: Implement retention policies and regular cleanup

### Privacy and Security

1. **Data Anonymization**: Remove or hash sensitive personal information
2. **Access Control**: Implement proper user authentication and authorization
3. **Data Retention**: Follow privacy regulations with appropriate retention policies
4. **Audit Logging**: Log all memory access and modifications

### Monitoring

```python
# Add monitoring and metrics
from prometheus_client import Counter, Histogram, Gauge

# Metrics
MEMORY_OPERATIONS = Counter('memory_operations_total', 'Total memory operations', ['operation', 'type'])
MEMORY_SEARCH_DURATION = Histogram('memory_search_duration_seconds', 'Memory search duration')
ACTIVE_MEMORIES = Gauge('active_memories_total', 'Total active memories', ['user_id'])

class MemoryService:
    async def add_conversation_memory(self, **kwargs):
        MEMORY_OPERATIONS.labels(operation='add', type='conversation').inc()
        # ... implementation
    
    async def search_memories(self, **kwargs):
        with MEMORY_SEARCH_DURATION.time():
            # ... implementation
            pass
```

## Troubleshooting

### Common Issues

**Memory Service Not Initializing**
```python
# Check mem0 configuration
config = service._build_mem0_config()
print("mem0 config:", config)

# Verify LLM connectivity
try:
    from mem0 import Memory
    memory = Memory(config)
    print("mem0 initialized successfully")
except Exception as e:
    print(f"mem0 initialization failed: {e}")
```

**Vector Store Connection Issues**
```python
# Test vector store connectivity
if settings.storage.vector_store == "chroma":
    import chromadb
    client = chromadb.Client()
    print("ChromaDB connected")
elif settings.storage.vector_store == "qdrant":
    from qdrant_client import QdrantClient
    client = QdrantClient(host=settings.qdrant.host, port=settings.qdrant.port)
    print("Qdrant connected")
```

**Search Results Quality Issues**
```python
# Adjust search parameters
results = await memory.search_memories(
    user_id=user_id,
    query=query,
    threshold=0.6,  # Lower threshold for more results
    limit=20        # Increase limit
)

# Check embedding quality
embedding_config = settings.get_embedding_config()
print("Embedding model:", embedding_config["model"])
print("Embedding dimension:", embedding_config["dimension"])
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable mem0 debug mode
settings.debug = True
settings.log_level = "DEBUG"

# Monitor memory operations
stats = await memory.get_statistics()
print("Memory statistics:", stats)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Ensure all tests pass
5. Update documentation
6. Submit a pull request

## License

This project is licensed under the MIT License.

---

Built with ❤️ using mem0, modern Python practices, and multi-LLM support for intelligent memory management.


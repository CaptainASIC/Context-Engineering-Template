# Neon Database & pgvector Semantic Search

A comprehensive implementation of semantic search using Neon PostgreSQL database with pgvector extension for high-performance vector similarity search and document retrieval.

## Features

- **Neon Database Integration**: Native support for Neon's serverless PostgreSQL
- **pgvector Extension**: High-performance vector similarity search
- **Multiple Distance Metrics**: Cosine, L2, and inner product similarity
- **Async Operations**: Full async/await support with connection pooling
- **Document Management**: Complete document lifecycle with chunking
- **Embedding Pipeline**: Automated embedding generation and storage
- **Advanced Search**: Filtered search with metadata and user context
- **Performance Optimization**: Vector indexes and query optimization
- **Monitoring & Analytics**: Search logging and performance metrics

## Architecture

```
vector_db/
├── src/
│   ├── models/          # SQLAlchemy models and schemas
│   ├── services/        # Database and business logic services
│   ├── embeddings/      # Embedding generation and management
│   ├── search/          # Search algorithms and ranking
│   └── utils/           # Utility functions and helpers
├── config/              # Configuration management
├── tests/               # Comprehensive test suite
├── migrations/          # Database migration scripts
├── data/                # Sample data and exports
└── scripts/             # Setup and maintenance scripts
```

## Installation

### 1. Database Setup

**Option A: Neon Database (Recommended)**
```bash
# Sign up at https://neon.tech
# Create a new project and get connection details
# Copy connection string from Neon console
```

**Option B: Local PostgreSQL with pgvector**
```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Install pgvector extension
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# Create database
sudo -u postgres createdb vectordb
sudo -u postgres psql vectordb -c "CREATE EXTENSION vector;"
```

### 2. Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file:

```env
# Database Configuration
DATABASE_HOST=your-neon-host.neon.tech
DATABASE_PORT=5432
DATABASE_NAME=neondb
DATABASE_USERNAME=your-username
DATABASE_PASSWORD=your-password
DATABASE_SSLMODE=require

# Neon-specific (optional)
NEON_PROJECT_ID=your-project-id
NEON_BRANCH_NAME=main
NEON_REGION=us-east-1
NEON_API_KEY=your-api-key

# Connection Pool Settings
DATABASE_MIN_CONNECTIONS=5
DATABASE_MAX_CONNECTIONS=20
DATABASE_CONNECTION_TIMEOUT=30

# Embedding Configuration
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_DIMENSION=384
EMBEDDING_BATCH_SIZE=32

# Search Configuration
DEFAULT_SEARCH_LIMIT=10
MAX_SEARCH_LIMIT=100
SIMILARITY_THRESHOLD=0.7
DISTANCE_METRIC=cosine

# Application Settings
DEBUG=false
LOG_LEVEL=INFO
ENVIRONMENT=production
BATCH_SIZE=100
MAX_FILE_SIZE_MB=50

# Optional: Redis for caching
REDIS_URL=redis://localhost:6379
ENABLE_QUERY_CACHE=true
QUERY_CACHE_TTL=300
```

## Usage

### Basic Setup

```python
import asyncio
from src.services.database_service import DatabaseService
from src.models.database_models import Document, DocumentChunk
from config.settings import get_settings

async def main():
    # Initialize database service
    settings = get_settings()
    db_service = DatabaseService(settings)
    await db_service.initialize()
    
    print("Database service initialized successfully!")
    
    # Check health
    health = await db_service.health_check()
    print(f"Database status: {health['status']}")

asyncio.run(main())
```

### Document Management

```python
from src.services.database_service import get_database_service
import hashlib

async def add_document():
    db = await get_database_service()
    
    # Document data
    content = "This is a sample document about machine learning and AI."
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    
    # Create document record
    async with db.get_async_session() as session:
        document = Document(
            title="ML Guide",
            content=content,
            content_hash=content_hash,
            document_type="text",
            source_type="upload",
            language="en",
            word_count=len(content.split()),
            character_count=len(content),
            processing_status="completed",
            user_id="user_123",
            tags=["machine-learning", "ai"],
            custom_metadata={"category": "education"}
        )
        
        session.add(document)
        await session.commit()
        await session.refresh(document)
        
        print(f"Created document: {document.id}")
        return document

# Run the function
document = await add_document()
```

### Document Chunking

```python
async def chunk_document(document_id: str, chunk_size: int = 1000, overlap: int = 200):
    db = await get_database_service()
    
    async with db.get_async_session() as session:
        # Get document
        result = await session.execute(
            text("SELECT * FROM documents WHERE id = :id"),
            {"id": document_id}
        )
        doc_data = result.fetchone()
        
        if not doc_data:
            raise ValueError("Document not found")
        
        content = doc_data.content
        chunks = []
        
        # Simple chunking strategy
        for i in range(0, len(content), chunk_size - overlap):
            chunk_content = content[i:i + chunk_size]
            
            if not chunk_content.strip():
                continue
            
            chunk = DocumentChunk(
                document_id=document_id,
                content=chunk_content,
                content_hash=hashlib.sha256(chunk_content.encode()).hexdigest(),
                chunk_index=len(chunks),
                start_position=i,
                end_position=min(i + chunk_size, len(content)),
                word_count=len(chunk_content.split()),
                character_count=len(chunk_content),
                chunk_type="text"
            )
            
            chunks.append(chunk)
        
        # Save chunks
        session.add_all(chunks)
        await session.commit()
        
        print(f"Created {len(chunks)} chunks for document {document_id}")
        return chunks
```

### Embedding Generation

```python
from sentence_transformers import SentenceTransformer
import numpy as np

async def generate_embeddings():
    db = await get_database_service()
    
    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create or get embedding model record
    embedding_model = await db.create_embedding_model(
        name="all-MiniLM-L6-v2",
        provider="sentence-transformers",
        dimension=384,
        max_sequence_length=512
    )
    
    # Get chunks without embeddings
    chunks_query = """
    SELECT dc.id, dc.content 
    FROM document_chunks dc
    LEFT JOIN chunk_embeddings ce ON dc.id = ce.chunk_id AND ce.model_id = :model_id
    WHERE ce.id IS NULL
    LIMIT 100
    """
    
    chunks = await db.execute_query(chunks_query, {"model_id": embedding_model.id})
    
    if not chunks:
        print("No chunks to process")
        return
    
    # Generate embeddings in batches
    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [chunk["content"] for chunk in batch]
        
        # Generate embeddings
        embeddings = model.encode(texts, normalize_embeddings=True)
        
        # Save embeddings
        embedding_records = []
        for j, chunk in enumerate(batch):
            embedding_vector = embeddings[j].tolist()
            
            embedding_records.append({
                "chunk_id": chunk["id"],
                "model_id": embedding_model.id,
                "embedding": embedding_vector,
                "source_text_hash": hashlib.sha256(chunk["content"].encode()).hexdigest(),
                "token_count": len(texts[j].split())
            })
        
        # Bulk insert embeddings
        await db.bulk_insert("chunk_embeddings", embedding_records)
        
        print(f"Processed batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
    
    print(f"Generated embeddings for {len(chunks)} chunks")
```

### Semantic Search

```python
async def semantic_search(query: str, limit: int = 10, threshold: float = 0.7):
    db = await get_database_service()
    
    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Get embedding model record
    embedding_model = await db.get_embedding_model("all-MiniLM-L6-v2")
    if not embedding_model:
        raise ValueError("Embedding model not found")
    
    # Generate query embedding
    query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()
    
    # Search similar chunks
    results = await db.search_similar_chunks(
        query_embedding=query_embedding,
        model_id=str(embedding_model.id),
        limit=limit,
        threshold=threshold,
        distance_metric="cosine"
    )
    
    # Log search query
    await db.log_search_query(
        query_text=query,
        query_type="semantic",
        model_id=str(embedding_model.id),
        results_count=len(results),
        execution_time_ms=100.0  # Would measure actual time
    )
    
    return results

# Example search
results = await semantic_search("machine learning algorithms", limit=5)
for result in results:
    print(f"Score: {result['similarity']:.3f}")
    print(f"Document: {result['document_title']}")
    print(f"Content: {result['content'][:200]}...")
    print("---")
```

### Advanced Search with Filters

```python
async def advanced_search(
    query: str,
    document_types: List[str] = None,
    tags: List[str] = None,
    user_id: str = None,
    date_range: tuple = None
):
    db = await get_database_service()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate query embedding
    query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()
    
    # Get embedding model
    embedding_model = await db.get_embedding_model("all-MiniLM-L6-v2")
    
    # Build filters
    filters = {}
    if document_types:
        filters["document_types"] = document_types
    if tags:
        filters["tags"] = tags
    if user_id:
        filters["user_id"] = user_id
    
    # Search with filters
    results = await db.search_similar_chunks(
        query_embedding=query_embedding,
        model_id=str(embedding_model.id),
        limit=20,
        threshold=0.6,
        distance_metric="cosine",
        filters=filters
    )
    
    return results

# Example filtered search
results = await advanced_search(
    query="neural networks",
    document_types=["pdf", "text"],
    tags=["machine-learning"],
    user_id="user_123"
)
```

## Database Schema

### Core Tables

#### documents
- **id**: UUID primary key
- **title**: Document title
- **content**: Full document content
- **content_hash**: SHA256 hash for deduplication
- **document_type**: File type (text, pdf, docx, etc.)
- **source_type**: Source (upload, url, api)
- **processing_status**: Processing state
- **user_id**: Owner user ID
- **tags**: JSON array of tags
- **custom_metadata**: JSON metadata object

#### document_chunks
- **id**: UUID primary key
- **document_id**: Foreign key to documents
- **content**: Chunk text content
- **chunk_index**: Position in document
- **start_position**: Character start position
- **end_position**: Character end position
- **chunk_type**: Type of chunk (text, header, table)

#### embedding_models
- **id**: UUID primary key
- **name**: Model identifier
- **provider**: Model provider (sentence-transformers, openai)
- **dimension**: Embedding vector dimension
- **max_sequence_length**: Maximum input length

#### chunk_embeddings
- **id**: UUID primary key
- **chunk_id**: Foreign key to document_chunks
- **model_id**: Foreign key to embedding_models
- **embedding**: Vector column (pgvector)
- **source_text_hash**: Hash of source text

### Vector Indexes

```sql
-- IVFFlat index for approximate nearest neighbor search
CREATE INDEX idx_chunk_embeddings_vector_ivfflat
ON chunk_embeddings 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- HNSW index for high-dimensional vectors
CREATE INDEX idx_chunk_embeddings_vector_hnsw
ON chunk_embeddings 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

## Performance Optimization

### Vector Index Configuration

```python
# Configure index parameters based on data size
async def optimize_vector_indexes():
    db = await get_database_service()
    
    # Get embedding count
    stats = await db.get_database_stats()
    embedding_count = sum(e["embedding_count"] for e in stats["embeddings"])
    
    # Calculate optimal parameters
    if embedding_count < 10000:
        lists = max(embedding_count // 100, 10)
    else:
        lists = int(embedding_count ** 0.5)
    
    # Recreate index with optimal parameters
    await db.execute_command(f"""
        DROP INDEX IF EXISTS idx_chunk_embeddings_vector_ivfflat;
        CREATE INDEX idx_chunk_embeddings_vector_ivfflat
        ON chunk_embeddings 
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = {lists});
    """)
    
    print(f"Optimized vector index with {lists} lists for {embedding_count} embeddings")
```

### Query Optimization

```python
# Set optimal search parameters
async def optimize_search_parameters():
    db = await get_database_service()
    
    # Set ivfflat probes based on accuracy requirements
    await db.execute_command("SET ivfflat.probes = 10;")  # Higher = more accurate, slower
    
    # Set work_mem for large vector operations
    await db.execute_command("SET work_mem = '256MB';")
    
    # Enable parallel query execution
    await db.execute_command("SET max_parallel_workers_per_gather = 4;")
```

### Batch Processing

```python
async def batch_process_documents(file_paths: List[str]):
    db = await get_database_service()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    batch_size = 100
    
    for i in range(0, len(file_paths), batch_size):
        batch_files = file_paths[i:i + batch_size]
        
        # Process documents in parallel
        documents = []
        chunks = []
        embeddings = []
        
        for file_path in batch_files:
            # Load and process document
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Create document record
            doc = create_document_record(content, file_path)
            documents.append(doc)
            
            # Create chunks
            doc_chunks = create_chunks(content, doc.id)
            chunks.extend(doc_chunks)
            
            # Generate embeddings
            chunk_texts = [chunk.content for chunk in doc_chunks]
            chunk_embeddings = model.encode(chunk_texts, normalize_embeddings=True)
            
            for j, chunk in enumerate(doc_chunks):
                embeddings.append({
                    "chunk_id": chunk.id,
                    "embedding": chunk_embeddings[j].tolist()
                })
        
        # Bulk insert all data
        await db.bulk_insert("documents", [doc.dict() for doc in documents])
        await db.bulk_insert("document_chunks", [chunk.dict() for chunk in chunks])
        await db.bulk_insert("chunk_embeddings", embeddings)
        
        print(f"Processed batch {i//batch_size + 1}")
```

## Monitoring and Analytics

### Search Analytics

```python
async def get_search_analytics(days: int = 7):
    db = await get_database_service()
    
    analytics = await db.execute_query("""
        SELECT 
            DATE(created_at) as search_date,
            COUNT(*) as total_searches,
            AVG(execution_time_ms) as avg_execution_time,
            AVG(results_count) as avg_results_count,
            COUNT(*) FILTER (WHERE cache_hit = true) as cache_hits,
            COUNT(DISTINCT user_id) as unique_users
        FROM search_queries
        WHERE created_at >= NOW() - INTERVAL '%s days'
        GROUP BY DATE(created_at)
        ORDER BY search_date DESC
    """, {"days": days})
    
    return analytics

# Get popular queries
async def get_popular_queries(limit: int = 10):
    db = await get_database_service()
    
    popular = await db.execute_query("""
        SELECT 
            query_text,
            COUNT(*) as search_count,
            AVG(results_count) as avg_results,
            AVG(execution_time_ms) as avg_time
        FROM search_queries
        WHERE created_at >= NOW() - INTERVAL '30 days'
        GROUP BY query_text
        ORDER BY search_count DESC
        LIMIT :limit
    """, {"limit": limit})
    
    return popular
```

### Performance Monitoring

```python
async def monitor_performance():
    db = await get_database_service()
    
    # Database statistics
    stats = await db.get_database_stats()
    
    # Health check
    health = await db.health_check()
    
    # Query performance
    slow_queries = await db.execute_query("""
        SELECT 
            query_text,
            execution_time_ms,
            results_count,
            created_at
        FROM search_queries
        WHERE execution_time_ms > 1000  -- Queries slower than 1 second
        ORDER BY execution_time_ms DESC
        LIMIT 10
    """)
    
    return {
        "stats": stats,
        "health": health,
        "slow_queries": slow_queries
    }
```

## Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test categories
pytest tests/test_database_service.py
pytest tests/test_embeddings.py
pytest tests/test_search.py

# Run integration tests (requires test database)
pytest tests/integration/ --db-url=postgresql://test:test@localhost/test_vectordb
```

### Test Configuration

```python
# conftest.py
import pytest
import asyncio
from src.services.database_service import DatabaseService
from config.settings import create_test_settings

@pytest.fixture
async def test_db():
    """Test database fixture."""
    settings = create_test_settings()
    db = DatabaseService(settings)
    await db.initialize()
    
    yield db
    
    await db.disconnect()

@pytest.fixture
def sample_embeddings():
    """Sample embedding vectors for testing."""
    import numpy as np
    return np.random.rand(10, 384).tolist()
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

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import asyncio; from src.services.database_service import get_database_service; asyncio.run(get_database_service().health_check())"

# Run application
CMD ["python", "main.py"]
```

### Production Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  vector-db:
    build: .
    environment:
      - DATABASE_HOST=your-neon-host.neon.tech
      - DATABASE_PASSWORD=${DATABASE_PASSWORD}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    ports:
      - "8000:8000"
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
  
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

volumes:
  redis_data:
```

## Best Practices

### Vector Search Optimization

1. **Index Selection**: Use IVFFlat for large datasets, HNSW for smaller high-dimensional data
2. **Probe Tuning**: Balance accuracy vs speed with ivfflat.probes setting
3. **Batch Processing**: Process embeddings in batches for better throughput
4. **Normalization**: Normalize embeddings for cosine similarity

### Database Performance

1. **Connection Pooling**: Configure appropriate pool sizes
2. **Query Optimization**: Use prepared statements and parameter binding
3. **Index Maintenance**: Regularly update vector index statistics
4. **Partitioning**: Consider table partitioning for large datasets

### Security

1. **SSL/TLS**: Always use encrypted connections in production
2. **Authentication**: Use strong passwords and consider certificate auth
3. **Network Security**: Restrict database access to application servers
4. **Data Encryption**: Consider column-level encryption for sensitive data

## Troubleshooting

### Common Issues

**Vector Index Not Used**
```sql
-- Check if index exists
SELECT indexname FROM pg_indexes WHERE tablename = 'chunk_embeddings';

-- Force index usage
SET enable_seqscan = off;
```

**Slow Vector Queries**
```sql
-- Increase probes for better accuracy (slower)
SET ivfflat.probes = 20;

-- Check query plan
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM chunk_embeddings 
ORDER BY embedding <=> '[0.1,0.2,0.3]'::vector 
LIMIT 10;
```

**Connection Pool Exhaustion**
```python
# Monitor connection pool
health = await db.health_check()
print(health["details"]["connection_pool"])

# Increase pool size
settings.database.max_connections = 50
```

### Debug Mode

```python
# Enable debug logging
settings.debug = True
settings.log_level = "DEBUG"

# Log all SQL queries
import logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
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

Built with ❤️ using Neon, PostgreSQL, pgvector, and modern Python practices.


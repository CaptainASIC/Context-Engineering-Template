# Neo4j Knowledge Graph Integration

A comprehensive Neo4j knowledge graph implementation for connecting policies, use cases, and people with advanced graph operations and relationship management.

## Features

- **Multi-Entity Support**: Person, Policy, UseCase, Document, Organization, and Concept nodes
- **Rich Relationships**: Typed relationships with properties and validation
- **Async Operations**: Full async/await support with connection pooling
- **Schema Validation**: Pydantic models with property validation
- **Graph Algorithms**: Shortest path, neighbor discovery, and traversal
- **Batch Operations**: Efficient bulk data processing
- **Health Monitoring**: Built-in health checks and statistics
- **Index Management**: Automatic index and constraint creation

## Architecture

```
knowledge_graph/
├── src/
│   ├── models/          # Pydantic graph models
│   ├── services/        # Neo4j database service
│   ├── queries/         # Pre-built Cypher queries
│   └── utils/           # Utility functions
├── config/              # Configuration management
├── tests/               # Comprehensive test suite
├── data/                # Sample data and exports
└── scripts/             # Database setup scripts
```

## Installation

1. **Install Neo4j Database**:
   ```bash
   # Using Docker
   docker run -d \
     --name neo4j \
     -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/password \
     neo4j:5.15
   
   # Or install locally from https://neo4j.com/download/
   ```

2. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your Neo4j credentials
   ```

## Configuration

Create a `.env` file with your Neo4j settings:

```env
# Neo4j Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# Connection Settings
NEO4J_MAX_CONNECTION_LIFETIME=30
NEO4J_MAX_CONNECTION_POOL_SIZE=50
NEO4J_CONNECTION_ACQUISITION_TIMEOUT=60

# Application Settings
DEBUG=false
LOG_LEVEL=INFO
BATCH_SIZE=1000

# NLP Settings
ENABLE_NLP=true
NLP_MODEL=en_core_web_sm
SIMILARITY_THRESHOLD=0.8
```

## Usage

### Basic Operations

```python
import asyncio
from src.services.neo4j_service import Neo4jService
from src.models.graph_models import PersonNode, PolicyNode, WorksForRelationship
from config.settings import get_settings

async def main():
    # Initialize service
    settings = get_settings()
    service = Neo4jService(settings)
    await service.connect()
    
    # Create a person node
    person_data = {
        "name": "Alice Johnson",
        "email": "alice@company.com",
        "title": "Data Scientist",
        "department": "Analytics",
        "skills": ["Python", "Machine Learning", "Neo4j"]
    }
    
    person_result = await service.create_node("Person", person_data)
    person_id = person_result["id"]
    
    # Create an organization node
    org_data = {
        "name": "Tech Corp",
        "organization_type": "company",
        "industry": "Technology",
        "size": "large"
    }
    
    org_result = await service.create_node("Organization", org_data)
    org_id = org_result["id"]
    
    # Create relationship
    relationship_props = {
        "role": "Senior Data Scientist",
        "start_date": "2023-01-15",
        "is_current": True
    }
    
    await service.create_relationship(
        from_node_id=person_id,
        to_node_id=org_id,
        relationship_type="WORKS_FOR",
        properties=relationship_props
    )
    
    print(f"Created person {person_id} working for organization {org_id}")

asyncio.run(main())
```

### Using Pydantic Models

```python
from src.models.graph_models import PersonNode, PolicyNode, ImplementsRelationship

# Create typed nodes
person = PersonNode(
    name="Bob Smith",
    email="bob@company.com",
    title="Policy Manager",
    skills=["Compliance", "Risk Management"]
)

policy = PolicyNode(
    name="Data Retention Policy",
    description="Guidelines for data retention and deletion",
    version="1.2",
    status="active",
    category="data_governance"
)

# Convert to Neo4j properties
person_props = person.to_neo4j_properties()
policy_props = policy.to_neo4j_properties()
```

### Advanced Queries

```python
# Find shortest path between nodes
path = await service.find_shortest_path(
    from_node_id=person_id,
    to_node_id=policy_id,
    max_length=5,
    relationship_types=["IMPLEMENTS", "MANAGES"]
)

if path:
    print(f"Path length: {path['length']}")
    print(f"Nodes in path: {len(path['nodes'])}")

# Get node neighbors
neighbors = await service.get_node_neighbors(
    node_id=person_id,
    direction="out",
    relationship_types=["WORKS_FOR", "MANAGES"],
    limit=10
)

for neighbor in neighbors:
    node = neighbor["node"]
    rel = neighbor["relationship"]
    print(f"{node['properties']['name']} ({rel['type']})")
```

### Batch Operations

```python
# Create multiple nodes efficiently
people_data = [
    {"name": "Charlie Brown", "email": "charlie@company.com", "title": "Developer"},
    {"name": "Diana Prince", "email": "diana@company.com", "title": "Designer"},
    {"name": "Eve Wilson", "email": "eve@company.com", "title": "Manager"}
]

# Process in batches
for person_data in people_data:
    result = await service.create_node("Person", person_data, merge=True)
    print(f"Created/updated person: {result['id']}")
```

## Graph Schema

### Node Types

#### Person
- **Properties**: name, email, title, department, skills, phone, location, bio
- **Relationships**: WORKS_FOR, MANAGES, IMPLEMENTS, AUTHORED

#### Policy
- **Properties**: name, description, version, status, category, tags, effective_date, expiry_date
- **Relationships**: IMPLEMENTED_BY, REFERENCES, PART_OF

#### UseCase
- **Properties**: name, description, priority, status, business_value, complexity, tags
- **Relationships**: IMPLEMENTS, DEPENDS_ON, RELATES_TO

#### Document
- **Properties**: title, content, document_type, source, author, tags, file_path
- **Relationships**: AUTHORED, REFERENCES, PART_OF

#### Organization
- **Properties**: name, organization_type, industry, size, location, website
- **Relationships**: EMPLOYS, PART_OF, MANAGES

#### Concept
- **Properties**: name, definition, category, synonyms, related_terms, domain
- **Relationships**: RELATES_TO, SIMILAR_TO, PART_OF

### Relationship Types

#### WORKS_FOR
- **Between**: Person → Organization
- **Properties**: role, start_date, end_date, is_current

#### MANAGES
- **Between**: Person → Person/Organization
- **Properties**: start_date, end_date, is_current, management_type

#### IMPLEMENTS
- **Between**: Person/Organization → Policy/UseCase
- **Properties**: implementation_date, status, notes, completion_percentage

#### RELATES_TO
- **Between**: Any → Any
- **Properties**: relationship_type, strength, confidence, source, bidirectional

#### AUTHORED
- **Between**: Person → Document
- **Properties**: authored_date, role, contribution_percentage

## Database Setup

### Initialize Schema

```python
from src.services.neo4j_service import Neo4jService

async def setup_database():
    service = Neo4jService()
    await service.connect()
    
    # Create indexes
    index_results = await service.create_indexes()
    print(f"Created indexes: {index_results['created']}")
    
    # Create constraints
    constraint_results = await service.create_constraints()
    print(f"Created constraints: {constraint_results['created']}")
    
    await service.disconnect()

asyncio.run(setup_database())
```

### Sample Data Import

```python
async def import_sample_data():
    service = Neo4jService()
    await service.connect()
    
    # Import from JSON file
    import json
    with open('data/sample_data.json', 'r') as f:
        data = json.load(f)
    
    # Create nodes
    for node_data in data['nodes']:
        await service.create_node(
            label=node_data['label'],
            properties=node_data['properties'],
            merge=True
        )
    
    # Create relationships
    for rel_data in data['relationships']:
        await service.create_relationship(
            from_node_id=rel_data['from_id'],
            to_node_id=rel_data['to_id'],
            relationship_type=rel_data['type'],
            properties=rel_data['properties']
        )
    
    await service.disconnect()
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_neo4j_service.py

# Run with Neo4j test database
pytest --neo4j-uri=bolt://localhost:7688
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: Database interaction testing
- **Model Tests**: Pydantic model validation
- **Service Tests**: Neo4j service operations

## Performance Optimization

### Connection Pooling
```python
# Configure connection pool
settings.neo4j.max_connection_pool_size = 100
settings.neo4j.max_connection_lifetime = 60
```

### Batch Processing
```python
# Process large datasets efficiently
async def bulk_import(data_list, batch_size=1000):
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i + batch_size]
        await process_batch(batch)
```

### Query Optimization
```python
# Use indexes for better performance
query = """
CREATE INDEX person_email IF NOT EXISTS 
FOR (p:Person) ON (p.email)
"""

# Use MERGE for upsert operations
query = """
MERGE (p:Person {email: $email})
ON CREATE SET p.name = $name, p.created_at = datetime()
ON MATCH SET p.updated_at = datetime()
"""
```

## Monitoring and Analytics

### Database Statistics
```python
async def get_graph_stats():
    service = Neo4jService()
    await service.connect()
    
    stats = await service.get_database_stats()
    
    print(f"Total nodes: {stats['node_count']}")
    print(f"Total relationships: {stats['relationship_count']}")
    print(f"Node types: {stats['node_labels']}")
    print(f"Relationship types: {stats['relationship_types']}")
    
    # Node counts by type
    for label, count in stats['node_label_counts'].items():
        print(f"{label}: {count} nodes")
```

### Health Monitoring
```python
async def monitor_health():
    service = Neo4jService()
    health = await service.health_check()
    
    print(f"Status: {health['status']}")
    print(f"Node count: {health['details']['stats']['nodes']}")
    print(f"Relationship count: {health['details']['stats']['relationships']}")
```

## Graph Algorithms

### Community Detection
```python
# Find communities using Cypher
query = """
CALL gds.louvain.stream('myGraph')
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).name AS name, communityId
ORDER BY communityId, name
"""
```

### Centrality Analysis
```python
# PageRank centrality
query = """
CALL gds.pageRank.stream('myGraph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score
ORDER BY score DESC
LIMIT 10
"""
```

### Path Analysis
```python
# All shortest paths
paths = await service.execute_query("""
MATCH (start:Person {name: $start_name}), (end:Policy {name: $end_name})
CALL gds.shortestPath.dijkstra.stream('myGraph', {
    sourceNode: start,
    targetNode: end
})
YIELD path
RETURN path
""", {"start_name": "Alice", "end_name": "Data Policy"})
```

## Best Practices

### Data Modeling
1. **Normalize Relationships**: Use specific relationship types rather than generic ones
2. **Property Placement**: Store frequently queried properties on nodes
3. **Index Strategy**: Create indexes on properties used in WHERE clauses
4. **Constraint Usage**: Use unique constraints to prevent duplicates

### Query Performance
1. **Use Parameters**: Always use parameterized queries
2. **Limit Results**: Use LIMIT to prevent large result sets
3. **Profile Queries**: Use PROFILE to analyze query performance
4. **Avoid Cartesian Products**: Be careful with multiple MATCH clauses

### Security
1. **Authentication**: Use strong passwords and authentication
2. **Authorization**: Implement role-based access control
3. **Encryption**: Use encrypted connections in production
4. **Input Validation**: Validate all input data

## Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

### Production Configuration
```env
# Production settings
NEO4J_URI=neo4j+s://your-cluster.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-secure-password
NEO4J_ENCRYPTED=true
NEO4J_TRUST=TRUST_SYSTEM_CA_SIGNED_CERTIFICATES

# Performance settings
NEO4J_MAX_CONNECTION_POOL_SIZE=200
BATCH_SIZE=5000
```

## Troubleshooting

### Common Issues

**Connection Timeout**
```bash
Error: Connection acquisition timeout
Solution: Increase connection_acquisition_timeout or pool size
```

**Memory Issues**
```bash
Error: OutOfMemoryError
Solution: Increase Neo4j heap size or reduce batch size
```

**Constraint Violations**
```bash
Error: Node already exists
Solution: Use MERGE instead of CREATE or handle duplicates
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

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review Neo4j documentation
3. Create a GitHub issue
4. Join the Neo4j community

---

Built with ❤️ using Neo4j and modern Python practices.


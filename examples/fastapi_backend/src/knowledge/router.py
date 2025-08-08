"""
Knowledge Graph router with Neo4j integration patterns.

This module provides endpoints for managing knowledge graphs, entities,
relationships, and graph-based queries for context engineering.
"""

from datetime import datetime
from typing import Annotated, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Query
from loguru import logger
from pydantic import BaseModel, Field

from src.auth.router import get_current_active_user
from src.common.exceptions import (
    KnowledgeGraphException,
    EntityNotFoundException,
    RelationshipException,
    ValidationException
)

router = APIRouter()

# Knowledge Graph models
class Entity(BaseModel):
    """Knowledge graph entity model."""
    id: UUID = Field(default_factory=uuid4)
    type: str = Field(..., description="Entity type (person, policy, use_case, concept)")
    name: str = Field(..., description="Entity name")
    description: Optional[str] = Field(None, description="Entity description")
    properties: Dict = Field(default={}, description="Additional properties")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        from_attributes = True


class Relationship(BaseModel):
    """Knowledge graph relationship model."""
    id: UUID = Field(default_factory=uuid4)
    source_id: UUID = Field(..., description="Source entity ID")
    target_id: UUID = Field(..., description="Target entity ID")
    type: str = Field(..., description="Relationship type")
    properties: Dict = Field(default={}, description="Relationship properties")
    strength: float = Field(default=1.0, ge=0.0, le=1.0, description="Relationship strength")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        from_attributes = True


class GraphQuery(BaseModel):
    """Graph query model."""
    query_type: str = Field(..., description="Query type (cypher, path, neighbors)")
    parameters: Dict = Field(default={}, description="Query parameters")
    limit: int = Field(default=100, ge=1, le=1000, description="Result limit")


class GraphQueryResult(BaseModel):
    """Graph query result model."""
    entities: List[Entity] = Field(default=[], description="Matching entities")
    relationships: List[Relationship] = Field(default=[], description="Matching relationships")
    paths: List[List[UUID]] = Field(default=[], description="Graph paths")
    metadata: Dict = Field(default={}, description="Query metadata")


class EntityCreate(BaseModel):
    """Entity creation model."""
    type: str = Field(..., description="Entity type")
    name: str = Field(..., description="Entity name")
    description: Optional[str] = None
    properties: Dict = Field(default={})


class RelationshipCreate(BaseModel):
    """Relationship creation model."""
    source_id: UUID
    target_id: UUID
    type: str
    properties: Dict = Field(default={})
    strength: float = Field(default=1.0, ge=0.0, le=1.0)


# Mock data storage (in production, use Neo4j)
entities_db: Dict[UUID, Entity] = {}
relationships_db: Dict[UUID, Relationship] = {}

# Supported entity types
ENTITY_TYPES = ["person", "policy", "use_case", "concept", "document", "organization"]

# Supported relationship types
RELATIONSHIP_TYPES = [
    "related_to", "implements", "uses", "created_by", "part_of",
    "depends_on", "influences", "similar_to", "contains"
]


def validate_entity_type(entity_type: str) -> bool:
    """Validate entity type."""
    return entity_type in ENTITY_TYPES


def validate_relationship_type(rel_type: str) -> bool:
    """Validate relationship type."""
    return rel_type in RELATIONSHIP_TYPES


def find_entity_neighbors(entity_id: UUID, max_depth: int = 2) -> List[UUID]:
    """Find neighboring entities up to max_depth."""
    neighbors = set()
    current_level = {entity_id}
    
    for depth in range(max_depth):
        next_level = set()
        for entity in current_level:
            # Find connected entities
            for rel in relationships_db.values():
                if rel.source_id == entity and rel.target_id not in neighbors:
                    next_level.add(rel.target_id)
                elif rel.target_id == entity and rel.source_id not in neighbors:
                    next_level.add(rel.source_id)
        
        neighbors.update(next_level)
        current_level = next_level
        
        if not current_level:
            break
    
    return list(neighbors)


@router.post("/entities", response_model=Entity)
async def create_entity(
    entity_data: EntityCreate,
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> Entity:
    """
    Create a new knowledge graph entity.
    
    - **type**: Entity type (person, policy, use_case, concept, document, organization)
    - **name**: Entity name
    - **description**: Optional description
    - **properties**: Additional properties as key-value pairs
    """
    if not validate_entity_type(entity_data.type):
        raise ValidationException(f"Invalid entity type: {entity_data.type}")
    
    entity = Entity(
        type=entity_data.type,
        name=entity_data.name,
        description=entity_data.description,
        properties=entity_data.properties
    )
    
    entities_db[entity.id] = entity
    
    logger.info(f"Created entity {entity.id} of type {entity.type}")
    
    return entity


@router.get("/entities", response_model=List[Entity])
async def list_entities(
    current_user: Annotated[dict, Depends(get_current_active_user)],
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    search: Optional[str] = Query(None, description="Search in name and description"),
    limit: int = Query(100, ge=1, le=1000, description="Result limit")
) -> List[Entity]:
    """
    List knowledge graph entities with optional filtering.
    
    - **entity_type**: Filter by entity type
    - **search**: Search term for name and description
    - **limit**: Maximum number of results
    """
    entities = list(entities_db.values())
    
    # Filter by type
    if entity_type:
        if not validate_entity_type(entity_type):
            raise ValidationException(f"Invalid entity type: {entity_type}")
        entities = [e for e in entities if e.type == entity_type]
    
    # Search filter
    if search:
        search_lower = search.lower()
        entities = [
            e for e in entities
            if search_lower in e.name.lower() or 
               (e.description and search_lower in e.description.lower())
        ]
    
    return entities[:limit]


@router.get("/entities/{entity_id}", response_model=Entity)
async def get_entity(
    entity_id: UUID,
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> Entity:
    """Get a specific entity by ID."""
    entity = entities_db.get(entity_id)
    if not entity:
        raise EntityNotFoundException(str(entity_id))
    
    return entity


@router.put("/entities/{entity_id}", response_model=Entity)
async def update_entity(
    entity_id: UUID,
    entity_data: EntityCreate,
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> Entity:
    """Update an existing entity."""
    entity = entities_db.get(entity_id)
    if not entity:
        raise EntityNotFoundException(str(entity_id))
    
    if not validate_entity_type(entity_data.type):
        raise ValidationException(f"Invalid entity type: {entity_data.type}")
    
    entity.type = entity_data.type
    entity.name = entity_data.name
    entity.description = entity_data.description
    entity.properties = entity_data.properties
    entity.updated_at = datetime.utcnow()
    
    logger.info(f"Updated entity {entity_id}")
    
    return entity


@router.delete("/entities/{entity_id}")
async def delete_entity(
    entity_id: UUID,
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> dict:
    """Delete an entity and all its relationships."""
    entity = entities_db.get(entity_id)
    if not entity:
        raise EntityNotFoundException(str(entity_id))
    
    # Remove entity
    del entities_db[entity_id]
    
    # Remove all relationships involving this entity
    relationships_to_remove = [
        rel_id for rel_id, rel in relationships_db.items()
        if rel.source_id == entity_id or rel.target_id == entity_id
    ]
    
    for rel_id in relationships_to_remove:
        del relationships_db[rel_id]
    
    logger.info(f"Deleted entity {entity_id} and {len(relationships_to_remove)} relationships")
    
    return {"message": f"Entity {entity_id} deleted successfully"}


@router.post("/relationships", response_model=Relationship)
async def create_relationship(
    rel_data: RelationshipCreate,
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> Relationship:
    """
    Create a relationship between two entities.
    
    - **source_id**: Source entity ID
    - **target_id**: Target entity ID
    - **type**: Relationship type
    - **properties**: Additional properties
    - **strength**: Relationship strength (0.0 to 1.0)
    """
    # Validate entities exist
    if rel_data.source_id not in entities_db:
        raise EntityNotFoundException(str(rel_data.source_id))
    if rel_data.target_id not in entities_db:
        raise EntityNotFoundException(str(rel_data.target_id))
    
    if not validate_relationship_type(rel_data.type):
        raise ValidationException(f"Invalid relationship type: {rel_data.type}")
    
    if rel_data.source_id == rel_data.target_id:
        raise RelationshipException("Cannot create relationship to self")
    
    relationship = Relationship(
        source_id=rel_data.source_id,
        target_id=rel_data.target_id,
        type=rel_data.type,
        properties=rel_data.properties,
        strength=rel_data.strength
    )
    
    relationships_db[relationship.id] = relationship
    
    logger.info(f"Created relationship {relationship.id} of type {relationship.type}")
    
    return relationship


@router.get("/relationships", response_model=List[Relationship])
async def list_relationships(
    current_user: Annotated[dict, Depends(get_current_active_user)],
    entity_id: Optional[UUID] = Query(None, description="Filter by entity ID"),
    rel_type: Optional[str] = Query(None, description="Filter by relationship type"),
    limit: int = Query(100, ge=1, le=1000, description="Result limit")
) -> List[Relationship]:
    """
    List relationships with optional filtering.
    
    - **entity_id**: Filter relationships involving this entity
    - **rel_type**: Filter by relationship type
    - **limit**: Maximum number of results
    """
    relationships = list(relationships_db.values())
    
    # Filter by entity
    if entity_id:
        relationships = [
            r for r in relationships
            if r.source_id == entity_id or r.target_id == entity_id
        ]
    
    # Filter by type
    if rel_type:
        if not validate_relationship_type(rel_type):
            raise ValidationException(f"Invalid relationship type: {rel_type}")
        relationships = [r for r in relationships if r.type == rel_type]
    
    return relationships[:limit]


@router.get("/entities/{entity_id}/neighbors", response_model=List[Entity])
async def get_entity_neighbors(
    entity_id: UUID,
    current_user: Annotated[dict, Depends(get_current_active_user)],
    max_depth: int = Query(2, ge=1, le=5, description="Maximum traversal depth")
) -> List[Entity]:
    """
    Get neighboring entities up to specified depth.
    
    - **max_depth**: Maximum graph traversal depth (1-5)
    """
    if entity_id not in entities_db:
        raise EntityNotFoundException(str(entity_id))
    
    neighbor_ids = find_entity_neighbors(entity_id, max_depth)
    neighbors = [entities_db[nid] for nid in neighbor_ids if nid in entities_db]
    
    return neighbors


@router.post("/query", response_model=GraphQueryResult)
async def query_graph(
    query: GraphQuery,
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> GraphQueryResult:
    """
    Execute a graph query.
    
    - **query_type**: Type of query (cypher, path, neighbors)
    - **parameters**: Query parameters
    - **limit**: Result limit
    """
    try:
        if query.query_type == "neighbors":
            entity_id = UUID(query.parameters.get("entity_id"))
            max_depth = query.parameters.get("max_depth", 2)
            
            neighbor_ids = find_entity_neighbors(entity_id, max_depth)
            entities = [entities_db[nid] for nid in neighbor_ids if nid in entities_db]
            
            return GraphQueryResult(
                entities=entities[:query.limit],
                metadata={"query_type": "neighbors", "depth": max_depth}
            )
        
        elif query.query_type == "path":
            source_id = UUID(query.parameters.get("source_id"))
            target_id = UUID(query.parameters.get("target_id"))
            
            # Simple path finding (in production, use proper graph algorithms)
            paths = []  # Placeholder for path finding logic
            
            return GraphQueryResult(
                paths=paths,
                metadata={"query_type": "path", "source": str(source_id), "target": str(target_id)}
            )
        
        else:
            raise ValidationException(f"Unsupported query type: {query.query_type}")
    
    except Exception as e:
        logger.error(f"Graph query error: {e}")
        raise KnowledgeGraphException(f"Query execution failed: {str(e)}")


@router.get("/schema")
async def get_graph_schema(
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> dict:
    """
    Get knowledge graph schema information.
    
    Returns supported entity types, relationship types, and statistics.
    """
    entity_type_counts = {}
    for entity in entities_db.values():
        entity_type_counts[entity.type] = entity_type_counts.get(entity.type, 0) + 1
    
    relationship_type_counts = {}
    for rel in relationships_db.values():
        relationship_type_counts[rel.type] = relationship_type_counts.get(rel.type, 0) + 1
    
    return {
        "supported_entity_types": ENTITY_TYPES,
        "supported_relationship_types": RELATIONSHIP_TYPES,
        "statistics": {
            "total_entities": len(entities_db),
            "total_relationships": len(relationships_db),
            "entity_type_distribution": entity_type_counts,
            "relationship_type_distribution": relationship_type_counts
        }
    }


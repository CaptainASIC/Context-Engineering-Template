"""
Graph Models for Knowledge Graph entities.

This module defines Pydantic models for different types of nodes
and relationships in the knowledge graph, including validation
and serialization logic.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from uuid import uuid4, UUID
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator


class NodeType(str, Enum):
    """Enumeration of supported node types."""
    PERSON = "Person"
    POLICY = "Policy"
    USE_CASE = "UseCase"
    DOCUMENT = "Document"
    ORGANIZATION = "Organization"
    CONCEPT = "Concept"


class RelationshipType(str, Enum):
    """Enumeration of supported relationship types."""
    WORKS_FOR = "WORKS_FOR"
    MANAGES = "MANAGES"
    IMPLEMENTS = "IMPLEMENTS"
    RELATES_TO = "RELATES_TO"
    AUTHORED = "AUTHORED"
    REFERENCES = "REFERENCES"
    PART_OF = "PART_OF"
    DEPENDS_ON = "DEPENDS_ON"
    SIMILAR_TO = "SIMILAR_TO"


class BaseGraphNode(BaseModel):
    """Base class for all graph nodes."""
    
    id: Optional[int] = Field(None, description="Neo4j node ID")
    uuid: UUID = Field(default_factory=uuid4, description="Unique identifier")
    name: str = Field(..., description="Node name")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j properties dictionary."""
        props = self.dict(exclude={"id"})
        
        # Convert datetime objects to ISO strings
        for key, value in props.items():
            if isinstance(value, datetime):
                props[key] = value.isoformat()
            elif isinstance(value, UUID):
                props[key] = str(value)
            elif isinstance(value, list):
                # Neo4j handles lists natively
                pass
        
        return props
    
    @classmethod
    def from_neo4j_node(cls, node_data: Dict[str, Any], node_id: int):
        """Create instance from Neo4j node data."""
        data = dict(node_data)
        data["id"] = node_id
        
        # Convert ISO strings back to datetime
        for field_name, field_info in cls.__fields__.items():
            if field_info.type_ == datetime and field_name in data:
                if isinstance(data[field_name], str):
                    data[field_name] = datetime.fromisoformat(data[field_name])
        
        return cls(**data)


class PersonNode(BaseGraphNode):
    """Person node model."""
    
    email: Optional[str] = Field(None, description="Email address")
    title: Optional[str] = Field(None, description="Job title")
    department: Optional[str] = Field(None, description="Department")
    skills: List[str] = Field(default_factory=list, description="Skills and expertise")
    phone: Optional[str] = Field(None, description="Phone number")
    location: Optional[str] = Field(None, description="Location")
    bio: Optional[str] = Field(None, description="Biography")
    
    @validator('email')
    def validate_email(cls, v):
        if v and '@' not in v:
            raise ValueError('Invalid email format')
        return v
    
    @validator('skills')
    def validate_skills(cls, v):
        return [skill.strip() for skill in v if skill.strip()]
    
    def get_node_type(self) -> str:
        return NodeType.PERSON.value


class PolicyNode(BaseGraphNode):
    """Policy node model."""
    
    description: str = Field(..., description="Policy description")
    version: str = Field(default="1.0", description="Policy version")
    status: str = Field(default="draft", description="Policy status")
    category: Optional[str] = Field(None, description="Policy category")
    tags: List[str] = Field(default_factory=list, description="Policy tags")
    effective_date: Optional[datetime] = Field(None, description="Effective date")
    expiry_date: Optional[datetime] = Field(None, description="Expiry date")
    approval_required: bool = Field(default=True, description="Requires approval")
    
    @validator('status')
    def validate_status(cls, v):
        allowed_statuses = ['draft', 'review', 'approved', 'active', 'deprecated']
        if v.lower() not in allowed_statuses:
            raise ValueError(f'Status must be one of: {allowed_statuses}')
        return v.lower()
    
    @validator('version')
    def validate_version(cls, v):
        if not v or not v.strip():
            raise ValueError('Version cannot be empty')
        return v.strip()
    
    def get_node_type(self) -> str:
        return NodeType.POLICY.value
    
    def is_active(self) -> bool:
        """Check if policy is currently active."""
        now = datetime.now(timezone.utc)
        
        if self.status != 'active':
            return False
        
        if self.effective_date and self.effective_date > now:
            return False
        
        if self.expiry_date and self.expiry_date < now:
            return False
        
        return True


class UseCaseNode(BaseGraphNode):
    """Use case node model."""
    
    description: str = Field(..., description="Use case description")
    priority: str = Field(default="medium", description="Priority level")
    status: str = Field(default="proposed", description="Implementation status")
    business_value: Optional[str] = Field(None, description="Business value description")
    complexity: str = Field(default="medium", description="Implementation complexity")
    tags: List[str] = Field(default_factory=list, description="Use case tags")
    estimated_effort: Optional[str] = Field(None, description="Estimated effort")
    success_criteria: List[str] = Field(default_factory=list, description="Success criteria")
    
    @validator('priority')
    def validate_priority(cls, v):
        allowed_priorities = ['low', 'medium', 'high', 'critical']
        if v.lower() not in allowed_priorities:
            raise ValueError(f'Priority must be one of: {allowed_priorities}')
        return v.lower()
    
    @validator('status')
    def validate_status(cls, v):
        allowed_statuses = ['proposed', 'approved', 'in_progress', 'completed', 'cancelled']
        if v.lower() not in allowed_statuses:
            raise ValueError(f'Status must be one of: {allowed_statuses}')
        return v.lower()
    
    @validator('complexity')
    def validate_complexity(cls, v):
        allowed_complexity = ['low', 'medium', 'high', 'very_high']
        if v.lower() not in allowed_complexity:
            raise ValueError(f'Complexity must be one of: {allowed_complexity}')
        return v.lower()
    
    def get_node_type(self) -> str:
        return NodeType.USE_CASE.value


class DocumentNode(BaseGraphNode):
    """Document node model."""
    
    title: str = Field(..., description="Document title")
    content: Optional[str] = Field(None, description="Document content")
    document_type: str = Field(..., description="Document type")
    source: Optional[str] = Field(None, description="Document source")
    author: Optional[str] = Field(None, description="Document author")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    file_path: Optional[str] = Field(None, description="File path")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    checksum: Optional[str] = Field(None, description="File checksum")
    
    @validator('document_type')
    def validate_document_type(cls, v):
        allowed_types = ['pdf', 'docx', 'txt', 'md', 'html', 'json', 'xml']
        if v.lower() not in allowed_types:
            raise ValueError(f'Document type must be one of: {allowed_types}')
        return v.lower()
    
    def get_node_type(self) -> str:
        return NodeType.DOCUMENT.value


class OrganizationNode(BaseGraphNode):
    """Organization node model."""
    
    organization_type: str = Field(..., description="Organization type")
    industry: Optional[str] = Field(None, description="Industry")
    size: Optional[str] = Field(None, description="Organization size")
    location: Optional[str] = Field(None, description="Primary location")
    website: Optional[str] = Field(None, description="Website URL")
    description: Optional[str] = Field(None, description="Organization description")
    
    @validator('organization_type')
    def validate_org_type(cls, v):
        allowed_types = ['company', 'department', 'team', 'division', 'subsidiary']
        if v.lower() not in allowed_types:
            raise ValueError(f'Organization type must be one of: {allowed_types}')
        return v.lower()
    
    @validator('size')
    def validate_size(cls, v):
        if v:
            allowed_sizes = ['startup', 'small', 'medium', 'large', 'enterprise']
            if v.lower() not in allowed_sizes:
                raise ValueError(f'Size must be one of: {allowed_sizes}')
            return v.lower()
        return v
    
    def get_node_type(self) -> str:
        return NodeType.ORGANIZATION.value


class ConceptNode(BaseGraphNode):
    """Concept node model."""
    
    definition: str = Field(..., description="Concept definition")
    category: Optional[str] = Field(None, description="Concept category")
    synonyms: List[str] = Field(default_factory=list, description="Synonyms")
    related_terms: List[str] = Field(default_factory=list, description="Related terms")
    domain: Optional[str] = Field(None, description="Domain or field")
    complexity_level: str = Field(default="intermediate", description="Complexity level")
    
    @validator('complexity_level')
    def validate_complexity(cls, v):
        allowed_levels = ['basic', 'intermediate', 'advanced', 'expert']
        if v.lower() not in allowed_levels:
            raise ValueError(f'Complexity level must be one of: {allowed_levels}')
        return v.lower()
    
    def get_node_type(self) -> str:
        return NodeType.CONCEPT.value


class BaseRelationship(BaseModel):
    """Base class for all relationships."""
    
    id: Optional[int] = Field(None, description="Neo4j relationship ID")
    from_node_id: int = Field(..., description="Source node ID")
    to_node_id: int = Field(..., description="Target node ID")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j properties dictionary."""
        props = self.dict(exclude={"id", "from_node_id", "to_node_id"})
        
        # Convert datetime objects to ISO strings
        for key, value in props.items():
            if isinstance(value, datetime):
                props[key] = value.isoformat()
        
        return props
    
    @classmethod
    def from_neo4j_relationship(cls, rel_data: Dict[str, Any], rel_id: int, from_id: int, to_id: int):
        """Create instance from Neo4j relationship data."""
        data = dict(rel_data)
        data.update({
            "id": rel_id,
            "from_node_id": from_id,
            "to_node_id": to_id
        })
        
        # Convert ISO strings back to datetime
        for field_name, field_info in cls.__fields__.items():
            if field_info.type_ == datetime and field_name in data:
                if isinstance(data[field_name], str):
                    data[field_name] = datetime.fromisoformat(data[field_name])
        
        return cls(**data)


class WorksForRelationship(BaseRelationship):
    """Works for relationship between Person and Organization."""
    
    role: Optional[str] = Field(None, description="Role in organization")
    start_date: Optional[datetime] = Field(None, description="Start date")
    end_date: Optional[datetime] = Field(None, description="End date")
    is_current: bool = Field(default=True, description="Is current employment")
    
    def get_relationship_type(self) -> str:
        return RelationshipType.WORKS_FOR.value


class ManagesRelationship(BaseRelationship):
    """Management relationship between Persons."""
    
    start_date: Optional[datetime] = Field(None, description="Management start date")
    end_date: Optional[datetime] = Field(None, description="Management end date")
    is_current: bool = Field(default=True, description="Is current management")
    management_type: str = Field(default="direct", description="Type of management")
    
    @validator('management_type')
    def validate_management_type(cls, v):
        allowed_types = ['direct', 'indirect', 'matrix', 'project']
        if v.lower() not in allowed_types:
            raise ValueError(f'Management type must be one of: {allowed_types}')
        return v.lower()
    
    def get_relationship_type(self) -> str:
        return RelationshipType.MANAGES.value


class ImplementsRelationship(BaseRelationship):
    """Implementation relationship between Person/Organization and UseCase/Policy."""
    
    implementation_date: Optional[datetime] = Field(None, description="Implementation date")
    status: str = Field(default="planned", description="Implementation status")
    notes: Optional[str] = Field(None, description="Implementation notes")
    completion_percentage: float = Field(default=0.0, ge=0.0, le=100.0, description="Completion percentage")
    
    @validator('status')
    def validate_status(cls, v):
        allowed_statuses = ['planned', 'in_progress', 'completed', 'on_hold', 'cancelled']
        if v.lower() not in allowed_statuses:
            raise ValueError(f'Status must be one of: {allowed_statuses}')
        return v.lower()
    
    def get_relationship_type(self) -> str:
        return RelationshipType.IMPLEMENTS.value


class RelatesToRelationship(BaseRelationship):
    """Generic relationship between any two nodes."""
    
    relationship_type: str = Field(..., description="Type of relationship")
    strength: float = Field(default=1.0, ge=0.0, le=1.0, description="Relationship strength")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in relationship")
    source: Optional[str] = Field(None, description="Source of relationship information")
    bidirectional: bool = Field(default=False, description="Is relationship bidirectional")
    
    def get_relationship_type(self) -> str:
        return RelationshipType.RELATES_TO.value


class AuthoredRelationship(BaseRelationship):
    """Authorship relationship between Person and Document."""
    
    authored_date: Optional[datetime] = Field(None, description="Authoring date")
    role: str = Field(default="author", description="Authoring role")
    contribution_percentage: float = Field(default=100.0, ge=0.0, le=100.0, description="Contribution percentage")
    
    @validator('role')
    def validate_role(cls, v):
        allowed_roles = ['author', 'co-author', 'editor', 'reviewer', 'contributor']
        if v.lower() not in allowed_roles:
            raise ValueError(f'Role must be one of: {allowed_roles}')
        return v.lower()
    
    def get_relationship_type(self) -> str:
        return RelationshipType.AUTHORED.value


# Node type mapping for factory pattern
NODE_TYPE_MAPPING = {
    NodeType.PERSON: PersonNode,
    NodeType.POLICY: PolicyNode,
    NodeType.USE_CASE: UseCaseNode,
    NodeType.DOCUMENT: DocumentNode,
    NodeType.ORGANIZATION: OrganizationNode,
    NodeType.CONCEPT: ConceptNode
}

# Relationship type mapping
RELATIONSHIP_TYPE_MAPPING = {
    RelationshipType.WORKS_FOR: WorksForRelationship,
    RelationshipType.MANAGES: ManagesRelationship,
    RelationshipType.IMPLEMENTS: ImplementsRelationship,
    RelationshipType.RELATES_TO: RelatesToRelationship,
    RelationshipType.AUTHORED: AuthoredRelationship
}


def create_node_from_type(node_type: str, **kwargs) -> BaseGraphNode:
    """
    Factory function to create node instances.
    
    Args:
        node_type: Type of node to create
        **kwargs: Node properties
        
    Returns:
        Node instance
    """
    node_class = NODE_TYPE_MAPPING.get(NodeType(node_type))
    if not node_class:
        raise ValueError(f"Unknown node type: {node_type}")
    
    return node_class(**kwargs)


def create_relationship_from_type(rel_type: str, **kwargs) -> BaseRelationship:
    """
    Factory function to create relationship instances.
    
    Args:
        rel_type: Type of relationship to create
        **kwargs: Relationship properties
        
    Returns:
        Relationship instance
    """
    rel_class = RELATIONSHIP_TYPE_MAPPING.get(RelationshipType(rel_type))
    if not rel_class:
        raise ValueError(f"Unknown relationship type: {rel_type}")
    
    return rel_class(**kwargs)


class GraphQueryResult(BaseModel):
    """Result of a graph query operation."""
    
    nodes: List[Dict[str, Any]] = Field(default_factory=list, description="Result nodes")
    relationships: List[Dict[str, Any]] = Field(default_factory=list, description="Result relationships")
    paths: List[Dict[str, Any]] = Field(default_factory=list, description="Result paths")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Query metadata")
    execution_time: Optional[float] = Field(None, description="Query execution time in seconds")
    
    def get_node_count(self) -> int:
        """Get number of nodes in result."""
        return len(self.nodes)
    
    def get_relationship_count(self) -> int:
        """Get number of relationships in result."""
        return len(self.relationships)
    
    def get_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        """Get nodes filtered by type."""
        return [node for node in self.nodes if node_type in node.get("labels", [])]
    
    def get_relationships_by_type(self, rel_type: str) -> List[Dict[str, Any]]:
        """Get relationships filtered by type."""
        return [rel for rel in self.relationships if rel.get("type") == rel_type]


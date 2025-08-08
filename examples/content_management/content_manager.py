"""
LIXIL AI Hub Content Management System

This module provides comprehensive content management capabilities for the
LIXIL AI Hub Platform, including policy upload, versioning, permanent statement
management, and content lifecycle operations.

Key Features:
- Policy document upload and validation
- Version control with change tracking
- Permanent statement/FAQ management
- Content approval workflows
- Automated content validation
- Integration with vector database and knowledge graphs

Author: LIXIL AI Hub Platform Team
Version: 1.0.0
"""

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path

import asyncpg
import aiofiles
from pydantic import BaseModel, Field, validator
import neo4j
from neo4j import AsyncGraphDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentType(str, Enum):
    """Types of content managed by the system."""
    POLICY_DOCUMENT = "policy_document"
    PERMANENT_STATEMENT = "permanent_statement"
    FAQ_ITEM = "faq_item"
    TRAINING_MATERIAL = "training_material"
    USE_CASE = "use_case"


class ContentStatus(str, Enum):
    """Status of content in the management system."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    REJECTED = "rejected"


class ChangeType(str, Enum):
    """Types of changes for version tracking."""
    CREATED = "created"
    UPDATED = "updated"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    DELETED = "deleted"


@dataclass
class ContentMetadata:
    """Metadata for managed content."""
    content_id: str
    title: str
    content_type: ContentType
    status: ContentStatus
    version: str
    language: str
    created_at: datetime
    updated_at: datetime
    created_by: str
    updated_by: str
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    description: Optional[str] = None
    priority: int = 0
    expires_at: Optional[datetime] = None


@dataclass
class ContentVersion:
    """Version information for content tracking."""
    version_id: str
    content_id: str
    version_number: str
    change_type: ChangeType
    changed_by: str
    changed_at: datetime
    change_summary: str
    content_hash: str
    previous_version: Optional[str] = None


class ContentUploadRequest(BaseModel):
    """Request model for content upload."""
    title: str = Field(..., min_length=1, max_length=200)
    content_type: ContentType
    content: str = Field(..., min_length=1)
    language: str = Field(default="en", regex="^[a-z]{2}(-[A-Z]{2})?$")
    category: Optional[str] = Field(None, max_length=100)
    tags: List[str] = Field(default_factory=list)
    description: Optional[str] = Field(None, max_length=500)
    priority: int = Field(default=0, ge=0, le=10)
    expires_at: Optional[datetime] = None
    created_by: str = Field(..., min_length=1, max_length=100)

    @validator('tags')
    def validate_tags(cls, v):
        """Validate tags format and length."""
        if len(v) > 10:
            raise ValueError("Maximum 10 tags allowed")
        for tag in v:
            if len(tag) > 50:
                raise ValueError("Tag length must be 50 characters or less")
        return v

    @validator('expires_at')
    def validate_expiration(cls, v):
        """Validate expiration date is in the future."""
        if v and v <= datetime.now():
            raise ValueError("Expiration date must be in the future")
        return v


class ContentUpdateRequest(BaseModel):
    """Request model for content updates."""
    content_id: str
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    content: Optional[str] = Field(None, min_length=1)
    category: Optional[str] = Field(None, max_length=100)
    tags: Optional[List[str]] = None
    description: Optional[str] = Field(None, max_length=500)
    priority: Optional[int] = Field(None, ge=0, le=10)
    expires_at: Optional[datetime] = None
    updated_by: str = Field(..., min_length=1, max_length=100)
    change_summary: str = Field(..., min_length=1, max_length=200)


class PermanentStatementRequest(BaseModel):
    """Request model for permanent statements/FAQs."""
    question: str = Field(..., min_length=1, max_length=500)
    answer: str = Field(..., min_length=1, max_length=2000)
    category: str = Field(..., min_length=1, max_length=100)
    language: str = Field(default="en", regex="^[a-z]{2}(-[A-Z]{2})?$")
    tags: List[str] = Field(default_factory=list)
    priority: int = Field(default=5, ge=1, le=10)
    created_by: str = Field(..., min_length=1, max_length=100)


class ContentSearchRequest(BaseModel):
    """Request model for content search."""
    query: Optional[str] = None
    content_type: Optional[ContentType] = None
    status: Optional[ContentStatus] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    language: Optional[str] = None
    created_by: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = Field(default=50, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class ContentValidationResult(BaseModel):
    """Result of content validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    quality_score: float = Field(ge=0.0, le=1.0)


class DatabaseManager:
    """
    Database manager for content storage and retrieval.
    
    Handles connections to both PostgreSQL (for content storage) and
    Neo4j (for relationship management) databases.
    """

    def __init__(self, postgres_url: str, neo4j_url: str, neo4j_auth: Tuple[str, str]):
        """
        Initialize database connections.
        
        Args:
            postgres_url: PostgreSQL connection URL
            neo4j_url: Neo4j connection URL
            neo4j_auth: Neo4j authentication tuple (username, password)
        """
        self.postgres_url = postgres_url
        self.neo4j_url = neo4j_url
        self.neo4j_auth = neo4j_auth
        self.pg_pool = None
        self.neo4j_driver = None

    async def initialize(self):
        """Initialize database connections."""
        # Initialize PostgreSQL connection pool
        self.pg_pool = await asyncpg.create_pool(
            self.postgres_url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        
        # Initialize Neo4j driver
        self.neo4j_driver = AsyncGraphDatabase.driver(
            self.neo4j_url,
            auth=self.neo4j_auth
        )
        
        # Create database schema if needed
        await self._create_schema()
        
        logger.info("Database connections initialized")

    async def close(self):
        """Close database connections."""
        if self.pg_pool:
            await self.pg_pool.close()
        if self.neo4j_driver:
            await self.neo4j_driver.close()

    async def _create_schema(self):
        """Create database schema for content management."""
        schema_sql = """
        -- Content table
        CREATE TABLE IF NOT EXISTS content (
            content_id VARCHAR(64) PRIMARY KEY,
            title VARCHAR(200) NOT NULL,
            content_type VARCHAR(50) NOT NULL,
            status VARCHAR(50) NOT NULL,
            version VARCHAR(20) NOT NULL,
            language VARCHAR(10) NOT NULL,
            content_text TEXT NOT NULL,
            content_hash VARCHAR(64) NOT NULL,
            category VARCHAR(100),
            tags JSONB DEFAULT '[]',
            description TEXT,
            priority INTEGER DEFAULT 0,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
            created_by VARCHAR(100) NOT NULL,
            updated_by VARCHAR(100) NOT NULL,
            expires_at TIMESTAMP WITH TIME ZONE,
            metadata JSONB DEFAULT '{}'
        );

        -- Content versions table
        CREATE TABLE IF NOT EXISTS content_versions (
            version_id VARCHAR(64) PRIMARY KEY,
            content_id VARCHAR(64) NOT NULL,
            version_number VARCHAR(20) NOT NULL,
            change_type VARCHAR(50) NOT NULL,
            changed_by VARCHAR(100) NOT NULL,
            changed_at TIMESTAMP WITH TIME ZONE NOT NULL,
            change_summary TEXT NOT NULL,
            content_hash VARCHAR(64) NOT NULL,
            previous_version VARCHAR(64),
            content_snapshot JSONB NOT NULL,
            FOREIGN KEY (content_id) REFERENCES content(content_id)
        );

        -- Permanent statements table
        CREATE TABLE IF NOT EXISTS permanent_statements (
            statement_id VARCHAR(64) PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            category VARCHAR(100) NOT NULL,
            language VARCHAR(10) NOT NULL,
            tags JSONB DEFAULT '[]',
            priority INTEGER DEFAULT 5,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL,
            created_by VARCHAR(100) NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
            updated_by VARCHAR(100) NOT NULL,
            is_active BOOLEAN DEFAULT TRUE
        );

        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_content_type_status ON content(content_type, status);
        CREATE INDEX IF NOT EXISTS idx_content_category ON content(category);
        CREATE INDEX IF NOT EXISTS idx_content_created_at ON content(created_at);
        CREATE INDEX IF NOT EXISTS idx_content_tags ON content USING GIN(tags);
        CREATE INDEX IF NOT EXISTS idx_versions_content_id ON content_versions(content_id);
        CREATE INDEX IF NOT EXISTS idx_statements_category ON permanent_statements(category);
        CREATE INDEX IF NOT EXISTS idx_statements_language ON permanent_statements(language);
        """
        
        async with self.pg_pool.acquire() as conn:
            await conn.execute(schema_sql)

    async def store_content(self, metadata: ContentMetadata, content: str) -> str:
        """Store content in the database."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        async with self.pg_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO content (
                    content_id, title, content_type, status, version, language,
                    content_text, content_hash, category, tags, description, priority,
                    created_at, updated_at, created_by, updated_by, expires_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
            """, 
                metadata.content_id, metadata.title, metadata.content_type.value,
                metadata.status.value, metadata.version, metadata.language,
                content, content_hash, metadata.category, json.dumps(metadata.tags),
                metadata.description, metadata.priority, metadata.created_at,
                metadata.updated_at, metadata.created_by, metadata.updated_by,
                metadata.expires_at
            )
        
        return content_hash

    async def get_content(self, content_id: str) -> Optional[Tuple[ContentMetadata, str]]:
        """Retrieve content by ID."""
        async with self.pg_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM content WHERE content_id = $1
            """, content_id)
            
            if not row:
                return None
            
            metadata = ContentMetadata(
                content_id=row['content_id'],
                title=row['title'],
                content_type=ContentType(row['content_type']),
                status=ContentStatus(row['status']),
                version=row['version'],
                language=row['language'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                created_by=row['created_by'],
                updated_by=row['updated_by'],
                category=row['category'],
                tags=json.loads(row['tags']) if row['tags'] else [],
                description=row['description'],
                priority=row['priority'],
                expires_at=row['expires_at']
            )
            
            return metadata, row['content_text']

    async def search_content(self, search_request: ContentSearchRequest) -> List[ContentMetadata]:
        """Search content based on criteria."""
        conditions = []
        params = []
        param_count = 0
        
        if search_request.query:
            param_count += 1
            conditions.append(f"(title ILIKE ${param_count} OR content_text ILIKE ${param_count})")
            params.append(f"%{search_request.query}%")
        
        if search_request.content_type:
            param_count += 1
            conditions.append(f"content_type = ${param_count}")
            params.append(search_request.content_type.value)
        
        if search_request.status:
            param_count += 1
            conditions.append(f"status = ${param_count}")
            params.append(search_request.status.value)
        
        if search_request.category:
            param_count += 1
            conditions.append(f"category = ${param_count}")
            params.append(search_request.category)
        
        if search_request.language:
            param_count += 1
            conditions.append(f"language = ${param_count}")
            params.append(search_request.language)
        
        if search_request.created_by:
            param_count += 1
            conditions.append(f"created_by = ${param_count}")
            params.append(search_request.created_by)
        
        if search_request.date_from:
            param_count += 1
            conditions.append(f"created_at >= ${param_count}")
            params.append(search_request.date_from)
        
        if search_request.date_to:
            param_count += 1
            conditions.append(f"created_at <= ${param_count}")
            params.append(search_request.date_to)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        param_count += 1
        limit_param = f"${param_count}"
        params.append(search_request.limit)
        
        param_count += 1
        offset_param = f"${param_count}"
        params.append(search_request.offset)
        
        query = f"""
            SELECT content_id, title, content_type, status, version, language,
                   category, tags, description, priority, created_at, updated_at,
                   created_by, updated_by, expires_at
            FROM content
            {where_clause}
            ORDER BY updated_at DESC
            LIMIT {limit_param} OFFSET {offset_param}
        """
        
        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            
            results = []
            for row in rows:
                metadata = ContentMetadata(
                    content_id=row['content_id'],
                    title=row['title'],
                    content_type=ContentType(row['content_type']),
                    status=ContentStatus(row['status']),
                    version=row['version'],
                    language=row['language'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    created_by=row['created_by'],
                    updated_by=row['updated_by'],
                    category=row['category'],
                    tags=json.loads(row['tags']) if row['tags'] else [],
                    description=row['description'],
                    priority=row['priority'],
                    expires_at=row['expires_at']
                )
                results.append(metadata)
            
            return results


class ContentValidator:
    """
    Content validation system for quality assurance.
    
    Provides automated validation of content quality, format,
    and compliance with LIXIL standards.
    """

    def __init__(self):
        """Initialize the content validator."""
        self.min_content_length = 50
        self.max_content_length = 50000
        self.required_sections = ["introduction", "overview", "purpose"]
        self.forbidden_words = ["confidential", "internal only", "draft"]

    async def validate_content(self, content: str, content_type: ContentType) -> ContentValidationResult:
        """
        Validate content quality and compliance.
        
        Args:
            content: Content text to validate
            content_type: Type of content being validated
            
        Returns:
            ContentValidationResult with validation details
        """
        errors = []
        warnings = []
        suggestions = []
        quality_score = 1.0

        # Basic length validation
        if len(content) < self.min_content_length:
            errors.append(f"Content too short. Minimum {self.min_content_length} characters required.")
            quality_score -= 0.3
        
        if len(content) > self.max_content_length:
            errors.append(f"Content too long. Maximum {self.max_content_length} characters allowed.")
            quality_score -= 0.2

        # Content structure validation
        if content_type == ContentType.POLICY_DOCUMENT:
            structure_score = await self._validate_policy_structure(content)
            quality_score *= structure_score
            
            if structure_score < 0.8:
                warnings.append("Policy document may be missing standard sections.")
                suggestions.append("Consider adding sections: Introduction, Scope, Requirements, Compliance.")

        # Language and readability checks
        readability_score = await self._check_readability(content)
        quality_score *= readability_score
        
        if readability_score < 0.7:
            warnings.append("Content may be difficult to read.")
            suggestions.append("Consider shorter sentences and simpler language.")

        # Forbidden content check
        forbidden_found = [word for word in self.forbidden_words if word.lower() in content.lower()]
        if forbidden_found:
            errors.append(f"Content contains forbidden words: {', '.join(forbidden_found)}")
            quality_score -= 0.4

        # Ensure quality score is within bounds
        quality_score = max(0.0, min(1.0, quality_score))

        return ContentValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            quality_score=quality_score
        )

    async def _validate_policy_structure(self, content: str) -> float:
        """Validate policy document structure."""
        content_lower = content.lower()
        sections_found = 0
        
        # Check for common policy sections
        policy_sections = [
            "introduction", "purpose", "scope", "policy", "procedure",
            "responsibility", "compliance", "enforcement", "review"
        ]
        
        for section in policy_sections:
            if section in content_lower:
                sections_found += 1
        
        # Score based on section coverage
        return min(1.0, sections_found / 5.0)  # Expect at least 5 sections for full score

    async def _check_readability(self, content: str) -> float:
        """Check content readability (simplified implementation)."""
        sentences = content.split('.')
        words = content.split()
        
        if not sentences or not words:
            return 0.5
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Simple readability score based on sentence length
        if avg_sentence_length <= 15:
            return 1.0
        elif avg_sentence_length <= 25:
            return 0.8
        elif avg_sentence_length <= 35:
            return 0.6
        else:
            return 0.4


class ContentManager:
    """
    Main content management system for the LIXIL AI Hub Platform.
    
    Orchestrates content upload, validation, versioning, and lifecycle management.
    """

    def __init__(self, db_manager: DatabaseManager):
        """Initialize the content manager."""
        self.db_manager = db_manager
        self.validator = ContentValidator()

    async def upload_content(self, request: ContentUploadRequest) -> Tuple[str, ContentValidationResult]:
        """
        Upload new content to the system.
        
        Args:
            request: Content upload request
            
        Returns:
            Tuple of (content_id, validation_result)
        """
        # Validate content
        validation_result = await self.validator.validate_content(
            request.content, request.content_type
        )
        
        if not validation_result.is_valid:
            logger.warning(f"Content validation failed: {validation_result.errors}")
            # Still create content but mark as draft
            status = ContentStatus.DRAFT
        else:
            status = ContentStatus.PENDING_REVIEW

        # Generate content ID
        content_id = self._generate_content_id(request.title, request.created_by)
        
        # Create metadata
        now = datetime.now()
        metadata = ContentMetadata(
            content_id=content_id,
            title=request.title,
            content_type=request.content_type,
            status=status,
            version="1.0",
            language=request.language,
            created_at=now,
            updated_at=now,
            created_by=request.created_by,
            updated_by=request.created_by,
            category=request.category,
            tags=request.tags,
            description=request.description,
            priority=request.priority,
            expires_at=request.expires_at
        )
        
        # Store content
        content_hash = await self.db_manager.store_content(metadata, request.content)
        
        # Create initial version record
        await self._create_version_record(
            content_id=content_id,
            version_number="1.0",
            change_type=ChangeType.CREATED,
            changed_by=request.created_by,
            change_summary="Initial content creation",
            content_hash=content_hash
        )
        
        logger.info(f"Content uploaded: {content_id} by {request.created_by}")
        return content_id, validation_result

    async def update_content(self, request: ContentUpdateRequest) -> Tuple[bool, ContentValidationResult]:
        """
        Update existing content.
        
        Args:
            request: Content update request
            
        Returns:
            Tuple of (success, validation_result)
        """
        # Get existing content
        existing = await self.db_manager.get_content(request.content_id)
        if not existing:
            raise ValueError(f"Content not found: {request.content_id}")
        
        existing_metadata, existing_content = existing
        
        # Prepare updated content
        updated_content = request.content if request.content else existing_content
        
        # Validate updated content
        validation_result = await self.validator.validate_content(
            updated_content, existing_metadata.content_type
        )
        
        # Update metadata
        now = datetime.now()
        updated_metadata = ContentMetadata(
            content_id=existing_metadata.content_id,
            title=request.title if request.title else existing_metadata.title,
            content_type=existing_metadata.content_type,
            status=ContentStatus.PENDING_REVIEW if validation_result.is_valid else ContentStatus.DRAFT,
            version=self._increment_version(existing_metadata.version),
            language=existing_metadata.language,
            created_at=existing_metadata.created_at,
            updated_at=now,
            created_by=existing_metadata.created_by,
            updated_by=request.updated_by,
            category=request.category if request.category else existing_metadata.category,
            tags=request.tags if request.tags else existing_metadata.tags,
            description=request.description if request.description else existing_metadata.description,
            priority=request.priority if request.priority is not None else existing_metadata.priority,
            expires_at=request.expires_at if request.expires_at else existing_metadata.expires_at
        )
        
        # Store updated content
        content_hash = await self.db_manager.store_content(updated_metadata, updated_content)
        
        # Create version record
        await self._create_version_record(
            content_id=request.content_id,
            version_number=updated_metadata.version,
            change_type=ChangeType.UPDATED,
            changed_by=request.updated_by,
            change_summary=request.change_summary,
            content_hash=content_hash,
            previous_version=existing_metadata.version
        )
        
        logger.info(f"Content updated: {request.content_id} by {request.updated_by}")
        return True, validation_result

    async def create_permanent_statement(self, request: PermanentStatementRequest) -> str:
        """Create a permanent statement/FAQ item."""
        statement_id = self._generate_statement_id(request.question, request.created_by)
        
        async with self.db_manager.pg_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO permanent_statements (
                    statement_id, question, answer, category, language, tags,
                    priority, created_at, created_by, updated_at, updated_by
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
                statement_id, request.question, request.answer, request.category,
                request.language, json.dumps(request.tags), request.priority,
                datetime.now(), request.created_by, datetime.now(), request.created_by
            )
        
        logger.info(f"Permanent statement created: {statement_id}")
        return statement_id

    async def search_content(self, search_request: ContentSearchRequest) -> List[ContentMetadata]:
        """Search content based on criteria."""
        return await self.db_manager.search_content(search_request)

    async def approve_content(self, content_id: str, approved_by: str) -> bool:
        """Approve content for publication."""
        async with self.db_manager.pg_pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE content 
                SET status = $1, updated_at = $2, updated_by = $3
                WHERE content_id = $4 AND status = $5
            """, 
                ContentStatus.APPROVED.value, datetime.now(), approved_by,
                content_id, ContentStatus.PENDING_REVIEW.value
            )
            
            if result == "UPDATE 1":
                await self._create_version_record(
                    content_id=content_id,
                    version_number=None,  # Keep same version
                    change_type=ChangeType.APPROVED,
                    changed_by=approved_by,
                    change_summary="Content approved for publication",
                    content_hash=None  # No content change
                )
                logger.info(f"Content approved: {content_id} by {approved_by}")
                return True
            
            return False

    async def publish_content(self, content_id: str, published_by: str) -> bool:
        """Publish approved content."""
        async with self.db_manager.pg_pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE content 
                SET status = $1, updated_at = $2, updated_by = $3
                WHERE content_id = $4 AND status = $5
            """, 
                ContentStatus.PUBLISHED.value, datetime.now(), published_by,
                content_id, ContentStatus.APPROVED.value
            )
            
            if result == "UPDATE 1":
                await self._create_version_record(
                    content_id=content_id,
                    version_number=None,
                    change_type=ChangeType.PUBLISHED,
                    changed_by=published_by,
                    change_summary="Content published",
                    content_hash=None
                )
                logger.info(f"Content published: {content_id} by {published_by}")
                return True
            
            return False

    def _generate_content_id(self, title: str, created_by: str) -> str:
        """Generate unique content ID."""
        timestamp = datetime.now().isoformat()
        content_string = f"{title}_{created_by}_{timestamp}"
        return hashlib.md5(content_string.encode()).hexdigest()

    def _generate_statement_id(self, question: str, created_by: str) -> str:
        """Generate unique statement ID."""
        timestamp = datetime.now().isoformat()
        statement_string = f"{question}_{created_by}_{timestamp}"
        return hashlib.md5(statement_string.encode()).hexdigest()

    def _increment_version(self, current_version: str) -> str:
        """Increment version number."""
        try:
            major, minor = current_version.split('.')
            return f"{major}.{int(minor) + 1}"
        except ValueError:
            return "1.1"

    async def _create_version_record(
        self,
        content_id: str,
        version_number: Optional[str],
        change_type: ChangeType,
        changed_by: str,
        change_summary: str,
        content_hash: Optional[str],
        previous_version: Optional[str] = None
    ):
        """Create a version tracking record."""
        version_id = hashlib.md5(f"{content_id}_{datetime.now().isoformat()}".encode()).hexdigest()
        
        # Get current content for snapshot if needed
        content_snapshot = {}
        if content_hash:
            existing = await self.db_manager.get_content(content_id)
            if existing:
                metadata, content_text = existing
                content_snapshot = {
                    "title": metadata.title,
                    "content": content_text,
                    "metadata": {
                        "category": metadata.category,
                        "tags": metadata.tags,
                        "description": metadata.description,
                        "priority": metadata.priority
                    }
                }
        
        async with self.db_manager.pg_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO content_versions (
                    version_id, content_id, version_number, change_type,
                    changed_by, changed_at, change_summary, content_hash,
                    previous_version, content_snapshot
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
                version_id, content_id, version_number or "0.0", change_type.value,
                changed_by, datetime.now(), change_summary, content_hash or "",
                previous_version, json.dumps(content_snapshot)
            )


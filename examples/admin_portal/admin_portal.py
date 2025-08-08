"""
LIXIL AI Hub Admin Portal Backend

This module provides the backend API for the AI Council admin portal,
enabling policy management, permanent statement creation, content approval,
and system administration for the LIXIL AI Hub Platform.

Key Features:
- Policy document upload and versioning
- Permanent statement management
- Content approval workflows
- User role management
- System analytics and monitoring
- Audit trail and compliance reporting

Author: LIXIL AI Hub Platform Team
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import hashlib
import mimetypes
from pathlib import Path

import asyncpg
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import aiofiles

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolicyStatus(str, Enum):
    """Policy document status."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    REJECTED = "rejected"


class ContentType(str, Enum):
    """Content types for admin portal."""
    POLICY_DOCUMENT = "policy_document"
    PERMANENT_STATEMENT = "permanent_statement"
    FAQ_ENTRY = "faq_entry"
    TRAINING_MATERIAL = "training_material"
    ANNOUNCEMENT = "announcement"


class ApprovalAction(str, Enum):
    """Approval actions."""
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"
    PUBLISH = "publish"
    ARCHIVE = "archive"


class UserRole(str, Enum):
    """User roles for admin portal."""
    SUPER_ADMIN = "super_admin"
    AI_COUNCIL_MEMBER = "ai_council_member"
    REGIONAL_ADMIN = "regional_admin"
    CONTENT_MANAGER = "content_manager"
    POLICY_REVIEWER = "policy_reviewer"


@dataclass
class PolicyDocument:
    """Policy document data structure."""
    policy_id: str
    title: str
    content: str
    version: str
    status: PolicyStatus
    category: str
    language: str
    region: Optional[str]
    created_by: str
    created_at: datetime
    updated_at: datetime
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    published_at: Optional[datetime] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    checksum: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PermanentStatement:
    """Permanent statement data structure."""
    statement_id: str
    question: str
    answer: str
    category: str
    priority: int
    language: str
    region: Optional[str]
    is_active: bool
    created_by: str
    created_at: datetime
    updated_at: datetime
    tags: List[str] = field(default_factory=list)
    usage_count: int = 0
    last_used: Optional[datetime] = None


class PolicyUploadRequest(BaseModel):
    """Request model for policy upload."""
    title: str = Field(..., min_length=1, max_length=200)
    category: str = Field(..., min_length=1, max_length=100)
    language: str = Field(default="en", regex="^[a-z]{2}$")
    region: Optional[str] = Field(None, max_length=10)
    tags: List[str] = Field(default_factory=list, max_items=10)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('tags')
    def validate_tags(cls, v):
        """Validate tags format."""
        for tag in v:
            if len(tag) > 50:
                raise ValueError("Each tag must be 50 characters or less")
        return v


class PermanentStatementRequest(BaseModel):
    """Request model for permanent statement creation."""
    question: str = Field(..., min_length=10, max_length=500)
    answer: str = Field(..., min_length=10, max_length=2000)
    category: str = Field(..., min_length=1, max_length=100)
    priority: int = Field(default=5, ge=1, le=10)
    language: str = Field(default="en", regex="^[a-z]{2}$")
    region: Optional[str] = Field(None, max_length=10)
    tags: List[str] = Field(default_factory=list, max_items=10)


class ApprovalRequest(BaseModel):
    """Request model for content approval."""
    content_id: str = Field(..., min_length=1)
    content_type: ContentType
    action: ApprovalAction
    comments: Optional[str] = Field(None, max_length=1000)
    publish_immediately: bool = Field(default=False)


class UserManagementRequest(BaseModel):
    """Request model for user management."""
    user_id: str = Field(..., min_length=1)
    action: str = Field(..., regex="^(activate|deactivate|assign_role|remove_role)$")
    role: Optional[UserRole] = None
    reason: Optional[str] = Field(None, max_length=500)


class AdminAnalyticsRequest(BaseModel):
    """Request model for analytics data."""
    start_date: datetime
    end_date: datetime
    metrics: List[str] = Field(default_factory=list)
    region: Optional[str] = None
    granularity: str = Field(default="day", regex="^(hour|day|week|month)$")


class PolicySearchRequest(BaseModel):
    """Request model for policy search."""
    query: Optional[str] = None
    category: Optional[str] = None
    status: Optional[PolicyStatus] = None
    language: Optional[str] = None
    region: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class AdminPortalManager:
    """
    Main manager for admin portal functionality.
    
    Handles policy management, content approval, user administration,
    and system analytics for AI Council members.
    """

    def __init__(self, database_url: str, file_storage_path: str):
        """
        Initialize admin portal manager.
        
        Args:
            database_url: PostgreSQL database connection URL
            file_storage_path: Path for storing uploaded files
        """
        self.database_url = database_url
        self.file_storage_path = Path(file_storage_path)
        self.file_storage_path.mkdir(parents=True, exist_ok=True)
        self.db_pool = None

    async def initialize(self):
        """Initialize database connection and schema."""
        self.db_pool = await asyncpg.create_pool(
            self.database_url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        
        await self._create_schema()
        logger.info("Admin portal manager initialized")

    async def close(self):
        """Close database connections."""
        if self.db_pool:
            await self.db_pool.close()

    async def _create_schema(self):
        """Create database schema for admin portal."""
        schema_sql = """
        -- Policy documents table
        CREATE TABLE IF NOT EXISTS policy_documents (
            policy_id VARCHAR(64) PRIMARY KEY,
            title VARCHAR(200) NOT NULL,
            content TEXT NOT NULL,
            version VARCHAR(20) NOT NULL,
            status VARCHAR(50) NOT NULL,
            category VARCHAR(100) NOT NULL,
            language VARCHAR(2) NOT NULL,
            region VARCHAR(10),
            created_by VARCHAR(64) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
            approved_by VARCHAR(64),
            approved_at TIMESTAMP WITH TIME ZONE,
            published_at TIMESTAMP WITH TIME ZONE,
            file_path TEXT,
            file_size INTEGER,
            checksum VARCHAR(64),
            tags JSONB,
            metadata JSONB
        );

        -- Permanent statements table
        CREATE TABLE IF NOT EXISTS permanent_statements (
            statement_id VARCHAR(64) PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            category VARCHAR(100) NOT NULL,
            priority INTEGER NOT NULL DEFAULT 5,
            language VARCHAR(2) NOT NULL,
            region VARCHAR(10),
            is_active BOOLEAN DEFAULT TRUE,
            created_by VARCHAR(64) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
            tags JSONB,
            usage_count INTEGER DEFAULT 0,
            last_used TIMESTAMP WITH TIME ZONE
        );

        -- Content approval workflow table
        CREATE TABLE IF NOT EXISTS content_approvals (
            approval_id SERIAL PRIMARY KEY,
            content_id VARCHAR(64) NOT NULL,
            content_type VARCHAR(50) NOT NULL,
            action VARCHAR(50) NOT NULL,
            approved_by VARCHAR(64) NOT NULL,
            approved_at TIMESTAMP WITH TIME ZONE NOT NULL,
            comments TEXT,
            previous_status VARCHAR(50),
            new_status VARCHAR(50)
        );

        -- Admin audit log table
        CREATE TABLE IF NOT EXISTS admin_audit_log (
            log_id SERIAL PRIMARY KEY,
            user_id VARCHAR(64) NOT NULL,
            action VARCHAR(100) NOT NULL,
            resource_type VARCHAR(50) NOT NULL,
            resource_id VARCHAR(64),
            details JSONB,
            ip_address INET,
            user_agent TEXT,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL
        );

        -- System analytics table
        CREATE TABLE IF NOT EXISTS system_analytics (
            metric_id SERIAL PRIMARY KEY,
            metric_name VARCHAR(100) NOT NULL,
            metric_value DECIMAL(15, 4) NOT NULL,
            metric_unit VARCHAR(20),
            region VARCHAR(10),
            recorded_at TIMESTAMP WITH TIME ZONE NOT NULL,
            metadata JSONB
        );

        -- User management log table
        CREATE TABLE IF NOT EXISTS user_management_log (
            log_id SERIAL PRIMARY KEY,
            target_user_id VARCHAR(64) NOT NULL,
            admin_user_id VARCHAR(64) NOT NULL,
            action VARCHAR(50) NOT NULL,
            previous_value TEXT,
            new_value TEXT,
            reason TEXT,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL
        );

        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_policy_documents_status ON policy_documents(status);
        CREATE INDEX IF NOT EXISTS idx_policy_documents_category ON policy_documents(category);
        CREATE INDEX IF NOT EXISTS idx_policy_documents_language ON policy_documents(language);
        CREATE INDEX IF NOT EXISTS idx_policy_documents_created_at ON policy_documents(created_at);
        CREATE INDEX IF NOT EXISTS idx_permanent_statements_category ON permanent_statements(category);
        CREATE INDEX IF NOT EXISTS idx_permanent_statements_active ON permanent_statements(is_active);
        CREATE INDEX IF NOT EXISTS idx_permanent_statements_priority ON permanent_statements(priority);
        CREATE INDEX IF NOT EXISTS idx_content_approvals_content ON content_approvals(content_id);
        CREATE INDEX IF NOT EXISTS idx_admin_audit_log_user ON admin_audit_log(user_id);
        CREATE INDEX IF NOT EXISTS idx_admin_audit_log_timestamp ON admin_audit_log(timestamp);
        CREATE INDEX IF NOT EXISTS idx_system_analytics_metric ON system_analytics(metric_name);
        CREATE INDEX IF NOT EXISTS idx_system_analytics_recorded_at ON system_analytics(recorded_at);
        """
        
        async with self.db_pool.acquire() as conn:
            await conn.execute(schema_sql)

    async def upload_policy_document(self, file: UploadFile, request: PolicyUploadRequest, 
                                   user_id: str) -> str:
        """Upload and process a policy document."""
        # Validate file
        if not file.filename:
            raise ValueError("No file provided")
        
        allowed_types = {'.pdf', '.docx', '.doc', '.txt', '.md'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_types:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Generate policy ID and file path
        policy_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{policy_id}{file_ext}"
        file_path = self.file_storage_path / "policies" / safe_filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save file and calculate checksum
        file_content = await file.read()
        file_size = len(file_content)
        checksum = hashlib.sha256(file_content).hexdigest()
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)
        
        # Extract text content (simplified - would use proper document parsers)
        if file_ext == '.txt':
            content = file_content.decode('utf-8')
        elif file_ext == '.md':
            content = file_content.decode('utf-8')
        else:
            content = f"[Binary file: {file.filename}]"  # Placeholder for document parsing
        
        # Determine version
        version = await self._get_next_version(request.title, request.category)
        
        # Create policy document
        now = datetime.now()
        policy = PolicyDocument(
            policy_id=policy_id,
            title=request.title,
            content=content,
            version=version,
            status=PolicyStatus.DRAFT,
            category=request.category,
            language=request.language,
            region=request.region,
            created_by=user_id,
            created_at=now,
            updated_at=now,
            file_path=str(file_path),
            file_size=file_size,
            checksum=checksum,
            tags=request.tags,
            metadata=request.metadata
        )
        
        # Save to database
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO policy_documents (
                    policy_id, title, content, version, status, category, language,
                    region, created_by, created_at, updated_at, file_path, file_size,
                    checksum, tags, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            """,
                policy_id, request.title, content, version, PolicyStatus.DRAFT.value,
                request.category, request.language, request.region, user_id, now, now,
                str(file_path), file_size, checksum, json.dumps(request.tags),
                json.dumps(request.metadata)
            )
        
        # Log action
        await self._log_admin_action(
            user_id, "upload_policy", "policy_document", policy_id,
            {"title": request.title, "category": request.category}
        )
        
        logger.info(f"Policy document uploaded: {policy_id} by {user_id}")
        return policy_id

    async def create_permanent_statement(self, request: PermanentStatementRequest, 
                                       user_id: str) -> str:
        """Create a new permanent statement."""
        statement_id = str(uuid.uuid4())
        now = datetime.now()
        
        statement = PermanentStatement(
            statement_id=statement_id,
            question=request.question,
            answer=request.answer,
            category=request.category,
            priority=request.priority,
            language=request.language,
            region=request.region,
            is_active=True,
            created_by=user_id,
            created_at=now,
            updated_at=now,
            tags=request.tags
        )
        
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO permanent_statements (
                    statement_id, question, answer, category, priority, language,
                    region, is_active, created_by, created_at, updated_at, tags
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
                statement_id, request.question, request.answer, request.category,
                request.priority, request.language, request.region, True, user_id,
                now, now, json.dumps(request.tags)
            )
        
        # Log action
        await self._log_admin_action(
            user_id, "create_statement", "permanent_statement", statement_id,
            {"question": request.question[:100], "category": request.category}
        )
        
        logger.info(f"Permanent statement created: {statement_id} by {user_id}")
        return statement_id

    async def approve_content(self, request: ApprovalRequest, user_id: str) -> bool:
        """Approve or reject content."""
        async with self.db_pool.acquire() as conn:
            # Get current content status
            if request.content_type == ContentType.POLICY_DOCUMENT:
                current = await conn.fetchrow("""
                    SELECT status FROM policy_documents WHERE policy_id = $1
                """, request.content_id)
                table_name = "policy_documents"
                id_column = "policy_id"
            elif request.content_type == ContentType.PERMANENT_STATEMENT:
                current = await conn.fetchrow("""
                    SELECT is_active FROM permanent_statements WHERE statement_id = $1
                """, request.content_id)
                table_name = "permanent_statements"
                id_column = "statement_id"
            else:
                raise ValueError(f"Unsupported content type: {request.content_type}")
            
            if not current:
                raise ValueError("Content not found")
            
            # Determine new status
            if request.content_type == ContentType.POLICY_DOCUMENT:
                previous_status = current["status"]
                if request.action == ApprovalAction.APPROVE:
                    new_status = PolicyStatus.APPROVED.value
                elif request.action == ApprovalAction.REJECT:
                    new_status = PolicyStatus.REJECTED.value
                elif request.action == ApprovalAction.PUBLISH:
                    new_status = PolicyStatus.PUBLISHED.value
                elif request.action == ApprovalAction.ARCHIVE:
                    new_status = PolicyStatus.ARCHIVED.value
                else:
                    new_status = PolicyStatus.PENDING_REVIEW.value
                
                # Update policy document
                update_fields = ["status = $2", "updated_at = $3"]
                params = [request.content_id, new_status, datetime.now()]
                
                if request.action == ApprovalAction.APPROVE:
                    update_fields.extend(["approved_by = $4", "approved_at = $5"])
                    params.extend([user_id, datetime.now()])
                elif request.action == ApprovalAction.PUBLISH:
                    update_fields.extend(["approved_by = $4", "approved_at = $5", "published_at = $6"])
                    params.extend([user_id, datetime.now(), datetime.now()])
                
                await conn.execute(f"""
                    UPDATE {table_name} SET {', '.join(update_fields)}
                    WHERE {id_column} = $1
                """, *params)
                
            else:  # Permanent statement
                previous_status = "active" if current["is_active"] else "inactive"
                new_status = "active" if request.action == ApprovalAction.APPROVE else "inactive"
                
                await conn.execute(f"""
                    UPDATE {table_name} SET is_active = $2, updated_at = $3
                    WHERE {id_column} = $1
                """, request.content_id, request.action == ApprovalAction.APPROVE, datetime.now())
            
            # Log approval action
            await conn.execute("""
                INSERT INTO content_approvals (
                    content_id, content_type, action, approved_by, approved_at,
                    comments, previous_status, new_status
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
                request.content_id, request.content_type.value, request.action.value,
                user_id, datetime.now(), request.comments, previous_status, new_status
            )
        
        # Log admin action
        await self._log_admin_action(
            user_id, f"approve_content_{request.action.value}", request.content_type.value,
            request.content_id, {"comments": request.comments}
        )
        
        logger.info(f"Content {request.action.value}: {request.content_id} by {user_id}")
        return True

    async def search_policies(self, request: PolicySearchRequest) -> Dict[str, Any]:
        """Search policy documents with filters."""
        # Build dynamic query
        where_conditions = ["1=1"]
        params = []
        param_count = 0
        
        if request.query:
            param_count += 1
            where_conditions.append(f"(title ILIKE ${param_count} OR content ILIKE ${param_count})")
            params.append(f"%{request.query}%")
        
        if request.category:
            param_count += 1
            where_conditions.append(f"category = ${param_count}")
            params.append(request.category)
        
        if request.status:
            param_count += 1
            where_conditions.append(f"status = ${param_count}")
            params.append(request.status.value)
        
        if request.language:
            param_count += 1
            where_conditions.append(f"language = ${param_count}")
            params.append(request.language)
        
        if request.region:
            param_count += 1
            where_conditions.append(f"region = ${param_count}")
            params.append(request.region)
        
        if request.created_after:
            param_count += 1
            where_conditions.append(f"created_at >= ${param_count}")
            params.append(request.created_after)
        
        if request.created_before:
            param_count += 1
            where_conditions.append(f"created_at <= ${param_count}")
            params.append(request.created_before)
        
        if request.tags:
            param_count += 1
            where_conditions.append(f"tags ?| ${param_count}")
            params.append(request.tags)
        
        # Add pagination
        param_count += 1
        limit_param = param_count
        params.append(request.limit)
        
        param_count += 1
        offset_param = param_count
        params.append(request.offset)
        
        where_clause = " AND ".join(where_conditions)
        
        async with self.db_pool.acquire() as conn:
            # Get total count
            count_query = f"""
                SELECT COUNT(*) FROM policy_documents WHERE {where_clause}
            """
            total_count = await conn.fetchval(count_query, *params[:-2])  # Exclude limit/offset
            
            # Get results
            search_query = f"""
                SELECT policy_id, title, version, status, category, language, region,
                       created_by, created_at, updated_at, approved_by, approved_at,
                       published_at, tags, file_size
                FROM policy_documents
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ${limit_param} OFFSET ${offset_param}
            """
            
            results = await conn.fetch(search_query, *params)
            
            policies = []
            for row in results:
                policies.append({
                    "policy_id": row["policy_id"],
                    "title": row["title"],
                    "version": row["version"],
                    "status": row["status"],
                    "category": row["category"],
                    "language": row["language"],
                    "region": row["region"],
                    "created_by": row["created_by"],
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat(),
                    "approved_by": row["approved_by"],
                    "approved_at": row["approved_at"].isoformat() if row["approved_at"] else None,
                    "published_at": row["published_at"].isoformat() if row["published_at"] else None,
                    "tags": json.loads(row["tags"]) if row["tags"] else [],
                    "file_size": row["file_size"]
                })
        
        return {
            "policies": policies,
            "total_count": total_count,
            "page_size": request.limit,
            "offset": request.offset,
            "has_more": (request.offset + request.limit) < total_count
        }

    async def get_admin_analytics(self, request: AdminAnalyticsRequest) -> Dict[str, Any]:
        """Get analytics data for admin dashboard."""
        analytics = {}
        
        async with self.db_pool.acquire() as conn:
            # Policy document statistics
            policy_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_policies,
                    COUNT(CASE WHEN status = 'published' THEN 1 END) as published_policies,
                    COUNT(CASE WHEN status = 'pending_review' THEN 1 END) as pending_policies,
                    COUNT(CASE WHEN created_at >= $1 THEN 1 END) as new_policies
                FROM policy_documents
                WHERE created_at BETWEEN $1 AND $2
            """, request.start_date, request.end_date)
            
            # Permanent statement statistics
            statement_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_statements,
                    COUNT(CASE WHEN is_active = true THEN 1 END) as active_statements,
                    SUM(usage_count) as total_usage,
                    AVG(usage_count) as avg_usage
                FROM permanent_statements
                WHERE created_at BETWEEN $1 AND $2
            """, request.start_date, request.end_date)
            
            # Content approval activity
            approval_stats = await conn.fetch("""
                SELECT action, COUNT(*) as count
                FROM content_approvals
                WHERE approved_at BETWEEN $1 AND $2
                GROUP BY action
                ORDER BY count DESC
            """, request.start_date, request.end_date)
            
            # Admin activity
            admin_activity = await conn.fetch("""
                SELECT 
                    user_id,
                    COUNT(*) as action_count,
                    COUNT(DISTINCT action) as unique_actions
                FROM admin_audit_log
                WHERE timestamp BETWEEN $1 AND $2
                GROUP BY user_id
                ORDER BY action_count DESC
                LIMIT 10
            """, request.start_date, request.end_date)
            
            # Time series data (if requested)
            if request.granularity in ["day", "week", "month"]:
                time_series = await self._get_time_series_data(
                    request.start_date, request.end_date, request.granularity
                )
                analytics["time_series"] = time_series
        
        analytics.update({
            "policy_statistics": dict(policy_stats) if policy_stats else {},
            "statement_statistics": dict(statement_stats) if statement_stats else {},
            "approval_activity": [dict(row) for row in approval_stats],
            "admin_activity": [dict(row) for row in admin_activity],
            "period": {
                "start_date": request.start_date.isoformat(),
                "end_date": request.end_date.isoformat(),
                "granularity": request.granularity
            }
        })
        
        return analytics

    async def _get_next_version(self, title: str, category: str) -> str:
        """Get next version number for a policy document."""
        async with self.db_pool.acquire() as conn:
            latest_version = await conn.fetchval("""
                SELECT version FROM policy_documents
                WHERE title = $1 AND category = $2
                ORDER BY created_at DESC
                LIMIT 1
            """, title, category)
            
            if not latest_version:
                return "1.0.0"
            
            # Parse version and increment
            try:
                major, minor, patch = map(int, latest_version.split('.'))
                return f"{major}.{minor}.{patch + 1}"
            except:
                return "1.0.0"

    async def _get_time_series_data(self, start_date: datetime, end_date: datetime, 
                                  granularity: str) -> List[Dict[str, Any]]:
        """Get time series analytics data."""
        if granularity == "day":
            date_trunc = "day"
        elif granularity == "week":
            date_trunc = "week"
        elif granularity == "month":
            date_trunc = "month"
        else:
            date_trunc = "day"
        
        async with self.db_pool.acquire() as conn:
            time_series = await conn.fetch(f"""
                SELECT 
                    DATE_TRUNC('{date_trunc}', created_at) as period,
                    COUNT(*) as policy_count,
                    COUNT(CASE WHEN status = 'published' THEN 1 END) as published_count
                FROM policy_documents
                WHERE created_at BETWEEN $1 AND $2
                GROUP BY DATE_TRUNC('{date_trunc}', created_at)
                ORDER BY period
            """, start_date, end_date)
            
            return [
                {
                    "period": row["period"].isoformat(),
                    "policy_count": row["policy_count"],
                    "published_count": row["published_count"]
                }
                for row in time_series
            ]

    async def _log_admin_action(self, user_id: str, action: str, resource_type: str,
                              resource_id: Optional[str], details: Dict[str, Any],
                              ip_address: Optional[str] = None, user_agent: Optional[str] = None):
        """Log admin action for audit trail."""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO admin_audit_log (
                    user_id, action, resource_type, resource_id, details,
                    ip_address, user_agent, timestamp
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
                user_id, action, resource_type, resource_id, json.dumps(details),
                ip_address, user_agent, datetime.now()
            )


# FastAPI application setup
app = FastAPI(
    title="LIXIL AI Hub Admin Portal API",
    description="Admin portal API for AI Council policy management",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://admin.lixil-ai-hub.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global admin portal manager instance
admin_portal: Optional[AdminPortalManager] = None


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Get current authenticated user from JWT token."""
    # This would integrate with the auth_manager from the auth_rbac example
    # For now, return a mock user ID
    return "admin_user_123"


@app.on_event("startup")
async def startup_event():
    """Initialize admin portal on startup."""
    global admin_portal
    admin_portal = AdminPortalManager(
        database_url="postgresql://user:pass@localhost:5432/lixil_ai_hub",
        file_storage_path="/app/storage"
    )
    await admin_portal.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    if admin_portal:
        await admin_portal.close()


@app.post("/api/admin/policies/upload")
async def upload_policy(
    file: UploadFile = File(...),
    title: str = Form(...),
    category: str = Form(...),
    language: str = Form("en"),
    region: Optional[str] = Form(None),
    tags: str = Form("[]"),
    metadata: str = Form("{}"),
    current_user: str = Depends(get_current_user)
):
    """Upload a new policy document."""
    try:
        request = PolicyUploadRequest(
            title=title,
            category=category,
            language=language,
            region=region,
            tags=json.loads(tags),
            metadata=json.loads(metadata)
        )
        
        policy_id = await admin_portal.upload_policy_document(file, request, current_user)
        return {"policy_id": policy_id, "message": "Policy uploaded successfully"}
        
    except Exception as e:
        logger.error(f"Policy upload failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/admin/statements")
async def create_statement(
    request: PermanentStatementRequest,
    current_user: str = Depends(get_current_user)
):
    """Create a new permanent statement."""
    try:
        statement_id = await admin_portal.create_permanent_statement(request, current_user)
        return {"statement_id": statement_id, "message": "Statement created successfully"}
        
    except Exception as e:
        logger.error(f"Statement creation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/admin/content/approve")
async def approve_content(
    request: ApprovalRequest,
    current_user: str = Depends(get_current_user)
):
    """Approve or reject content."""
    try:
        success = await admin_portal.approve_content(request, current_user)
        return {"success": success, "message": f"Content {request.action.value} successfully"}
        
    except Exception as e:
        logger.error(f"Content approval failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/admin/policies/search")
async def search_policies(request: PolicySearchRequest):
    """Search policy documents."""
    try:
        results = await admin_portal.search_policies(request)
        return results
        
    except Exception as e:
        logger.error(f"Policy search failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/admin/analytics")
async def get_analytics(request: AdminAnalyticsRequest):
    """Get admin analytics data."""
    try:
        analytics = await admin_portal.get_admin_analytics(request)
        return analytics
        
    except Exception as e:
        logger.error(f"Analytics request failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


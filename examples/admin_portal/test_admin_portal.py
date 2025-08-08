"""
Comprehensive test suite for LIXIL AI Hub Admin Portal.

Tests cover policy management, statement creation, content approval,
analytics, and admin functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json
import tempfile
from pathlib import Path

from admin_portal import (
    AdminPortalManager, PolicyDocument, PermanentStatement,
    PolicyUploadRequest, PermanentStatementRequest, ApprovalRequest,
    PolicySearchRequest, AdminAnalyticsRequest,
    PolicyStatus, ContentType, ApprovalAction
)


class TestPolicyUploadRequest:
    """Test policy upload request validation."""

    def test_valid_policy_upload_request(self):
        """Test valid policy upload request."""
        request = PolicyUploadRequest(
            title="AI Usage Policy",
            category="AI Policy",
            language="en",
            region="US",
            tags=["ai", "policy", "usage"],
            metadata={"version": "1.0", "department": "IT"}
        )
        
        assert request.title == "AI Usage Policy"
        assert request.category == "AI Policy"
        assert request.language == "en"
        assert len(request.tags) == 3

    def test_invalid_language_format(self):
        """Test policy upload with invalid language format."""
        with pytest.raises(ValueError):
            PolicyUploadRequest(
                title="Test Policy",
                category="Test",
                language="english"  # Should be 2-letter code
            )

    def test_too_many_tags(self):
        """Test policy upload with too many tags."""
        with pytest.raises(ValueError):
            PolicyUploadRequest(
                title="Test Policy",
                category="Test",
                tags=["tag" + str(i) for i in range(15)]  # More than 10 tags
            )

    def test_tag_too_long(self):
        """Test policy upload with tag that's too long."""
        with pytest.raises(ValueError):
            PolicyUploadRequest(
                title="Test Policy",
                category="Test",
                tags=["a" * 60]  # Tag longer than 50 characters
            )


class TestPermanentStatementRequest:
    """Test permanent statement request validation."""

    def test_valid_statement_request(self):
        """Test valid statement request."""
        request = PermanentStatementRequest(
            question="What is the AI usage policy?",
            answer="The AI usage policy defines how employees should use AI tools.",
            category="AI Policy",
            priority=8,
            language="en",
            tags=["ai", "policy"]
        )
        
        assert request.question == "What is the AI usage policy?"
        assert request.priority == 8
        assert len(request.tags) == 2

    def test_question_too_short(self):
        """Test statement with question that's too short."""
        with pytest.raises(ValueError):
            PermanentStatementRequest(
                question="What?",  # Too short
                answer="The AI usage policy defines how employees should use AI tools.",
                category="AI Policy"
            )

    def test_answer_too_long(self):
        """Test statement with answer that's too long."""
        with pytest.raises(ValueError):
            PermanentStatementRequest(
                question="What is the AI usage policy?",
                answer="a" * 2500,  # Too long
                category="AI Policy"
            )

    def test_invalid_priority_range(self):
        """Test statement with invalid priority."""
        with pytest.raises(ValueError):
            PermanentStatementRequest(
                question="What is the AI usage policy?",
                answer="The AI usage policy defines how employees should use AI tools.",
                category="AI Policy",
                priority=15  # Out of range (1-10)
            )


class TestPolicySearchRequest:
    """Test policy search request validation."""

    def test_valid_search_request(self):
        """Test valid search request."""
        request = PolicySearchRequest(
            query="AI policy",
            category="AI Policy",
            status=PolicyStatus.PUBLISHED,
            language="en",
            limit=50,
            offset=0
        )
        
        assert request.query == "AI policy"
        assert request.status == PolicyStatus.PUBLISHED
        assert request.limit == 50

    def test_limit_validation(self):
        """Test search request with invalid limit."""
        with pytest.raises(ValueError):
            PolicySearchRequest(limit=150)  # Exceeds maximum

    def test_negative_offset(self):
        """Test search request with negative offset."""
        with pytest.raises(ValueError):
            PolicySearchRequest(offset=-1)


@pytest.mark.asyncio
class TestAdminPortalManager:
    """Test admin portal manager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.database_url = "postgresql://test:test@localhost:5432/test_db"
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock database pool
        self.mock_pool = AsyncMock()
        self.mock_conn = AsyncMock()
        self.mock_pool.acquire.return_value.__aenter__.return_value = self.mock_conn
        
        self.admin_portal = AdminPortalManager(self.database_url, self.temp_dir)
        self.admin_portal.db_pool = self.mock_pool

    async def test_upload_policy_document_success(self):
        """Test successful policy document upload."""
        # Mock file upload
        mock_file = Mock()
        mock_file.filename = "test_policy.pdf"
        mock_file.read = AsyncMock(return_value=b"Test policy content")
        
        # Mock database responses
        self.mock_conn.fetchval.return_value = None  # No existing version
        self.mock_conn.execute.return_value = None
        
        request = PolicyUploadRequest(
            title="Test AI Policy",
            category="AI Policy",
            language="en",
            tags=["ai", "test"]
        )
        
        with patch('aiofiles.open', create=True) as mock_open:
            mock_open.return_value.__aenter__.return_value.write = AsyncMock()
            
            policy_id = await self.admin_portal.upload_policy_document(
                mock_file, request, "admin_user_123"
            )
        
        assert policy_id is not None
        assert len(policy_id) == 36  # UUID length with hyphens
        
        # Verify database calls
        assert self.mock_conn.execute.call_count == 2  # Policy insert + audit log

    async def test_upload_policy_document_invalid_file_type(self):
        """Test policy upload with invalid file type."""
        mock_file = Mock()
        mock_file.filename = "test_policy.exe"  # Invalid file type
        
        request = PolicyUploadRequest(
            title="Test Policy",
            category="Test"
        )
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            await self.admin_portal.upload_policy_document(
                mock_file, request, "admin_user_123"
            )

    async def test_upload_policy_document_no_filename(self):
        """Test policy upload with no filename."""
        mock_file = Mock()
        mock_file.filename = None
        
        request = PolicyUploadRequest(
            title="Test Policy",
            category="Test"
        )
        
        with pytest.raises(ValueError, match="No file provided"):
            await self.admin_portal.upload_policy_document(
                mock_file, request, "admin_user_123"
            )

    async def test_create_permanent_statement_success(self):
        """Test successful permanent statement creation."""
        self.mock_conn.execute.return_value = None
        
        request = PermanentStatementRequest(
            question="What is the AI usage policy?",
            answer="The AI usage policy defines how employees should use AI tools.",
            category="AI Policy",
            priority=8,
            tags=["ai", "policy"]
        )
        
        statement_id = await self.admin_portal.create_permanent_statement(
            request, "admin_user_123"
        )
        
        assert statement_id is not None
        assert len(statement_id) == 36  # UUID length
        
        # Verify database calls
        assert self.mock_conn.execute.call_count == 2  # Statement insert + audit log

    async def test_approve_content_policy_success(self):
        """Test successful policy content approval."""
        # Mock existing policy
        self.mock_conn.fetchrow.return_value = {"status": "pending_review"}
        self.mock_conn.execute.return_value = None
        
        request = ApprovalRequest(
            content_id="test_policy_123",
            content_type=ContentType.POLICY_DOCUMENT,
            action=ApprovalAction.APPROVE,
            comments="Looks good"
        )
        
        result = await self.admin_portal.approve_content(request, "admin_user_123")
        
        assert result is True
        
        # Verify database calls
        assert self.mock_conn.execute.call_count == 3  # Update + approval log + audit log

    async def test_approve_content_statement_success(self):
        """Test successful statement content approval."""
        # Mock existing statement
        self.mock_conn.fetchrow.return_value = {"is_active": False}
        self.mock_conn.execute.return_value = None
        
        request = ApprovalRequest(
            content_id="test_statement_123",
            content_type=ContentType.PERMANENT_STATEMENT,
            action=ApprovalAction.APPROVE
        )
        
        result = await self.admin_portal.approve_content(request, "admin_user_123")
        
        assert result is True

    async def test_approve_content_not_found(self):
        """Test content approval with non-existent content."""
        self.mock_conn.fetchrow.return_value = None
        
        request = ApprovalRequest(
            content_id="nonexistent_123",
            content_type=ContentType.POLICY_DOCUMENT,
            action=ApprovalAction.APPROVE
        )
        
        with pytest.raises(ValueError, match="Content not found"):
            await self.admin_portal.approve_content(request, "admin_user_123")

    async def test_search_policies_basic(self):
        """Test basic policy search."""
        # Mock search results
        self.mock_conn.fetchval.return_value = 5  # Total count
        self.mock_conn.fetch.return_value = [
            {
                "policy_id": "policy_1",
                "title": "AI Usage Policy",
                "version": "1.0.0",
                "status": "published",
                "category": "AI Policy",
                "language": "en",
                "region": "US",
                "created_by": "admin_user",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "approved_by": "admin_user",
                "approved_at": datetime.now(),
                "published_at": datetime.now(),
                "tags": '["ai", "policy"]',
                "file_size": 1024
            }
        ]
        
        request = PolicySearchRequest(
            query="AI",
            limit=10,
            offset=0
        )
        
        results = await self.admin_portal.search_policies(request)
        
        assert results["total_count"] == 5
        assert len(results["policies"]) == 1
        assert results["policies"][0]["title"] == "AI Usage Policy"
        assert results["has_more"] is False

    async def test_search_policies_with_filters(self):
        """Test policy search with multiple filters."""
        self.mock_conn.fetchval.return_value = 0
        self.mock_conn.fetch.return_value = []
        
        request = PolicySearchRequest(
            query="AI policy",
            category="AI Policy",
            status=PolicyStatus.PUBLISHED,
            language="en",
            region="US",
            tags=["ai", "policy"],
            created_after=datetime.now() - timedelta(days=30),
            limit=20,
            offset=0
        )
        
        results = await self.admin_portal.search_policies(request)
        
        assert results["total_count"] == 0
        assert len(results["policies"]) == 0
        
        # Verify that the query was built with all filters
        assert self.mock_conn.fetch.called

    async def test_get_admin_analytics_success(self):
        """Test successful analytics data retrieval."""
        # Mock analytics data
        self.mock_conn.fetchrow.side_effect = [
            {  # Policy stats
                "total_policies": 25,
                "published_policies": 20,
                "pending_policies": 3,
                "new_policies": 5
            },
            {  # Statement stats
                "total_statements": 50,
                "active_statements": 45,
                "total_usage": 1250,
                "avg_usage": 25.0
            }
        ]
        
        self.mock_conn.fetch.side_effect = [
            [  # Approval stats
                {"action": "approve", "count": 15},
                {"action": "reject", "count": 2}
            ],
            [  # Admin activity
                {"user_id": "admin_1", "action_count": 25, "unique_actions": 8},
                {"user_id": "admin_2", "action_count": 18, "unique_actions": 6}
            ],
            [  # Time series
                {
                    "period": datetime.now().date(),
                    "policy_count": 3,
                    "published_count": 2
                }
            ]
        ]
        
        request = AdminAnalyticsRequest(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            granularity="day"
        )
        
        analytics = await self.admin_portal.get_admin_analytics(request)
        
        assert analytics["policy_statistics"]["total_policies"] == 25
        assert analytics["statement_statistics"]["total_statements"] == 50
        assert len(analytics["approval_activity"]) == 2
        assert len(analytics["admin_activity"]) == 2
        assert "time_series" in analytics

    async def test_get_next_version_new_policy(self):
        """Test version generation for new policy."""
        self.mock_conn.fetchval.return_value = None  # No existing version
        
        version = await self.admin_portal._get_next_version("New Policy", "Test Category")
        
        assert version == "1.0.0"

    async def test_get_next_version_existing_policy(self):
        """Test version generation for existing policy."""
        self.mock_conn.fetchval.return_value = "1.2.5"  # Existing version
        
        version = await self.admin_portal._get_next_version("Existing Policy", "Test Category")
        
        assert version == "1.2.6"

    async def test_get_next_version_invalid_format(self):
        """Test version generation with invalid existing version."""
        self.mock_conn.fetchval.return_value = "invalid_version"
        
        version = await self.admin_portal._get_next_version("Policy", "Category")
        
        assert version == "1.0.0"  # Falls back to default

    async def test_get_time_series_data(self):
        """Test time series data generation."""
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        self.mock_conn.fetch.return_value = [
            {
                "period": datetime.now().date(),
                "policy_count": 5,
                "published_count": 3
            },
            {
                "period": (datetime.now() - timedelta(days=1)).date(),
                "policy_count": 2,
                "published_count": 1
            }
        ]
        
        time_series = await self.admin_portal._get_time_series_data(
            start_date, end_date, "day"
        )
        
        assert len(time_series) == 2
        assert time_series[0]["policy_count"] == 5
        assert time_series[0]["published_count"] == 3

    async def test_log_admin_action(self):
        """Test admin action logging."""
        self.mock_conn.execute.return_value = None
        
        await self.admin_portal._log_admin_action(
            user_id="admin_user_123",
            action="upload_policy",
            resource_type="policy_document",
            resource_id="policy_123",
            details={"title": "Test Policy", "category": "Test"},
            ip_address="127.0.0.1",
            user_agent="test-agent"
        )
        
        # Verify audit log was created
        assert self.mock_conn.execute.called
        call_args = self.mock_conn.execute.call_args[0]
        assert "admin_audit_log" in call_args[0]
        assert call_args[1] == "admin_user_123"
        assert call_args[2] == "upload_policy"


class TestPolicyDocument:
    """Test policy document data structure."""

    def test_policy_document_creation(self):
        """Test policy document creation."""
        now = datetime.now()
        
        policy = PolicyDocument(
            policy_id="test_policy_123",
            title="AI Usage Policy",
            content="This is the policy content...",
            version="1.0.0",
            status=PolicyStatus.DRAFT,
            category="AI Policy",
            language="en",
            region="US",
            created_by="admin_user",
            created_at=now,
            updated_at=now,
            tags=["ai", "policy"],
            metadata={"department": "IT"}
        )
        
        assert policy.policy_id == "test_policy_123"
        assert policy.status == PolicyStatus.DRAFT
        assert len(policy.tags) == 2
        assert policy.metadata["department"] == "IT"


class TestPermanentStatement:
    """Test permanent statement data structure."""

    def test_permanent_statement_creation(self):
        """Test permanent statement creation."""
        now = datetime.now()
        
        statement = PermanentStatement(
            statement_id="test_statement_123",
            question="What is the AI usage policy?",
            answer="The AI usage policy defines...",
            category="AI Policy",
            priority=8,
            language="en",
            region="US",
            is_active=True,
            created_by="admin_user",
            created_at=now,
            updated_at=now,
            tags=["ai", "policy"],
            usage_count=25
        )
        
        assert statement.statement_id == "test_statement_123"
        assert statement.priority == 8
        assert statement.is_active is True
        assert statement.usage_count == 25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


"""
LIXIL AI Hub Use Case Radar Status Tracking Manager

This module provides comprehensive use case tracking and visualization capabilities
for the LIXIL AI Hub Platform, including status management, progress monitoring,
and interactive radar chart generation for stakeholder visibility.

Key Features:
- Use case lifecycle management (Idea → Development → Pilot → Production)
- Status tracking with automated notifications
- Interactive radar chart visualization
- Stakeholder assignment and collaboration
- ROI tracking and impact measurement
- Integration with project management tools

Author: LIXIL AI Hub Platform Team
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid

import asyncpg
from pydantic import BaseModel, Field, validator
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UseCaseStatus(str, Enum):
    """Status levels for use cases in the radar."""
    IDEA = "idea"
    PROPOSAL = "proposal"
    APPROVED = "approved"
    DEVELOPMENT = "development"
    PILOT = "pilot"
    PRODUCTION = "production"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class UseCasePriority(str, Enum):
    """Priority levels for use cases."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class UseCaseCategory(str, Enum):
    """Categories for use case classification."""
    CUSTOMER_SERVICE = "customer_service"
    OPERATIONS = "operations"
    MARKETING = "marketing"
    SALES = "sales"
    HR = "hr"
    FINANCE = "finance"
    PRODUCT_DEVELOPMENT = "product_development"
    SUPPLY_CHAIN = "supply_chain"
    QUALITY_ASSURANCE = "quality_assurance"
    RESEARCH = "research"


class ImpactLevel(str, Enum):
    """Impact levels for ROI assessment."""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    TRANSFORMATIONAL = "transformational"


@dataclass
class UseCaseMetrics:
    """Metrics for use case tracking."""
    estimated_cost: float
    actual_cost: float
    estimated_savings: float
    actual_savings: float
    estimated_timeline_weeks: int
    actual_timeline_weeks: int
    user_adoption_rate: float
    satisfaction_score: float
    roi_percentage: float


@dataclass
class Stakeholder:
    """Stakeholder information for use cases."""
    user_id: str
    name: str
    role: str
    department: str
    involvement_level: str  # "sponsor", "owner", "contributor", "reviewer"
    contact_email: str


class UseCaseRequest(BaseModel):
    """Request model for creating/updating use cases."""
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=10, max_length=2000)
    category: UseCaseCategory
    priority: UseCasePriority = UseCasePriority.MEDIUM
    business_justification: str = Field(..., min_length=10, max_length=1000)
    success_criteria: List[str] = Field(..., min_items=1)
    estimated_cost: float = Field(ge=0)
    estimated_savings: float = Field(ge=0)
    estimated_timeline_weeks: int = Field(ge=1, le=104)  # Max 2 years
    target_users: int = Field(ge=1)
    department: str = Field(..., min_length=1, max_length=100)
    region: str = Field(..., min_length=1, max_length=50)
    submitted_by: str = Field(..., min_length=1, max_length=100)
    stakeholders: List[Dict[str, str]] = Field(default_factory=list)

    @validator('success_criteria')
    def validate_success_criteria(cls, v):
        """Validate success criteria format."""
        if len(v) > 10:
            raise ValueError("Maximum 10 success criteria allowed")
        for criteria in v:
            if len(criteria) > 200:
                raise ValueError("Each success criteria must be 200 characters or less")
        return v


class UseCaseUpdateRequest(BaseModel):
    """Request model for updating use case status."""
    use_case_id: str
    status: Optional[UseCaseStatus] = None
    priority: Optional[UseCasePriority] = None
    actual_cost: Optional[float] = Field(None, ge=0)
    actual_savings: Optional[float] = Field(None, ge=0)
    actual_timeline_weeks: Optional[int] = Field(None, ge=1)
    user_adoption_rate: Optional[float] = Field(None, ge=0, le=1)
    satisfaction_score: Optional[float] = Field(None, ge=1, le=10)
    progress_notes: Optional[str] = Field(None, max_length=1000)
    updated_by: str = Field(..., min_length=1, max_length=100)


class RadarVisualizationRequest(BaseModel):
    """Request model for radar chart generation."""
    department: Optional[str] = None
    region: Optional[str] = None
    category: Optional[UseCaseCategory] = None
    status_filter: Optional[List[UseCaseStatus]] = None
    priority_filter: Optional[List[UseCasePriority]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    include_metrics: bool = True


class UseCaseRadarData(BaseModel):
    """Data structure for radar chart visualization."""
    use_case_id: str
    title: str
    category: UseCaseCategory
    status: UseCaseStatus
    priority: UseCasePriority
    progress_percentage: float
    impact_score: float
    risk_score: float
    timeline_health: float  # On track = 1.0, delayed = < 1.0
    cost_health: float  # On budget = 1.0, over budget = < 1.0
    x_position: float  # Radar chart X coordinate
    y_position: float  # Radar chart Y coordinate
    size: float  # Bubble size based on impact/cost
    color: str  # Color based on status/priority


class UseCaseDatabase:
    """
    Database manager for use case storage and retrieval.
    
    Handles all database operations for use case tracking,
    metrics storage, and historical data management.
    """

    def __init__(self, database_url: str):
        """
        Initialize database manager.
        
        Args:
            database_url: PostgreSQL connection URL
        """
        self.database_url = database_url
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
        logger.info("Use case database initialized")

    async def close(self):
        """Close database connections."""
        if self.db_pool:
            await self.db_pool.close()

    async def _create_schema(self):
        """Create database schema for use case tracking."""
        schema_sql = """
        -- Use cases table
        CREATE TABLE IF NOT EXISTS use_cases (
            use_case_id VARCHAR(64) PRIMARY KEY,
            title VARCHAR(200) NOT NULL,
            description TEXT NOT NULL,
            category VARCHAR(50) NOT NULL,
            status VARCHAR(50) NOT NULL,
            priority VARCHAR(20) NOT NULL,
            business_justification TEXT NOT NULL,
            success_criteria JSONB NOT NULL,
            department VARCHAR(100) NOT NULL,
            region VARCHAR(50) NOT NULL,
            submitted_by VARCHAR(100) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
            target_users INTEGER NOT NULL,
            is_active BOOLEAN DEFAULT TRUE
        );

        -- Use case metrics table
        CREATE TABLE IF NOT EXISTS use_case_metrics (
            use_case_id VARCHAR(64) NOT NULL,
            estimated_cost DECIMAL(12, 2) NOT NULL,
            actual_cost DECIMAL(12, 2) DEFAULT 0,
            estimated_savings DECIMAL(12, 2) NOT NULL,
            actual_savings DECIMAL(12, 2) DEFAULT 0,
            estimated_timeline_weeks INTEGER NOT NULL,
            actual_timeline_weeks INTEGER DEFAULT 0,
            user_adoption_rate DECIMAL(5, 4) DEFAULT 0,
            satisfaction_score DECIMAL(3, 2) DEFAULT 0,
            roi_percentage DECIMAL(8, 2) DEFAULT 0,
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
            FOREIGN KEY (use_case_id) REFERENCES use_cases(use_case_id)
        );

        -- Stakeholders table
        CREATE TABLE IF NOT EXISTS use_case_stakeholders (
            use_case_id VARCHAR(64) NOT NULL,
            user_id VARCHAR(64) NOT NULL,
            name VARCHAR(100) NOT NULL,
            role VARCHAR(100) NOT NULL,
            department VARCHAR(100) NOT NULL,
            involvement_level VARCHAR(20) NOT NULL,
            contact_email VARCHAR(255) NOT NULL,
            added_at TIMESTAMP WITH TIME ZONE NOT NULL,
            PRIMARY KEY (use_case_id, user_id),
            FOREIGN KEY (use_case_id) REFERENCES use_cases(use_case_id)
        );

        -- Status history table
        CREATE TABLE IF NOT EXISTS use_case_status_history (
            history_id SERIAL PRIMARY KEY,
            use_case_id VARCHAR(64) NOT NULL,
            previous_status VARCHAR(50),
            new_status VARCHAR(50) NOT NULL,
            changed_by VARCHAR(100) NOT NULL,
            change_reason TEXT,
            progress_notes TEXT,
            changed_at TIMESTAMP WITH TIME ZONE NOT NULL,
            FOREIGN KEY (use_case_id) REFERENCES use_cases(use_case_id)
        );

        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_use_cases_status ON use_cases(status);
        CREATE INDEX IF NOT EXISTS idx_use_cases_category ON use_cases(category);
        CREATE INDEX IF NOT EXISTS idx_use_cases_department ON use_cases(department);
        CREATE INDEX IF NOT EXISTS idx_use_cases_region ON use_cases(region);
        CREATE INDEX IF NOT EXISTS idx_use_cases_created_at ON use_cases(created_at);
        CREATE INDEX IF NOT EXISTS idx_stakeholders_user_id ON use_case_stakeholders(user_id);
        CREATE INDEX IF NOT EXISTS idx_status_history_use_case ON use_case_status_history(use_case_id);
        CREATE INDEX IF NOT EXISTS idx_status_history_changed_at ON use_case_status_history(changed_at);
        """
        
        async with self.db_pool.acquire() as conn:
            await conn.execute(schema_sql)

    async def create_use_case(self, request: UseCaseRequest) -> str:
        """Create a new use case."""
        use_case_id = str(uuid.uuid4())
        now = datetime.now()
        
        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                # Insert use case
                await conn.execute("""
                    INSERT INTO use_cases (
                        use_case_id, title, description, category, status, priority,
                        business_justification, success_criteria, department, region,
                        submitted_by, created_at, updated_at, target_users
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """,
                    use_case_id, request.title, request.description, request.category.value,
                    UseCaseStatus.IDEA.value, request.priority.value, request.business_justification,
                    json.dumps(request.success_criteria), request.department, request.region,
                    request.submitted_by, now, now, request.target_users
                )
                
                # Insert metrics
                await conn.execute("""
                    INSERT INTO use_case_metrics (
                        use_case_id, estimated_cost, estimated_savings, estimated_timeline_weeks, updated_at
                    ) VALUES ($1, $2, $3, $4, $5)
                """,
                    use_case_id, request.estimated_cost, request.estimated_savings,
                    request.estimated_timeline_weeks, now
                )
                
                # Insert stakeholders
                for stakeholder_data in request.stakeholders:
                    await conn.execute("""
                        INSERT INTO use_case_stakeholders (
                            use_case_id, user_id, name, role, department, involvement_level, contact_email, added_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                        use_case_id, stakeholder_data.get("user_id", str(uuid.uuid4())),
                        stakeholder_data["name"], stakeholder_data["role"],
                        stakeholder_data["department"], stakeholder_data["involvement_level"],
                        stakeholder_data["contact_email"], now
                    )
                
                # Insert initial status history
                await conn.execute("""
                    INSERT INTO use_case_status_history (
                        use_case_id, new_status, changed_by, change_reason, changed_at
                    ) VALUES ($1, $2, $3, $4, $5)
                """,
                    use_case_id, UseCaseStatus.IDEA.value, request.submitted_by,
                    "Initial use case submission", now
                )
        
        logger.info(f"Use case created: {use_case_id} by {request.submitted_by}")
        return use_case_id

    async def update_use_case(self, request: UseCaseUpdateRequest) -> bool:
        """Update use case status and metrics."""
        now = datetime.now()
        
        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                # Get current status for history
                current_status = await conn.fetchval(
                    "SELECT status FROM use_cases WHERE use_case_id = $1",
                    request.use_case_id
                )
                
                if not current_status:
                    return False
                
                # Update use case
                update_fields = ["updated_at = $2"]
                params = [request.use_case_id, now]
                param_count = 2
                
                if request.status:
                    param_count += 1
                    update_fields.append(f"status = ${param_count}")
                    params.append(request.status.value)
                
                if request.priority:
                    param_count += 1
                    update_fields.append(f"priority = ${param_count}")
                    params.append(request.priority.value)
                
                if update_fields:
                    await conn.execute(f"""
                        UPDATE use_cases SET {', '.join(update_fields)}
                        WHERE use_case_id = $1
                    """, *params)
                
                # Update metrics
                metric_updates = ["updated_at = $2"]
                metric_params = [request.use_case_id, now]
                metric_count = 2
                
                if request.actual_cost is not None:
                    metric_count += 1
                    metric_updates.append(f"actual_cost = ${metric_count}")
                    metric_params.append(request.actual_cost)
                
                if request.actual_savings is not None:
                    metric_count += 1
                    metric_updates.append(f"actual_savings = ${metric_count}")
                    metric_params.append(request.actual_savings)
                
                if request.actual_timeline_weeks is not None:
                    metric_count += 1
                    metric_updates.append(f"actual_timeline_weeks = ${metric_count}")
                    metric_params.append(request.actual_timeline_weeks)
                
                if request.user_adoption_rate is not None:
                    metric_count += 1
                    metric_updates.append(f"user_adoption_rate = ${metric_count}")
                    metric_params.append(request.user_adoption_rate)
                
                if request.satisfaction_score is not None:
                    metric_count += 1
                    metric_updates.append(f"satisfaction_score = ${metric_count}")
                    metric_params.append(request.satisfaction_score)
                
                # Calculate ROI if we have actual savings and cost
                if request.actual_savings is not None and request.actual_cost is not None:
                    roi = ((request.actual_savings - request.actual_cost) / max(request.actual_cost, 1)) * 100
                    metric_count += 1
                    metric_updates.append(f"roi_percentage = ${metric_count}")
                    metric_params.append(roi)
                
                await conn.execute(f"""
                    UPDATE use_case_metrics SET {', '.join(metric_updates)}
                    WHERE use_case_id = $1
                """, *metric_params)
                
                # Add status history if status changed
                if request.status and request.status.value != current_status:
                    await conn.execute("""
                        INSERT INTO use_case_status_history (
                            use_case_id, previous_status, new_status, changed_by, 
                            change_reason, progress_notes, changed_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                        request.use_case_id, current_status, request.status.value,
                        request.updated_by, f"Status updated to {request.status.value}",
                        request.progress_notes, now
                    )
        
        logger.info(f"Use case updated: {request.use_case_id} by {request.updated_by}")
        return True

    async def get_use_cases_for_radar(self, filters: RadarVisualizationRequest) -> List[Dict[str, Any]]:
        """Get use cases data for radar visualization."""
        conditions = ["uc.is_active = TRUE"]
        params = []
        param_count = 0
        
        if filters.department:
            param_count += 1
            conditions.append(f"uc.department = ${param_count}")
            params.append(filters.department)
        
        if filters.region:
            param_count += 1
            conditions.append(f"uc.region = ${param_count}")
            params.append(filters.region)
        
        if filters.category:
            param_count += 1
            conditions.append(f"uc.category = ${param_count}")
            params.append(filters.category.value)
        
        if filters.status_filter:
            param_count += 1
            status_list = [s.value for s in filters.status_filter]
            conditions.append(f"uc.status = ANY(${param_count})")
            params.append(status_list)
        
        if filters.priority_filter:
            param_count += 1
            priority_list = [p.value for p in filters.priority_filter]
            conditions.append(f"uc.priority = ANY(${param_count})")
            params.append(priority_list)
        
        if filters.date_from:
            param_count += 1
            conditions.append(f"uc.created_at >= ${param_count}")
            params.append(filters.date_from)
        
        if filters.date_to:
            param_count += 1
            conditions.append(f"uc.created_at <= ${param_count}")
            params.append(filters.date_to)
        
        where_clause = "WHERE " + " AND ".join(conditions)
        
        query = f"""
            SELECT 
                uc.use_case_id, uc.title, uc.category, uc.status, uc.priority,
                uc.created_at, uc.target_users, uc.department, uc.region,
                m.estimated_cost, m.actual_cost, m.estimated_savings, m.actual_savings,
                m.estimated_timeline_weeks, m.actual_timeline_weeks,
                m.user_adoption_rate, m.satisfaction_score, m.roi_percentage
            FROM use_cases uc
            LEFT JOIN use_case_metrics m ON uc.use_case_id = m.use_case_id
            {where_clause}
            ORDER BY uc.created_at DESC
        """
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            
            return [dict(row) for row in rows]


class RadarCalculator:
    """
    Calculator for radar chart positioning and metrics.
    
    Transforms use case data into radar chart coordinates
    with intelligent positioning based on multiple factors.
    """

    def __init__(self):
        """Initialize radar calculator."""
        self.status_weights = {
            UseCaseStatus.IDEA: 0.1,
            UseCaseStatus.PROPOSAL: 0.2,
            UseCaseStatus.APPROVED: 0.3,
            UseCaseStatus.DEVELOPMENT: 0.5,
            UseCaseStatus.PILOT: 0.7,
            UseCaseStatus.PRODUCTION: 1.0,
            UseCaseStatus.COMPLETED: 1.0,
            UseCaseStatus.PAUSED: 0.3,
            UseCaseStatus.CANCELLED: 0.0
        }
        
        self.priority_multipliers = {
            UseCasePriority.LOW: 0.5,
            UseCasePriority.MEDIUM: 1.0,
            UseCasePriority.HIGH: 1.5,
            UseCasePriority.CRITICAL: 2.0
        }

    def calculate_radar_positions(self, use_cases: List[Dict[str, Any]]) -> List[UseCaseRadarData]:
        """Calculate radar chart positions for use cases."""
        radar_data = []
        
        for uc in use_cases:
            # Calculate progress percentage
            progress = self._calculate_progress(uc)
            
            # Calculate impact score
            impact = self._calculate_impact_score(uc)
            
            # Calculate risk score
            risk = self._calculate_risk_score(uc)
            
            # Calculate health metrics
            timeline_health = self._calculate_timeline_health(uc)
            cost_health = self._calculate_cost_health(uc)
            
            # Calculate radar coordinates
            x_pos, y_pos = self._calculate_coordinates(progress, impact, risk)
            
            # Calculate bubble size
            size = self._calculate_bubble_size(uc, impact)
            
            # Determine color
            color = self._determine_color(uc)
            
            radar_item = UseCaseRadarData(
                use_case_id=uc["use_case_id"],
                title=uc["title"],
                category=UseCaseCategory(uc["category"]),
                status=UseCaseStatus(uc["status"]),
                priority=UseCasePriority(uc["priority"]),
                progress_percentage=progress,
                impact_score=impact,
                risk_score=risk,
                timeline_health=timeline_health,
                cost_health=cost_health,
                x_position=x_pos,
                y_position=y_pos,
                size=size,
                color=color
            )
            
            radar_data.append(radar_item)
        
        return radar_data

    def _calculate_progress(self, use_case: Dict[str, Any]) -> float:
        """Calculate progress percentage based on status and metrics."""
        status = UseCaseStatus(use_case["status"])
        base_progress = self.status_weights.get(status, 0.0)
        
        # Adjust based on actual vs estimated timeline
        if (use_case["actual_timeline_weeks"] and 
            use_case["estimated_timeline_weeks"]):
            timeline_ratio = use_case["actual_timeline_weeks"] / use_case["estimated_timeline_weeks"]
            if timeline_ratio <= 1.0:
                # On or ahead of schedule
                base_progress = min(1.0, base_progress * 1.1)
            else:
                # Behind schedule
                base_progress = max(0.0, base_progress * 0.9)
        
        return min(1.0, base_progress)

    def _calculate_impact_score(self, use_case: Dict[str, Any]) -> float:
        """Calculate impact score based on savings, users, and ROI."""
        impact_factors = []
        
        # Savings impact
        estimated_savings = use_case.get("estimated_savings", 0) or 0
        actual_savings = use_case.get("actual_savings", 0) or 0
        savings_score = min(1.0, max(estimated_savings, actual_savings) / 100000)  # Normalize to 100k
        impact_factors.append(savings_score)
        
        # User reach impact
        target_users = use_case.get("target_users", 0) or 0
        user_score = min(1.0, target_users / 1000)  # Normalize to 1000 users
        impact_factors.append(user_score)
        
        # ROI impact
        roi = use_case.get("roi_percentage", 0) or 0
        roi_score = min(1.0, max(0, roi) / 200)  # Normalize to 200% ROI
        impact_factors.append(roi_score)
        
        # Priority multiplier
        priority = UseCasePriority(use_case["priority"])
        multiplier = self.priority_multipliers.get(priority, 1.0)
        
        # Calculate weighted average
        if impact_factors:
            base_impact = sum(impact_factors) / len(impact_factors)
            return min(1.0, base_impact * multiplier)
        
        return 0.5  # Default medium impact

    def _calculate_risk_score(self, use_case: Dict[str, Any]) -> float:
        """Calculate risk score based on timeline and cost overruns."""
        risk_factors = []
        
        # Timeline risk
        if (use_case["actual_timeline_weeks"] and 
            use_case["estimated_timeline_weeks"]):
            timeline_ratio = use_case["actual_timeline_weeks"] / use_case["estimated_timeline_weeks"]
            timeline_risk = max(0, min(1.0, (timeline_ratio - 1.0) * 2))  # Risk increases with delay
            risk_factors.append(timeline_risk)
        
        # Cost risk
        if (use_case["actual_cost"] and 
            use_case["estimated_cost"]):
            cost_ratio = use_case["actual_cost"] / use_case["estimated_cost"]
            cost_risk = max(0, min(1.0, (cost_ratio - 1.0) * 2))  # Risk increases with cost overrun
            risk_factors.append(cost_risk)
        
        # Status-based risk
        status = UseCaseStatus(use_case["status"])
        status_risk = {
            UseCaseStatus.IDEA: 0.8,
            UseCaseStatus.PROPOSAL: 0.6,
            UseCaseStatus.APPROVED: 0.4,
            UseCaseStatus.DEVELOPMENT: 0.5,
            UseCaseStatus.PILOT: 0.3,
            UseCaseStatus.PRODUCTION: 0.1,
            UseCaseStatus.COMPLETED: 0.0,
            UseCaseStatus.PAUSED: 0.9,
            UseCaseStatus.CANCELLED: 1.0
        }.get(status, 0.5)
        risk_factors.append(status_risk)
        
        return sum(risk_factors) / len(risk_factors) if risk_factors else 0.5

    def _calculate_timeline_health(self, use_case: Dict[str, Any]) -> float:
        """Calculate timeline health (1.0 = on track, < 1.0 = delayed)."""
        if (use_case["actual_timeline_weeks"] and 
            use_case["estimated_timeline_weeks"]):
            ratio = use_case["estimated_timeline_weeks"] / use_case["actual_timeline_weeks"]
            return min(1.0, ratio)
        return 1.0  # Assume on track if no data

    def _calculate_cost_health(self, use_case: Dict[str, Any]) -> float:
        """Calculate cost health (1.0 = on budget, < 1.0 = over budget)."""
        if (use_case["actual_cost"] and 
            use_case["estimated_cost"]):
            ratio = use_case["estimated_cost"] / use_case["actual_cost"]
            return min(1.0, ratio)
        return 1.0  # Assume on budget if no data

    def _calculate_coordinates(self, progress: float, impact: float, risk: float) -> Tuple[float, float]:
        """Calculate X,Y coordinates for radar chart."""
        # X-axis: Progress vs Risk (progress - risk, normalized)
        x = (progress - risk + 1) / 2  # Normalize to 0-1
        
        # Y-axis: Impact (higher impact = higher Y)
        y = impact
        
        # Add some randomization to prevent overlap
        import random
        x += random.uniform(-0.05, 0.05)
        y += random.uniform(-0.05, 0.05)
        
        # Ensure coordinates stay within bounds
        x = max(0.05, min(0.95, x))
        y = max(0.05, min(0.95, y))
        
        return x, y

    def _calculate_bubble_size(self, use_case: Dict[str, Any], impact: float) -> float:
        """Calculate bubble size based on impact and cost."""
        # Base size on impact
        base_size = 20 + (impact * 30)  # 20-50 pixel range
        
        # Adjust for estimated cost (larger projects = larger bubbles)
        estimated_cost = use_case.get("estimated_cost", 0) or 0
        cost_multiplier = min(2.0, 1.0 + (estimated_cost / 50000))  # Up to 2x for 50k+ projects
        
        return base_size * cost_multiplier

    def _determine_color(self, use_case: Dict[str, Any]) -> str:
        """Determine color based on status and priority."""
        status = UseCaseStatus(use_case["status"])
        priority = UseCasePriority(use_case["priority"])
        
        # Status-based colors
        status_colors = {
            UseCaseStatus.IDEA: "#E3F2FD",  # Light blue
            UseCaseStatus.PROPOSAL: "#BBDEFB",  # Blue
            UseCaseStatus.APPROVED: "#90CAF9",  # Medium blue
            UseCaseStatus.DEVELOPMENT: "#FFF3E0",  # Light orange
            UseCaseStatus.PILOT: "#FFE0B2",  # Orange
            UseCaseStatus.PRODUCTION: "#C8E6C9",  # Light green
            UseCaseStatus.COMPLETED: "#4CAF50",  # Green
            UseCaseStatus.PAUSED: "#FFECB3",  # Yellow
            UseCaseStatus.CANCELLED: "#FFCDD2"  # Light red
        }
        
        base_color = status_colors.get(status, "#E0E0E0")
        
        # Adjust intensity based on priority
        if priority == UseCasePriority.CRITICAL:
            # Make colors more intense for critical items
            return base_color.replace("E0", "C0").replace("F0", "E0")
        
        return base_color


class UseCaseRadarManager:
    """
    Main manager for use case radar functionality.
    
    Orchestrates use case tracking, status management, and
    radar chart generation for the LIXIL AI Hub Platform.
    """

    def __init__(self, database_url: str):
        """
        Initialize use case radar manager.
        
        Args:
            database_url: PostgreSQL database connection URL
        """
        self.database = UseCaseDatabase(database_url)
        self.calculator = RadarCalculator()

    async def initialize(self):
        """Initialize the radar manager."""
        await self.database.initialize()
        logger.info("Use case radar manager initialized")

    async def close(self):
        """Close database connections."""
        await self.database.close()

    async def create_use_case(self, request: UseCaseRequest) -> str:
        """Create a new use case."""
        return await self.database.create_use_case(request)

    async def update_use_case(self, request: UseCaseUpdateRequest) -> bool:
        """Update use case status and metrics."""
        return await self.database.update_use_case(request)

    async def generate_radar_data(self, filters: RadarVisualizationRequest) -> List[UseCaseRadarData]:
        """Generate radar chart data."""
        use_cases = await self.database.get_use_cases_for_radar(filters)
        return self.calculator.calculate_radar_positions(use_cases)

    async def get_dashboard_stats(self, department: Optional[str] = None, 
                                region: Optional[str] = None) -> Dict[str, Any]:
        """Get dashboard statistics for use cases."""
        filters = RadarVisualizationRequest(department=department, region=region)
        use_cases = await self.database.get_use_cases_for_radar(filters)
        
        if not use_cases:
            return {
                "total_use_cases": 0,
                "status_breakdown": {},
                "category_breakdown": {},
                "total_estimated_savings": 0,
                "total_actual_savings": 0,
                "average_roi": 0,
                "completion_rate": 0
            }
        
        # Calculate statistics
        status_counts = {}
        category_counts = {}
        total_estimated_savings = 0
        total_actual_savings = 0
        roi_values = []
        completed_count = 0
        
        for uc in use_cases:
            # Status breakdown
            status = uc["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Category breakdown
            category = uc["category"]
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Financial metrics
            total_estimated_savings += uc.get("estimated_savings", 0) or 0
            total_actual_savings += uc.get("actual_savings", 0) or 0
            
            if uc.get("roi_percentage"):
                roi_values.append(uc["roi_percentage"])
            
            # Completion tracking
            if uc["status"] in [UseCaseStatus.PRODUCTION.value, UseCaseStatus.COMPLETED.value]:
                completed_count += 1
        
        return {
            "total_use_cases": len(use_cases),
            "status_breakdown": status_counts,
            "category_breakdown": category_counts,
            "total_estimated_savings": total_estimated_savings,
            "total_actual_savings": total_actual_savings,
            "average_roi": sum(roi_values) / len(roi_values) if roi_values else 0,
            "completion_rate": completed_count / len(use_cases) if use_cases else 0
        }


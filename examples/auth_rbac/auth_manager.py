"""
LIXIL AI Hub Role-Based Access Control (RBAC) Authentication Manager

This module provides comprehensive authentication and authorization capabilities
for the LIXIL AI Hub Platform, including user management, role-based permissions,
session handling, and integration with enterprise identity providers.

Key Features:
- JWT-based authentication with refresh tokens
- Role-based access control with granular permissions
- Multi-factor authentication (MFA) support
- Integration with LDAP/Active Directory
- Session management and security policies
- Audit logging for compliance

Author: LIXIL AI Hub Platform Team
Version: 1.0.0
"""

import asyncio
import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import os

import jwt
import bcrypt
import asyncpg
from pydantic import BaseModel, Field, validator, EmailStr
import pyotp
import qrcode
from io import BytesIO
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    """User roles in the LIXIL AI Hub system."""
    SUPER_ADMIN = "super_admin"
    AI_COUNCIL_MEMBER = "ai_council_member"
    CONTENT_MANAGER = "content_manager"
    POLICY_REVIEWER = "policy_reviewer"
    REGIONAL_ADMIN = "regional_admin"
    DEPARTMENT_ADMIN = "department_admin"
    STANDARD_USER = "standard_user"
    READ_ONLY_USER = "read_only_user"


class Permission(str, Enum):
    """System permissions for granular access control."""
    # Content Management
    CREATE_CONTENT = "create_content"
    UPDATE_CONTENT = "update_content"
    DELETE_CONTENT = "delete_content"
    APPROVE_CONTENT = "approve_content"
    PUBLISH_CONTENT = "publish_content"
    
    # User Management
    CREATE_USER = "create_user"
    UPDATE_USER = "update_user"
    DELETE_USER = "delete_user"
    ASSIGN_ROLES = "assign_roles"
    
    # System Administration
    MANAGE_SYSTEM = "manage_system"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    MANAGE_INTEGRATIONS = "manage_integrations"
    
    # AI Features
    ACCESS_AI_CHAT = "access_ai_chat"
    UPLOAD_DOCUMENTS = "upload_documents"
    MANAGE_KNOWLEDGE_BASE = "manage_knowledge_base"
    
    # Regional/Department
    MANAGE_REGION = "manage_region"
    MANAGE_DEPARTMENT = "manage_department"


class AuthenticationMethod(str, Enum):
    """Supported authentication methods."""
    PASSWORD = "password"
    LDAP = "ldap"
    SSO = "sso"
    MFA_TOTP = "mfa_totp"
    API_KEY = "api_key"


@dataclass
class UserProfile:
    """User profile information."""
    user_id: str
    email: str
    first_name: str
    last_name: str
    department: Optional[str] = None
    region: Optional[str] = None
    language: str = "en"
    timezone: str = "UTC"
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    email_verified: bool = False
    mfa_enabled: bool = False


@dataclass
class UserSession:
    """User session information."""
    session_id: str
    user_id: str
    roles: List[UserRole]
    permissions: Set[Permission]
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True


class LoginRequest(BaseModel):
    """Request model for user login."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    mfa_code: Optional[str] = Field(None, regex="^[0-9]{6}$")
    remember_me: bool = False

    @validator('password')
    def validate_password_strength(cls, v):
        """Validate password meets security requirements."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserRegistrationRequest(BaseModel):
    """Request model for user registration."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    department: Optional[str] = Field(None, max_length=100)
    region: Optional[str] = Field(None, max_length=50)
    language: str = Field(default="en", regex="^[a-z]{2}$")

    @validator('password')
    def validate_password_strength(cls, v):
        """Validate password meets security requirements."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class RoleAssignmentRequest(BaseModel):
    """Request model for role assignment."""
    user_id: str
    roles: List[UserRole]
    assigned_by: str
    reason: str = Field(..., min_length=1, max_length=200)


class AuthenticationResult(BaseModel):
    """Result of authentication attempt."""
    success: bool
    user_id: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_in: Optional[int] = None
    requires_mfa: bool = False
    error_message: Optional[str] = None
    session_id: Optional[str] = None


class PasswordManager:
    """
    Secure password management with bcrypt hashing.
    
    Provides password hashing, verification, and strength validation
    following security best practices.
    """

    def __init__(self, rounds: int = 12):
        """
        Initialize password manager.
        
        Args:
            rounds: bcrypt rounds for hashing (default: 12)
        """
        self.rounds = rounds

    def hash_password(self, password: str) -> str:
        """
        Hash a password using bcrypt.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password string
        """
        salt = bcrypt.gensalt(rounds=self.rounds)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    def verify_password(self, password: str, hashed: str) -> bool:
        """
        Verify a password against its hash.
        
        Args:
            password: Plain text password
            hashed: Hashed password
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False

    def generate_secure_password(self, length: int = 16) -> str:
        """
        Generate a secure random password.
        
        Args:
            length: Password length (default: 16)
            
        Returns:
            Secure random password
        """
        import string
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        password = ''.join(secrets.choice(alphabet) for _ in range(length))
        
        # Ensure password meets requirements
        if (any(c.isupper() for c in password) and 
            any(c.islower() for c in password) and 
            any(c.isdigit() for c in password)):
            return password
        else:
            # Regenerate if requirements not met
            return self.generate_secure_password(length)


class MFAManager:
    """
    Multi-Factor Authentication manager using TOTP.
    
    Provides TOTP-based MFA setup, verification, and backup codes
    for enhanced security.
    """

    def __init__(self, issuer_name: str = "LIXIL AI Hub"):
        """
        Initialize MFA manager.
        
        Args:
            issuer_name: Name of the issuing organization
        """
        self.issuer_name = issuer_name

    def generate_secret(self) -> str:
        """Generate a new TOTP secret."""
        return pyotp.random_base32()

    def generate_qr_code(self, user_email: str, secret: str) -> str:
        """
        Generate QR code for TOTP setup.
        
        Args:
            user_email: User's email address
            secret: TOTP secret
            
        Returns:
            Base64-encoded QR code image
        """
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_email,
            issuer_name=self.issuer_name
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode()

    def verify_totp(self, secret: str, token: str, window: int = 1) -> bool:
        """
        Verify TOTP token.
        
        Args:
            secret: User's TOTP secret
            token: TOTP token to verify
            window: Time window for verification (default: 1)
            
        Returns:
            True if token is valid, False otherwise
        """
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=window)
        except Exception as e:
            logger.error(f"TOTP verification error: {e}")
            return False

    def generate_backup_codes(self, count: int = 10) -> List[str]:
        """
        Generate backup codes for MFA recovery.
        
        Args:
            count: Number of backup codes to generate
            
        Returns:
            List of backup codes
        """
        codes = []
        for _ in range(count):
            code = ''.join(secrets.choice('0123456789') for _ in range(8))
            codes.append(f"{code[:4]}-{code[4:]}")
        return codes


class JWTManager:
    """
    JWT token management for authentication.
    
    Handles JWT token creation, validation, and refresh token management
    with configurable expiration and security settings.
    """

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        """
        Initialize JWT manager.
        
        Args:
            secret_key: Secret key for JWT signing
            algorithm: JWT algorithm (default: HS256)
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire = timedelta(hours=1)
        self.refresh_token_expire = timedelta(days=30)

    def create_access_token(self, user_id: str, roles: List[UserRole], 
                          permissions: Set[Permission]) -> str:
        """
        Create JWT access token.
        
        Args:
            user_id: User identifier
            roles: User roles
            permissions: User permissions
            
        Returns:
            JWT access token
        """
        now = datetime.utcnow()
        payload = {
            "sub": user_id,
            "iat": now,
            "exp": now + self.access_token_expire,
            "type": "access",
            "roles": [role.value for role in roles],
            "permissions": [perm.value for perm in permissions]
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(self, user_id: str) -> str:
        """
        Create JWT refresh token.
        
        Args:
            user_id: User identifier
            
        Returns:
            JWT refresh token
        """
        now = datetime.utcnow()
        payload = {
            "sub": user_id,
            "iat": now,
            "exp": now + self.refresh_token_expire,
            "type": "refresh"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Decoded token payload or None if invalid
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None

    def refresh_access_token(self, refresh_token: str, user_roles: List[UserRole],
                           user_permissions: Set[Permission]) -> Optional[str]:
        """
        Create new access token from refresh token.
        
        Args:
            refresh_token: Valid refresh token
            user_roles: Current user roles
            user_permissions: Current user permissions
            
        Returns:
            New access token or None if refresh token invalid
        """
        payload = self.verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            return None
        
        user_id = payload.get("sub")
        if not user_id:
            return None
        
        return self.create_access_token(user_id, user_roles, user_permissions)


class RolePermissionManager:
    """
    Role and permission management system.
    
    Defines role hierarchies, permission mappings, and provides
    authorization checking capabilities.
    """

    def __init__(self):
        """Initialize role permission manager."""
        self.role_permissions = self._initialize_role_permissions()
        self.role_hierarchy = self._initialize_role_hierarchy()

    def _initialize_role_permissions(self) -> Dict[UserRole, Set[Permission]]:
        """Initialize role to permission mappings."""
        return {
            UserRole.SUPER_ADMIN: {
                Permission.CREATE_CONTENT, Permission.UPDATE_CONTENT, Permission.DELETE_CONTENT,
                Permission.APPROVE_CONTENT, Permission.PUBLISH_CONTENT, Permission.CREATE_USER,
                Permission.UPDATE_USER, Permission.DELETE_USER, Permission.ASSIGN_ROLES,
                Permission.MANAGE_SYSTEM, Permission.VIEW_AUDIT_LOGS, Permission.MANAGE_INTEGRATIONS,
                Permission.ACCESS_AI_CHAT, Permission.UPLOAD_DOCUMENTS, Permission.MANAGE_KNOWLEDGE_BASE,
                Permission.MANAGE_REGION, Permission.MANAGE_DEPARTMENT
            },
            UserRole.AI_COUNCIL_MEMBER: {
                Permission.CREATE_CONTENT, Permission.UPDATE_CONTENT, Permission.DELETE_CONTENT,
                Permission.APPROVE_CONTENT, Permission.PUBLISH_CONTENT, Permission.ACCESS_AI_CHAT,
                Permission.UPLOAD_DOCUMENTS, Permission.MANAGE_KNOWLEDGE_BASE, Permission.VIEW_AUDIT_LOGS
            },
            UserRole.CONTENT_MANAGER: {
                Permission.CREATE_CONTENT, Permission.UPDATE_CONTENT, Permission.DELETE_CONTENT,
                Permission.ACCESS_AI_CHAT, Permission.UPLOAD_DOCUMENTS
            },
            UserRole.POLICY_REVIEWER: {
                Permission.UPDATE_CONTENT, Permission.APPROVE_CONTENT, Permission.ACCESS_AI_CHAT
            },
            UserRole.REGIONAL_ADMIN: {
                Permission.CREATE_CONTENT, Permission.UPDATE_CONTENT, Permission.APPROVE_CONTENT,
                Permission.CREATE_USER, Permission.UPDATE_USER, Permission.MANAGE_REGION,
                Permission.ACCESS_AI_CHAT, Permission.UPLOAD_DOCUMENTS
            },
            UserRole.DEPARTMENT_ADMIN: {
                Permission.CREATE_CONTENT, Permission.UPDATE_CONTENT, Permission.CREATE_USER,
                Permission.UPDATE_USER, Permission.MANAGE_DEPARTMENT, Permission.ACCESS_AI_CHAT,
                Permission.UPLOAD_DOCUMENTS
            },
            UserRole.STANDARD_USER: {
                Permission.ACCESS_AI_CHAT, Permission.UPLOAD_DOCUMENTS
            },
            UserRole.READ_ONLY_USER: {
                Permission.ACCESS_AI_CHAT
            }
        }

    def _initialize_role_hierarchy(self) -> Dict[UserRole, List[UserRole]]:
        """Initialize role hierarchy (roles that inherit from others)."""
        return {
            UserRole.SUPER_ADMIN: [UserRole.AI_COUNCIL_MEMBER, UserRole.REGIONAL_ADMIN],
            UserRole.AI_COUNCIL_MEMBER: [UserRole.CONTENT_MANAGER, UserRole.POLICY_REVIEWER],
            UserRole.REGIONAL_ADMIN: [UserRole.DEPARTMENT_ADMIN],
            UserRole.DEPARTMENT_ADMIN: [UserRole.STANDARD_USER],
            UserRole.CONTENT_MANAGER: [UserRole.STANDARD_USER],
            UserRole.POLICY_REVIEWER: [UserRole.READ_ONLY_USER],
            UserRole.STANDARD_USER: [UserRole.READ_ONLY_USER]
        }

    def get_user_permissions(self, roles: List[UserRole]) -> Set[Permission]:
        """
        Get all permissions for a user based on their roles.
        
        Args:
            roles: List of user roles
            
        Returns:
            Set of all permissions
        """
        all_permissions = set()
        
        for role in roles:
            # Add direct permissions
            if role in self.role_permissions:
                all_permissions.update(self.role_permissions[role])
            
            # Add inherited permissions
            inherited_roles = self._get_inherited_roles(role)
            for inherited_role in inherited_roles:
                if inherited_role in self.role_permissions:
                    all_permissions.update(self.role_permissions[inherited_role])
        
        return all_permissions

    def _get_inherited_roles(self, role: UserRole) -> List[UserRole]:
        """Get all roles inherited by a given role."""
        inherited = []
        if role in self.role_hierarchy:
            for inherited_role in self.role_hierarchy[role]:
                inherited.append(inherited_role)
                inherited.extend(self._get_inherited_roles(inherited_role))
        return inherited

    def has_permission(self, user_roles: List[UserRole], required_permission: Permission) -> bool:
        """
        Check if user has required permission.
        
        Args:
            user_roles: User's roles
            required_permission: Required permission
            
        Returns:
            True if user has permission, False otherwise
        """
        user_permissions = self.get_user_permissions(user_roles)
        return required_permission in user_permissions

    def can_assign_role(self, assigner_roles: List[UserRole], target_role: UserRole) -> bool:
        """
        Check if user can assign a specific role.
        
        Args:
            assigner_roles: Roles of the user doing the assignment
            target_role: Role being assigned
            
        Returns:
            True if assignment is allowed, False otherwise
        """
        # Super admin can assign any role
        if UserRole.SUPER_ADMIN in assigner_roles:
            return True
        
        # AI Council members can assign content and user roles
        if UserRole.AI_COUNCIL_MEMBER in assigner_roles:
            allowed_roles = {
                UserRole.CONTENT_MANAGER, UserRole.POLICY_REVIEWER,
                UserRole.STANDARD_USER, UserRole.READ_ONLY_USER
            }
            return target_role in allowed_roles
        
        # Regional admins can assign department and user roles
        if UserRole.REGIONAL_ADMIN in assigner_roles:
            allowed_roles = {
                UserRole.DEPARTMENT_ADMIN, UserRole.STANDARD_USER, UserRole.READ_ONLY_USER
            }
            return target_role in allowed_roles
        
        return False


class AuthenticationManager:
    """
    Main authentication manager for the LIXIL AI Hub Platform.
    
    Orchestrates user authentication, session management, and authorization
    with support for multiple authentication methods and security policies.
    """

    def __init__(self, database_url: str, jwt_secret: str):
        """
        Initialize authentication manager.
        
        Args:
            database_url: PostgreSQL database connection URL
            jwt_secret: Secret key for JWT token signing
        """
        self.database_url = database_url
        self.db_pool = None
        self.password_manager = PasswordManager()
        self.mfa_manager = MFAManager()
        self.jwt_manager = JWTManager(jwt_secret)
        self.role_manager = RolePermissionManager()
        self.active_sessions: Dict[str, UserSession] = {}

    async def initialize(self):
        """Initialize database connection and schema."""
        self.db_pool = await asyncpg.create_pool(
            self.database_url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        
        await self._create_auth_schema()
        logger.info("Authentication manager initialized")

    async def close(self):
        """Close database connections."""
        if self.db_pool:
            await self.db_pool.close()

    async def _create_auth_schema(self):
        """Create authentication database schema."""
        schema_sql = """
        -- Users table
        CREATE TABLE IF NOT EXISTS users (
            user_id VARCHAR(64) PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            first_name VARCHAR(50) NOT NULL,
            last_name VARCHAR(50) NOT NULL,
            department VARCHAR(100),
            region VARCHAR(50),
            language VARCHAR(10) DEFAULT 'en',
            timezone VARCHAR(50) DEFAULT 'UTC',
            created_at TIMESTAMP WITH TIME ZONE NOT NULL,
            last_login TIMESTAMP WITH TIME ZONE,
            is_active BOOLEAN DEFAULT TRUE,
            email_verified BOOLEAN DEFAULT FALSE,
            mfa_enabled BOOLEAN DEFAULT FALSE,
            mfa_secret VARCHAR(32),
            failed_login_attempts INTEGER DEFAULT 0,
            locked_until TIMESTAMP WITH TIME ZONE
        );

        -- User roles table
        CREATE TABLE IF NOT EXISTS user_roles (
            user_id VARCHAR(64) NOT NULL,
            role VARCHAR(50) NOT NULL,
            assigned_by VARCHAR(64) NOT NULL,
            assigned_at TIMESTAMP WITH TIME ZONE NOT NULL,
            reason TEXT,
            PRIMARY KEY (user_id, role),
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );

        -- Sessions table
        CREATE TABLE IF NOT EXISTS user_sessions (
            session_id VARCHAR(64) PRIMARY KEY,
            user_id VARCHAR(64) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL,
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
            ip_address INET,
            user_agent TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );

        -- Audit log table
        CREATE TABLE IF NOT EXISTS auth_audit_log (
            log_id SERIAL PRIMARY KEY,
            user_id VARCHAR(64),
            action VARCHAR(100) NOT NULL,
            details JSONB,
            ip_address INET,
            user_agent TEXT,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            success BOOLEAN NOT NULL
        );

        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
        CREATE INDEX IF NOT EXISTS idx_user_roles_user_id ON user_roles(user_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON user_sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON user_sessions(expires_at);
        CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON auth_audit_log(user_id);
        CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON auth_audit_log(timestamp);
        """
        
        async with self.db_pool.acquire() as conn:
            await conn.execute(schema_sql)

    async def register_user(self, request: UserRegistrationRequest) -> str:
        """
        Register a new user.
        
        Args:
            request: User registration request
            
        Returns:
            User ID of created user
            
        Raises:
            ValueError: If email already exists
        """
        # Check if email already exists
        async with self.db_pool.acquire() as conn:
            existing = await conn.fetchval(
                "SELECT user_id FROM users WHERE email = $1", request.email
            )
            if existing:
                raise ValueError("Email already registered")
        
        # Generate user ID and hash password
        user_id = self._generate_user_id(request.email)
        password_hash = self.password_manager.hash_password(request.password)
        
        # Create user record
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO users (
                    user_id, email, password_hash, first_name, last_name,
                    department, region, language, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
                user_id, request.email, password_hash, request.first_name,
                request.last_name, request.department, request.region,
                request.language, datetime.now()
            )
            
            # Assign default role
            await conn.execute("""
                INSERT INTO user_roles (user_id, role, assigned_by, assigned_at, reason)
                VALUES ($1, $2, $3, $4, $5)
            """,
                user_id, UserRole.STANDARD_USER.value, "system",
                datetime.now(), "Default role assignment"
            )
        
        await self._log_audit_event(user_id, "user_registered", {"email": request.email})
        logger.info(f"User registered: {request.email}")
        return user_id

    async def authenticate_user(self, request: LoginRequest, ip_address: str,
                              user_agent: str) -> AuthenticationResult:
        """
        Authenticate user login.
        
        Args:
            request: Login request
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Authentication result
        """
        try:
            # Get user by email
            async with self.db_pool.acquire() as conn:
                user_row = await conn.fetchrow("""
                    SELECT user_id, password_hash, is_active, mfa_enabled, mfa_secret,
                           failed_login_attempts, locked_until
                    FROM users WHERE email = $1
                """, request.email)
                
                if not user_row:
                    await self._log_audit_event(None, "login_failed", 
                                              {"email": request.email, "reason": "user_not_found"},
                                              ip_address, user_agent, False)
                    return AuthenticationResult(success=False, error_message="Invalid credentials")
                
                # Check if account is locked
                if (user_row['locked_until'] and 
                    user_row['locked_until'] > datetime.now()):
                    await self._log_audit_event(user_row['user_id'], "login_failed",
                                              {"reason": "account_locked"}, ip_address, user_agent, False)
                    return AuthenticationResult(success=False, error_message="Account temporarily locked")
                
                # Check if account is active
                if not user_row['is_active']:
                    await self._log_audit_event(user_row['user_id'], "login_failed",
                                              {"reason": "account_inactive"}, ip_address, user_agent, False)
                    return AuthenticationResult(success=False, error_message="Account inactive")
                
                # Verify password
                if not self.password_manager.verify_password(request.password, user_row['password_hash']):
                    await self._handle_failed_login(user_row['user_id'], ip_address, user_agent)
                    return AuthenticationResult(success=False, error_message="Invalid credentials")
                
                # Check MFA if enabled
                if user_row['mfa_enabled']:
                    if not request.mfa_code:
                        return AuthenticationResult(success=False, requires_mfa=True)
                    
                    if not self.mfa_manager.verify_totp(user_row['mfa_secret'], request.mfa_code):
                        await self._log_audit_event(user_row['user_id'], "login_failed",
                                                  {"reason": "invalid_mfa"}, ip_address, user_agent, False)
                        return AuthenticationResult(success=False, error_message="Invalid MFA code")
                
                # Reset failed login attempts
                await conn.execute("""
                    UPDATE users SET failed_login_attempts = 0, locked_until = NULL, last_login = $1
                    WHERE user_id = $2
                """, datetime.now(), user_row['user_id'])
                
                # Get user roles and permissions
                user_roles = await self._get_user_roles(user_row['user_id'])
                user_permissions = self.role_manager.get_user_permissions(user_roles)
                
                # Create session
                session_id = self._generate_session_id()
                expires_at = datetime.now() + (timedelta(days=30) if request.remember_me else timedelta(hours=8))
                
                session = UserSession(
                    session_id=session_id,
                    user_id=user_row['user_id'],
                    roles=user_roles,
                    permissions=user_permissions,
                    created_at=datetime.now(),
                    expires_at=expires_at,
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                
                # Store session
                await self._store_session(session)
                self.active_sessions[session_id] = session
                
                # Create tokens
                access_token = self.jwt_manager.create_access_token(
                    user_row['user_id'], user_roles, user_permissions
                )
                refresh_token = self.jwt_manager.create_refresh_token(user_row['user_id'])
                
                await self._log_audit_event(user_row['user_id'], "login_success",
                                          {"session_id": session_id}, ip_address, user_agent, True)
                
                return AuthenticationResult(
                    success=True,
                    user_id=user_row['user_id'],
                    access_token=access_token,
                    refresh_token=refresh_token,
                    expires_in=3600,  # 1 hour
                    session_id=session_id
                )
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return AuthenticationResult(success=False, error_message="Authentication failed")

    def _generate_user_id(self, email: str) -> str:
        """Generate unique user ID."""
        timestamp = datetime.now().isoformat()
        user_string = f"{email}_{timestamp}"
        return hashlib.md5(user_string.encode()).hexdigest()

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        return secrets.token_urlsafe(32)

    async def _get_user_roles(self, user_id: str) -> List[UserRole]:
        """Get user roles from database."""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT role FROM user_roles WHERE user_id = $1", user_id
            )
            return [UserRole(row['role']) for row in rows]

    async def _store_session(self, session: UserSession):
        """Store session in database."""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO user_sessions (
                    session_id, user_id, created_at, expires_at, ip_address, user_agent
                ) VALUES ($1, $2, $3, $4, $5, $6)
            """,
                session.session_id, session.user_id, session.created_at,
                session.expires_at, session.ip_address, session.user_agent
            )

    async def _handle_failed_login(self, user_id: str, ip_address: str, user_agent: str):
        """Handle failed login attempt."""
        async with self.db_pool.acquire() as conn:
            # Increment failed attempts
            result = await conn.fetchrow("""
                UPDATE users SET failed_login_attempts = failed_login_attempts + 1
                WHERE user_id = $1
                RETURNING failed_login_attempts
            """, user_id)
            
            failed_attempts = result['failed_login_attempts']
            
            # Lock account if too many failures
            if failed_attempts >= 5:
                lock_until = datetime.now() + timedelta(minutes=30)
                await conn.execute("""
                    UPDATE users SET locked_until = $1 WHERE user_id = $2
                """, lock_until, user_id)
                
                await self._log_audit_event(user_id, "account_locked",
                                          {"failed_attempts": failed_attempts},
                                          ip_address, user_agent, False)
            else:
                await self._log_audit_event(user_id, "login_failed",
                                          {"reason": "invalid_password", "attempts": failed_attempts},
                                          ip_address, user_agent, False)

    async def _log_audit_event(self, user_id: Optional[str], action: str, details: Dict[str, Any],
                             ip_address: Optional[str] = None, user_agent: Optional[str] = None,
                             success: bool = True):
        """Log audit event."""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO auth_audit_log (user_id, action, details, ip_address, user_agent, timestamp, success)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
                user_id, action, json.dumps(details), ip_address, user_agent, datetime.now(), success
            )


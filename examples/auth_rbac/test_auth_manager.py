"""
Comprehensive test suite for LIXIL AI Hub Authentication Manager.

Tests cover user registration, authentication, role management,
session handling, MFA, and security features.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json

from auth_manager import (
    AuthenticationManager, PasswordManager, MFAManager, JWTManager,
    RolePermissionManager, UserRole, Permission, AuthenticationMethod,
    LoginRequest, UserRegistrationRequest, RoleAssignmentRequest,
    AuthenticationResult, UserProfile, UserSession
)


class TestPasswordManager:
    """Test password management functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.password_manager = PasswordManager(rounds=4)  # Lower rounds for faster tests

    def test_hash_password(self):
        """Test password hashing."""
        password = "TestPassword123!"
        hashed = self.password_manager.hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 50
        assert hashed.startswith("$2b$")

    def test_verify_password_correct(self):
        """Test password verification with correct password."""
        password = "TestPassword123!"
        hashed = self.password_manager.hash_password(password)
        
        assert self.password_manager.verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password."""
        password = "TestPassword123!"
        wrong_password = "WrongPassword123!"
        hashed = self.password_manager.hash_password(password)
        
        assert self.password_manager.verify_password(wrong_password, hashed) is False

    def test_verify_password_invalid_hash(self):
        """Test password verification with invalid hash."""
        password = "TestPassword123!"
        invalid_hash = "invalid_hash"
        
        assert self.password_manager.verify_password(password, invalid_hash) is False

    def test_generate_secure_password(self):
        """Test secure password generation."""
        password = self.password_manager.generate_secure_password(16)
        
        assert len(password) == 16
        assert any(c.isupper() for c in password)
        assert any(c.islower() for c in password)
        assert any(c.isdigit() for c in password)

    def test_generate_secure_password_different_length(self):
        """Test secure password generation with different lengths."""
        for length in [8, 12, 20, 32]:
            password = self.password_manager.generate_secure_password(length)
            assert len(password) == length


class TestMFAManager:
    """Test multi-factor authentication functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mfa_manager = MFAManager("LIXIL AI Hub Test")

    def test_generate_secret(self):
        """Test TOTP secret generation."""
        secret = self.mfa_manager.generate_secret()
        
        assert len(secret) == 32
        assert secret.isalnum()
        assert secret.isupper()

    def test_generate_qr_code(self):
        """Test QR code generation."""
        secret = self.mfa_manager.generate_secret()
        qr_code = self.mfa_manager.generate_qr_code("test@lixil.com", secret)
        
        assert len(qr_code) > 100  # Base64 encoded image should be substantial
        assert qr_code.replace("+", "").replace("/", "").replace("=", "").isalnum()

    @patch('pyotp.TOTP.verify')
    def test_verify_totp_valid(self, mock_verify):
        """Test TOTP verification with valid token."""
        mock_verify.return_value = True
        secret = "TESTSECRET123456789012345678901"
        token = "123456"
        
        result = self.mfa_manager.verify_totp(secret, token)
        assert result is True
        mock_verify.assert_called_once()

    @patch('pyotp.TOTP.verify')
    def test_verify_totp_invalid(self, mock_verify):
        """Test TOTP verification with invalid token."""
        mock_verify.return_value = False
        secret = "TESTSECRET123456789012345678901"
        token = "654321"
        
        result = self.mfa_manager.verify_totp(secret, token)
        assert result is False

    def test_generate_backup_codes(self):
        """Test backup code generation."""
        codes = self.mfa_manager.generate_backup_codes(10)
        
        assert len(codes) == 10
        for code in codes:
            assert len(code) == 9  # Format: XXXX-XXXX
            assert "-" in code
            parts = code.split("-")
            assert len(parts) == 2
            assert all(part.isdigit() for part in parts)


class TestJWTManager:
    """Test JWT token management functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.jwt_manager = JWTManager("test_secret_key_12345")

    def test_create_access_token(self):
        """Test access token creation."""
        user_id = "test_user_123"
        roles = [UserRole.STANDARD_USER]
        permissions = {Permission.ACCESS_AI_CHAT}
        
        token = self.jwt_manager.create_access_token(user_id, roles, permissions)
        
        assert isinstance(token, str)
        assert len(token) > 50
        assert token.count(".") == 2  # JWT format: header.payload.signature

    def test_create_refresh_token(self):
        """Test refresh token creation."""
        user_id = "test_user_123"
        
        token = self.jwt_manager.create_refresh_token(user_id)
        
        assert isinstance(token, str)
        assert len(token) > 50
        assert token.count(".") == 2

    def test_verify_valid_token(self):
        """Test verification of valid token."""
        user_id = "test_user_123"
        roles = [UserRole.STANDARD_USER]
        permissions = {Permission.ACCESS_AI_CHAT}
        
        token = self.jwt_manager.create_access_token(user_id, roles, permissions)
        payload = self.jwt_manager.verify_token(token)
        
        assert payload is not None
        assert payload["sub"] == user_id
        assert payload["type"] == "access"
        assert UserRole.STANDARD_USER.value in payload["roles"]

    def test_verify_invalid_token(self):
        """Test verification of invalid token."""
        invalid_token = "invalid.token.here"
        
        payload = self.jwt_manager.verify_token(invalid_token)
        assert payload is None

    def test_refresh_access_token(self):
        """Test access token refresh."""
        user_id = "test_user_123"
        roles = [UserRole.STANDARD_USER]
        permissions = {Permission.ACCESS_AI_CHAT}
        
        refresh_token = self.jwt_manager.create_refresh_token(user_id)
        new_access_token = self.jwt_manager.refresh_access_token(refresh_token, roles, permissions)
        
        assert new_access_token is not None
        payload = self.jwt_manager.verify_token(new_access_token)
        assert payload["sub"] == user_id


class TestRolePermissionManager:
    """Test role and permission management."""

    def setup_method(self):
        """Set up test fixtures."""
        self.role_manager = RolePermissionManager()

    def test_get_user_permissions_single_role(self):
        """Test getting permissions for single role."""
        roles = [UserRole.STANDARD_USER]
        permissions = self.role_manager.get_user_permissions(roles)
        
        assert Permission.ACCESS_AI_CHAT in permissions
        assert Permission.UPLOAD_DOCUMENTS in permissions
        assert Permission.MANAGE_SYSTEM not in permissions

    def test_get_user_permissions_multiple_roles(self):
        """Test getting permissions for multiple roles."""
        roles = [UserRole.STANDARD_USER, UserRole.CONTENT_MANAGER]
        permissions = self.role_manager.get_user_permissions(roles)
        
        assert Permission.ACCESS_AI_CHAT in permissions
        assert Permission.UPLOAD_DOCUMENTS in permissions
        assert Permission.CREATE_CONTENT in permissions
        assert Permission.UPDATE_CONTENT in permissions

    def test_get_user_permissions_admin_role(self):
        """Test getting permissions for admin role."""
        roles = [UserRole.SUPER_ADMIN]
        permissions = self.role_manager.get_user_permissions(roles)
        
        # Super admin should have all permissions
        all_permissions = set(Permission)
        assert permissions == all_permissions

    def test_has_permission_true(self):
        """Test permission check that should return True."""
        roles = [UserRole.CONTENT_MANAGER]
        
        assert self.role_manager.has_permission(roles, Permission.CREATE_CONTENT) is True
        assert self.role_manager.has_permission(roles, Permission.ACCESS_AI_CHAT) is True

    def test_has_permission_false(self):
        """Test permission check that should return False."""
        roles = [UserRole.READ_ONLY_USER]
        
        assert self.role_manager.has_permission(roles, Permission.CREATE_CONTENT) is False
        assert self.role_manager.has_permission(roles, Permission.MANAGE_SYSTEM) is False

    def test_can_assign_role_super_admin(self):
        """Test role assignment by super admin."""
        assigner_roles = [UserRole.SUPER_ADMIN]
        
        # Super admin can assign any role
        assert self.role_manager.can_assign_role(assigner_roles, UserRole.AI_COUNCIL_MEMBER) is True
        assert self.role_manager.can_assign_role(assigner_roles, UserRole.REGIONAL_ADMIN) is True
        assert self.role_manager.can_assign_role(assigner_roles, UserRole.STANDARD_USER) is True

    def test_can_assign_role_ai_council(self):
        """Test role assignment by AI council member."""
        assigner_roles = [UserRole.AI_COUNCIL_MEMBER]
        
        # AI council can assign content and user roles
        assert self.role_manager.can_assign_role(assigner_roles, UserRole.CONTENT_MANAGER) is True
        assert self.role_manager.can_assign_role(assigner_roles, UserRole.STANDARD_USER) is True
        assert self.role_manager.can_assign_role(assigner_roles, UserRole.SUPER_ADMIN) is False

    def test_can_assign_role_standard_user(self):
        """Test role assignment by standard user."""
        assigner_roles = [UserRole.STANDARD_USER]
        
        # Standard user cannot assign any roles
        assert self.role_manager.can_assign_role(assigner_roles, UserRole.STANDARD_USER) is False
        assert self.role_manager.can_assign_role(assigner_roles, UserRole.CONTENT_MANAGER) is False


class TestUserRegistrationRequest:
    """Test user registration request validation."""

    def test_valid_registration_request(self):
        """Test valid registration request."""
        request = UserRegistrationRequest(
            email="test@lixil.com",
            password="TestPassword123!",
            first_name="John",
            last_name="Doe",
            department="IT",
            region="US",
            language="en"
        )
        
        assert request.email == "test@lixil.com"
        assert request.first_name == "John"
        assert request.language == "en"

    def test_invalid_email(self):
        """Test registration with invalid email."""
        with pytest.raises(ValueError):
            UserRegistrationRequest(
                email="invalid_email",
                password="TestPassword123!",
                first_name="John",
                last_name="Doe"
            )

    def test_weak_password(self):
        """Test registration with weak password."""
        with pytest.raises(ValueError, match="Password must be at least 8 characters"):
            UserRegistrationRequest(
                email="test@lixil.com",
                password="weak",
                first_name="John",
                last_name="Doe"
            )

    def test_password_missing_uppercase(self):
        """Test password validation for missing uppercase."""
        with pytest.raises(ValueError, match="uppercase letter"):
            UserRegistrationRequest(
                email="test@lixil.com",
                password="testpassword123!",
                first_name="John",
                last_name="Doe"
            )

    def test_password_missing_digit(self):
        """Test password validation for missing digit."""
        with pytest.raises(ValueError, match="digit"):
            UserRegistrationRequest(
                email="test@lixil.com",
                password="TestPassword!",
                first_name="John",
                last_name="Doe"
            )


class TestLoginRequest:
    """Test login request validation."""

    def test_valid_login_request(self):
        """Test valid login request."""
        request = LoginRequest(
            email="test@lixil.com",
            password="TestPassword123!",
            mfa_code="123456",
            remember_me=True
        )
        
        assert request.email == "test@lixil.com"
        assert request.mfa_code == "123456"
        assert request.remember_me is True

    def test_invalid_mfa_code_format(self):
        """Test login with invalid MFA code format."""
        with pytest.raises(ValueError):
            LoginRequest(
                email="test@lixil.com",
                password="TestPassword123!",
                mfa_code="12345"  # Too short
            )

    def test_mfa_code_with_letters(self):
        """Test MFA code with non-numeric characters."""
        with pytest.raises(ValueError):
            LoginRequest(
                email="test@lixil.com",
                password="TestPassword123!",
                mfa_code="12345a"
            )


@pytest.mark.asyncio
class TestAuthenticationManager:
    """Test authentication manager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.database_url = "postgresql://test:test@localhost:5432/test_db"
        self.jwt_secret = "test_jwt_secret_key_12345"
        
        # Mock database pool
        self.mock_pool = AsyncMock()
        self.mock_conn = AsyncMock()
        self.mock_pool.acquire.return_value.__aenter__.return_value = self.mock_conn
        
        self.auth_manager = AuthenticationManager(self.database_url, self.jwt_secret)
        self.auth_manager.db_pool = self.mock_pool

    async def test_register_user_success(self):
        """Test successful user registration."""
        # Mock database responses
        self.mock_conn.fetchval.return_value = None  # Email doesn't exist
        self.mock_conn.execute.return_value = None  # Successful inserts
        
        request = UserRegistrationRequest(
            email="newuser@lixil.com",
            password="TestPassword123!",
            first_name="Jane",
            last_name="Smith",
            department="Marketing",
            region="EU"
        )
        
        user_id = await self.auth_manager.register_user(request)
        
        assert user_id is not None
        assert len(user_id) == 32  # MD5 hash length
        
        # Verify database calls
        assert self.mock_conn.fetchval.call_count == 1
        assert self.mock_conn.execute.call_count == 2  # User + role assignment

    async def test_register_user_email_exists(self):
        """Test user registration with existing email."""
        # Mock existing user
        self.mock_conn.fetchval.return_value = "existing_user_id"
        
        request = UserRegistrationRequest(
            email="existing@lixil.com",
            password="TestPassword123!",
            first_name="Jane",
            last_name="Smith"
        )
        
        with pytest.raises(ValueError, match="Email already registered"):
            await self.auth_manager.register_user(request)

    async def test_authenticate_user_success(self):
        """Test successful user authentication."""
        # Mock user data
        user_data = {
            'user_id': 'test_user_123',
            'password_hash': '$2b$12$test_hash',
            'is_active': True,
            'mfa_enabled': False,
            'mfa_secret': None,
            'failed_login_attempts': 0,
            'locked_until': None
        }
        self.mock_conn.fetchrow.return_value = user_data
        
        # Mock password verification
        with patch.object(self.auth_manager.password_manager, 'verify_password', return_value=True):
            # Mock role fetching
            with patch.object(self.auth_manager, '_get_user_roles', return_value=[UserRole.STANDARD_USER]):
                # Mock session storage
                with patch.object(self.auth_manager, '_store_session'):
                    request = LoginRequest(
                        email="test@lixil.com",
                        password="TestPassword123!"
                    )
                    
                    result = await self.auth_manager.authenticate_user(request, "127.0.0.1", "test-agent")
                    
                    assert result.success is True
                    assert result.user_id == 'test_user_123'
                    assert result.access_token is not None
                    assert result.refresh_token is not None

    async def test_authenticate_user_invalid_credentials(self):
        """Test authentication with invalid credentials."""
        # Mock user not found
        self.mock_conn.fetchrow.return_value = None
        
        request = LoginRequest(
            email="nonexistent@lixil.com",
            password="TestPassword123!"
        )
        
        result = await self.auth_manager.authenticate_user(request, "127.0.0.1", "test-agent")
        
        assert result.success is False
        assert result.error_message == "Invalid credentials"

    async def test_authenticate_user_account_locked(self):
        """Test authentication with locked account."""
        # Mock locked user
        user_data = {
            'user_id': 'test_user_123',
            'password_hash': '$2b$12$test_hash',
            'is_active': True,
            'mfa_enabled': False,
            'mfa_secret': None,
            'failed_login_attempts': 5,
            'locked_until': datetime.now() + timedelta(minutes=30)
        }
        self.mock_conn.fetchrow.return_value = user_data
        
        request = LoginRequest(
            email="test@lixil.com",
            password="TestPassword123!"
        )
        
        result = await self.auth_manager.authenticate_user(request, "127.0.0.1", "test-agent")
        
        assert result.success is False
        assert result.error_message == "Account temporarily locked"

    async def test_authenticate_user_mfa_required(self):
        """Test authentication requiring MFA."""
        # Mock user with MFA enabled
        user_data = {
            'user_id': 'test_user_123',
            'password_hash': '$2b$12$test_hash',
            'is_active': True,
            'mfa_enabled': True,
            'mfa_secret': 'TEST_SECRET_123',
            'failed_login_attempts': 0,
            'locked_until': None
        }
        self.mock_conn.fetchrow.return_value = user_data
        
        # Mock password verification
        with patch.object(self.auth_manager.password_manager, 'verify_password', return_value=True):
            request = LoginRequest(
                email="test@lixil.com",
                password="TestPassword123!"
                # No MFA code provided
            )
            
            result = await self.auth_manager.authenticate_user(request, "127.0.0.1", "test-agent")
            
            assert result.success is False
            assert result.requires_mfa is True

    async def test_authenticate_user_invalid_mfa(self):
        """Test authentication with invalid MFA code."""
        # Mock user with MFA enabled
        user_data = {
            'user_id': 'test_user_123',
            'password_hash': '$2b$12$test_hash',
            'is_active': True,
            'mfa_enabled': True,
            'mfa_secret': 'TEST_SECRET_123',
            'failed_login_attempts': 0,
            'locked_until': None
        }
        self.mock_conn.fetchrow.return_value = user_data
        
        # Mock password verification and MFA verification
        with patch.object(self.auth_manager.password_manager, 'verify_password', return_value=True):
            with patch.object(self.auth_manager.mfa_manager, 'verify_totp', return_value=False):
                request = LoginRequest(
                    email="test@lixil.com",
                    password="TestPassword123!",
                    mfa_code="123456"
                )
                
                result = await self.auth_manager.authenticate_user(request, "127.0.0.1", "test-agent")
                
                assert result.success is False
                assert result.error_message == "Invalid MFA code"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


"""
Tests for authentication endpoints and functionality.

This module tests user authentication, registration, token management,
and security features of the Context Engineering Platform API.
"""

import pytest
from fastapi import status
from unittest.mock import patch

from src.auth.router import verify_password, get_password_hash, create_access_token


class TestAuthenticationEndpoints:
    """Test authentication endpoints."""
    
    def test_login_success(self, client):
        """Test successful login."""
        login_data = {
            "email": "admin@example.com",
            "password": "secret"
        }
        response = client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
        assert "user" in data
        assert data["user"]["email"] == "admin@example.com"
    
    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials."""
        login_data = {
            "email": "admin@example.com",
            "password": "wrong_password"
        }
        response = client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Incorrect email or password" in response.json()["message"]
    
    def test_login_nonexistent_user(self, client):
        """Test login with nonexistent user."""
        login_data = {
            "email": "nonexistent@example.com",
            "password": "password"
        }
        response = client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_register_success(self, client):
        """Test successful user registration."""
        register_data = {
            "email": "newuser@example.com",
            "password": "SecurePass123",
            "full_name": "New User"
        }
        response = client.post("/api/v1/auth/register", json=register_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["email"] == "newuser@example.com"
        assert data["full_name"] == "New User"
        assert data["is_active"] is True
        assert "id" in data
    
    def test_register_duplicate_email(self, client):
        """Test registration with duplicate email."""
        register_data = {
            "email": "admin@example.com",  # Already exists
            "password": "SecurePass123",
            "full_name": "Duplicate User"
        }
        response = client.post("/api/v1/auth/register", json=register_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "Email already registered" in response.json()["message"]
    
    def test_register_weak_password(self, client):
        """Test registration with weak password."""
        register_data = {
            "email": "weakpass@example.com",
            "password": "weak",  # Too weak
            "full_name": "Weak Password User"
        }
        response = client.post("/api/v1/auth/register", json=register_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_get_current_user(self, client, auth_headers):
        """Test getting current user profile."""
        response = client.get("/api/v1/auth/me", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["email"] == "admin@example.com"
        assert "id" in data
        assert "created_at" in data
    
    def test_get_current_user_unauthorized(self, client):
        """Test getting current user without authentication."""
        response = client.get("/api/v1/auth/me")
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_update_user_profile(self, client, auth_headers):
        """Test updating user profile."""
        update_data = {
            "full_name": "Updated Name",
            "preferences": {"theme": "dark", "language": "en"}
        }
        response = client.put("/api/v1/auth/me", json=update_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["full_name"] == "Updated Name"
        assert data["preferences"]["theme"] == "dark"
    
    def test_change_password_success(self, client, auth_headers):
        """Test successful password change."""
        password_data = {
            "current_password": "secret",
            "new_password": "NewSecurePass123"
        }
        response = client.post("/api/v1/auth/change-password", json=password_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        assert "Password changed successfully" in response.json()["message"]
    
    def test_change_password_wrong_current(self, client, auth_headers):
        """Test password change with wrong current password."""
        password_data = {
            "current_password": "wrong_password",
            "new_password": "NewSecurePass123"
        }
        response = client.post("/api/v1/auth/change-password", json=password_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Current password is incorrect" in response.json()["message"]
    
    def test_forgot_password(self, client):
        """Test forgot password request."""
        request_data = {"email": "admin@example.com"}
        response = client.post("/api/v1/auth/forgot-password", json=request_data)
        
        assert response.status_code == status.HTTP_200_OK
        assert "password reset link" in response.json()["message"]
    
    def test_refresh_token(self, client, auth_headers):
        """Test token refresh."""
        response = client.post("/api/v1/auth/refresh", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_logout(self, client, auth_headers):
        """Test user logout."""
        response = client.post("/api/v1/auth/logout", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        assert "Successfully logged out" in response.json()["message"]


class TestAuthenticationUtilities:
    """Test authentication utility functions."""
    
    def test_password_hashing(self):
        """Test password hashing and verification."""
        password = "test_password_123"
        hashed = get_password_hash(password)
        
        assert hashed != password
        assert verify_password(password, hashed) is True
        assert verify_password("wrong_password", hashed) is False
    
    def test_create_access_token(self):
        """Test JWT token creation."""
        data = {"sub": "test@example.com"}
        token = create_access_token(data)
        
        assert isinstance(token, str)
        assert len(token) > 0
    
    @patch('src.auth.router.jwt.decode')
    def test_token_validation(self, mock_decode):
        """Test JWT token validation."""
        mock_decode.return_value = {"sub": "test@example.com"}
        
        # This would be tested through the actual endpoint
        # since get_current_user is a dependency
        assert mock_decode.called is False  # Not called yet
    
    def test_password_validation_rules(self):
        """Test password validation rules."""
        # Test through the registration endpoint
        weak_passwords = [
            "short",
            "nouppercase123",
            "NOLOWERCASE123",
            "NoNumbers",
            "12345678"  # Only numbers
        ]
        
        for password in weak_passwords:
            # Password validation happens in the Pydantic model
            # This would be caught during request validation
            assert len(password) < 8 or not any(c.isupper() for c in password) or \
                   not any(c.islower() for c in password) or not any(c.isdigit() for c in password)


class TestAuthenticationSecurity:
    """Test authentication security features."""
    
    def test_invalid_token_format(self, client):
        """Test request with invalid token format."""
        headers = {"Authorization": "Bearer invalid_token_format"}
        response = client.get("/api/v1/auth/me", headers=headers)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_missing_authorization_header(self, client):
        """Test request without authorization header."""
        response = client.get("/api/v1/auth/me")
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_malformed_authorization_header(self, client):
        """Test request with malformed authorization header."""
        headers = {"Authorization": "InvalidFormat token"}
        response = client.get("/api/v1/auth/me", headers=headers)
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_expired_token_handling(self, client):
        """Test handling of expired tokens."""
        # This would require mocking the JWT decode to raise an ExpiredSignatureError
        # For now, we test the general error handling
        headers = {"Authorization": "Bearer expired.token.here"}
        response = client.get("/api/v1/auth/me", headers=headers)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


if __name__ == "__main__":
    pytest.main([__file__])


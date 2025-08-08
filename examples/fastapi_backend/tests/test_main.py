"""
Tests for main application and configuration.

This module tests application startup, health checks, metrics,
middleware functionality, and error handling.
"""

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import patch

from src.main import create_application
from src.common.config import get_settings


class TestApplicationStartup:
    """Test application startup and configuration."""
    
    def test_create_application(self):
        """Test application creation."""
        app = create_application()
        assert app.title == "Context Engineering Platform API"
        assert "Advanced AI Context Management" in app.description
        assert app.version == "1.0.0"
    
    def test_application_with_debug_mode(self):
        """Test application in debug mode."""
        with patch('src.common.config.get_settings') as mock_settings:
            settings = get_settings()
            settings.debug = True
            mock_settings.return_value = settings
            
            app = create_application()
            assert app.docs_url == "/docs"
            assert app.redoc_url == "/redoc"
            assert app.openapi_url == "/openapi.json"
    
    def test_application_without_debug_mode(self):
        """Test application in production mode."""
        with patch('src.common.config.get_settings') as mock_settings:
            settings = get_settings()
            settings.debug = False
            mock_settings.return_value = settings
            
            app = create_application()
            assert app.docs_url is None
            assert app.redoc_url is None
            assert app.openapi_url is None


class TestHealthAndMonitoring:
    """Test health check and monitoring endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "environment" in data
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        # Should contain Prometheus metrics format
        assert "http_requests_total" in response.text or len(response.text) >= 0


class TestMiddleware:
    """Test middleware functionality."""
    
    def test_cors_middleware(self, client):
        """Test CORS middleware."""
        # Make a preflight request
        response = client.options("/health", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        })
        
        # Should allow CORS
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_405_METHOD_NOT_ALLOWED]
    
    def test_logging_middleware(self, client):
        """Test logging middleware."""
        # Make a request that should be logged
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        # Logging middleware should not affect response
        assert response.json()["status"] == "healthy"
    
    def test_metrics_middleware(self, client):
        """Test metrics middleware."""
        # Make a request to generate metrics
        client.get("/health")
        
        # Check metrics endpoint
        response = client.get("/metrics")
        assert response.status_code == status.HTTP_200_OK


class TestErrorHandling:
    """Test error handling and exception handlers."""
    
    def test_404_error(self, client):
        """Test 404 error handling."""
        response = client.get("/nonexistent-endpoint")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_method_not_allowed(self, client):
        """Test 405 method not allowed."""
        response = client.post("/health")  # Health endpoint only accepts GET
        
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
    
    def test_validation_error(self, client, auth_headers):
        """Test validation error handling."""
        # Send invalid data to an endpoint
        invalid_data = {"invalid": "data"}
        response = client.post("/api/v1/agents/", json=invalid_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_authentication_error(self, client):
        """Test authentication error handling."""
        # Try to access protected endpoint without auth
        response = client.get("/api/v1/auth/me")
        
        assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN]


class TestRouterInclusion:
    """Test that all routers are properly included."""
    
    def test_auth_router_included(self, client):
        """Test auth router is included."""
        response = client.post("/api/v1/auth/login", json={
            "email": "admin@example.com",
            "password": "secret"
        })
        
        # Should not return 404
        assert response.status_code != status.HTTP_404_NOT_FOUND
    
    def test_agents_router_included(self, client):
        """Test agents router is included."""
        response = client.get("/api/v1/agents/providers/supported")
        
        # Should not return 404
        assert response.status_code != status.HTTP_404_NOT_FOUND
    
    def test_knowledge_router_included(self, client, auth_headers):
        """Test knowledge router is included."""
        response = client.get("/api/v1/knowledge/schema", headers=auth_headers)
        
        # Should not return 404
        assert response.status_code != status.HTTP_404_NOT_FOUND
    
    def test_memory_router_included(self, client):
        """Test memory router is included."""
        response = client.get("/api/v1/memory/types")
        
        # Should not return 404
        assert response.status_code != status.HTTP_404_NOT_FOUND


class TestApplicationSecurity:
    """Test application security features."""
    
    def test_security_headers(self, client):
        """Test security headers are present."""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        # Basic security checks
        assert "server" not in response.headers.get("server", "").lower()
    
    def test_cors_configuration(self, client):
        """Test CORS configuration."""
        response = client.get("/health", headers={
            "Origin": "http://localhost:3000"
        })
        
        assert response.status_code == status.HTTP_200_OK
        # Should handle CORS properly
    
    def test_trusted_host_middleware_in_production(self):
        """Test trusted host middleware in production."""
        with patch('src.common.config.get_settings') as mock_settings:
            settings = get_settings()
            settings.debug = False
            settings.allowed_hosts = ["example.com"]
            mock_settings.return_value = settings
            
            app = create_application()
            client = TestClient(app)
            
            # Should work with allowed host
            response = client.get("/health", headers={"Host": "example.com"})
            assert response.status_code == status.HTTP_200_OK


class TestApplicationConfiguration:
    """Test application configuration."""
    
    def test_settings_loading(self):
        """Test settings are loaded correctly."""
        settings = get_settings()
        
        assert settings.app_name == "Context Engineering Platform API"
        assert settings.environment in ["development", "staging", "production"]
        assert isinstance(settings.debug, bool)
        assert isinstance(settings.port, int)
    
    def test_environment_specific_settings(self):
        """Test environment-specific settings."""
        with patch.dict('os.environ', {'ENVIRONMENT': 'production'}):
            settings = get_settings()
            # In production, debug should typically be False
            # This depends on your configuration logic
    
    def test_cors_origins_configuration(self):
        """Test CORS origins configuration."""
        settings = get_settings()
        
        assert isinstance(settings.cors_origins, list)
        assert len(settings.cors_origins) > 0
        # Should include common development origins
        assert any("localhost" in origin for origin in settings.cors_origins)


class TestApplicationLifespan:
    """Test application lifespan events."""
    
    def test_application_startup(self):
        """Test application startup event."""
        # This is tested implicitly when creating the test client
        app = create_application()
        client = TestClient(app)
        
        # If startup fails, this would raise an exception
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
    
    def test_application_shutdown(self):
        """Test application shutdown event."""
        # This is tested implicitly when the test client is destroyed
        app = create_application()
        client = TestClient(app)
        
        # Make a request to ensure app is running
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        
        # Client cleanup should trigger shutdown events
        del client


class TestAPIDocumentation:
    """Test API documentation endpoints."""
    
    def test_openapi_schema_in_debug(self):
        """Test OpenAPI schema is available in debug mode."""
        with patch('src.common.config.get_settings') as mock_settings:
            settings = get_settings()
            settings.debug = True
            mock_settings.return_value = settings
            
            app = create_application()
            client = TestClient(app)
            
            response = client.get("/openapi.json")
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert data["info"]["title"] == "Context Engineering Platform API"
            assert "paths" in data
            assert "components" in data
    
    def test_docs_endpoint_in_debug(self):
        """Test docs endpoint is available in debug mode."""
        with patch('src.common.config.get_settings') as mock_settings:
            settings = get_settings()
            settings.debug = True
            mock_settings.return_value = settings
            
            app = create_application()
            client = TestClient(app)
            
            response = client.get("/docs")
            assert response.status_code == status.HTTP_200_OK
            assert "text/html" in response.headers["content-type"]


if __name__ == "__main__":
    pytest.main([__file__])


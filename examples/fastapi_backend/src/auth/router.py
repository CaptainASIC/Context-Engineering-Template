"""
Authentication router with JWT token management.

This module provides authentication endpoints including login, registration,
token refresh, and password management with proper security measures.
"""

from datetime import datetime, timedelta
from typing import Annotated, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from loguru import logger
from passlib.context import CryptContext

from src.common.config import get_settings
from src.common.exceptions import (
    AuthenticationException,
    AuthorizationException,
    InvalidTokenException,
    ValidationException
)
from src.auth.models import (
    LoginRequest,
    LoginResponse,
    User,
    UserCreate,
    UserProfile,
    Token,
    TokenData,
    ChangePasswordRequest,
    PasswordResetRequest
)

router = APIRouter()
settings = get_settings()

# Security setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Mock user database (in production, use a real database)
fake_users_db = {
    "admin@example.com": {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "email": "admin@example.com",
        "full_name": "System Administrator",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        "is_active": True,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "last_login": None,
        "preferences": {}
    },
    "user@example.com": {
        "id": "550e8400-e29b-41d4-a716-446655440001",
        "email": "user@example.com",
        "full_name": "Demo User",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        "is_active": True,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "last_login": None,
        "preferences": {}
    }
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


def get_user(email: str) -> Optional[dict]:
    """Get user by email from database."""
    return fake_users_db.get(email)


def authenticate_user(email: str, password: str) -> Optional[dict]:
    """Authenticate user with email and password."""
    user = get_user(email)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
) -> dict:
    """Get current authenticated user from JWT token."""
    token = credentials.credentials
    
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        email: str = payload.get("sub")
        if email is None:
            raise InvalidTokenException("Token missing subject")
        
        token_data = TokenData(email=email)
    except JWTError as e:
        logger.error(f"JWT decode error: {e}")
        raise InvalidTokenException("Could not validate credentials")
    
    user = get_user(email=token_data.email)
    if user is None:
        raise AuthenticationException("User not found")
    
    if not user["is_active"]:
        raise AuthorizationException("Inactive user")
    
    return user


async def get_current_active_user(
    current_user: Annotated[dict, Depends(get_current_user)]
) -> dict:
    """Get current active user."""
    if not current_user["is_active"]:
        raise AuthorizationException("Inactive user")
    return current_user


@router.post("/login", response_model=LoginResponse)
async def login(login_data: LoginRequest) -> LoginResponse:
    """
    Authenticate user and return access token.
    
    - **email**: User email address
    - **password**: User password
    
    Returns JWT access token for authenticated requests.
    """
    user = authenticate_user(login_data.email, login_data.password)
    if not user:
        raise AuthenticationException("Incorrect email or password")
    
    # Update last login
    user["last_login"] = datetime.utcnow().isoformat()
    
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user["email"]}, 
        expires_delta=access_token_expires
    )
    
    logger.info(f"User {user['email']} logged in successfully")
    
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
        user=User(**user)
    )


@router.post("/register", response_model=User)
async def register(user_data: UserCreate) -> User:
    """
    Register a new user account.
    
    - **email**: User email address (must be unique)
    - **password**: User password (minimum 8 characters with complexity requirements)
    - **full_name**: Optional full name
    
    Returns the created user information.
    """
    if get_user(user_data.email):
        raise ValidationException("Email already registered", field="email")
    
    hashed_password = get_password_hash(user_data.password)
    
    new_user = {
        "id": str(UUID()),
        "email": user_data.email,
        "full_name": user_data.full_name,
        "hashed_password": hashed_password,
        "is_active": True,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "last_login": None,
        "preferences": {}
    }
    
    fake_users_db[user_data.email] = new_user
    
    logger.info(f"New user registered: {user_data.email}")
    
    return User(**new_user)


@router.get("/me", response_model=UserProfile)
async def get_current_user_profile(
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> UserProfile:
    """
    Get current user profile information.
    
    Returns detailed profile information for the authenticated user.
    """
    return UserProfile(**current_user)


@router.put("/me", response_model=UserProfile)
async def update_current_user_profile(
    profile_data: dict,
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> UserProfile:
    """
    Update current user profile information.
    
    Allows users to update their profile information including preferences.
    """
    # Update allowed fields
    if "full_name" in profile_data:
        current_user["full_name"] = profile_data["full_name"]
    
    if "preferences" in profile_data:
        current_user["preferences"].update(profile_data["preferences"])
    
    current_user["updated_at"] = datetime.utcnow().isoformat()
    
    logger.info(f"User profile updated: {current_user['email']}")
    
    return UserProfile(**current_user)


@router.post("/change-password")
async def change_password(
    password_data: ChangePasswordRequest,
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> dict:
    """
    Change user password.
    
    - **current_password**: Current password for verification
    - **new_password**: New password (minimum 8 characters with complexity requirements)
    
    Returns success confirmation.
    """
    if not verify_password(password_data.current_password, current_user["hashed_password"]):
        raise AuthenticationException("Current password is incorrect")
    
    new_hashed_password = get_password_hash(password_data.new_password)
    current_user["hashed_password"] = new_hashed_password
    current_user["updated_at"] = datetime.utcnow().isoformat()
    
    logger.info(f"Password changed for user: {current_user['email']}")
    
    return {"message": "Password changed successfully"}


@router.post("/forgot-password")
async def forgot_password(request: PasswordResetRequest) -> dict:
    """
    Request password reset.
    
    - **email**: User email address
    
    In a real implementation, this would send a password reset email.
    """
    user = get_user(request.email)
    if not user:
        # Don't reveal if email exists or not for security
        return {"message": "If the email exists, a password reset link has been sent"}
    
    # In production, generate a secure token and send email
    logger.info(f"Password reset requested for: {request.email}")
    
    return {"message": "If the email exists, a password reset link has been sent"}


@router.post("/refresh")
async def refresh_token(
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> Token:
    """
    Refresh access token.
    
    Returns a new access token for the authenticated user.
    """
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": current_user["email"]}, 
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
        user=User(**current_user)
    )


@router.post("/logout")
async def logout(
    current_user: Annotated[dict, Depends(get_current_active_user)]
) -> dict:
    """
    Logout user.
    
    In a real implementation, this would invalidate the token.
    """
    logger.info(f"User logged out: {current_user['email']}")
    
    return {"message": "Successfully logged out"}


# LIXIL AI Hub Authentication and RBAC Module

This module provides comprehensive authentication and role-based access control (RBAC) capabilities for the LIXIL AI Hub Platform.

## Features

- **JWT-based Authentication**: Secure token-based authentication with refresh tokens
- **Role-Based Access Control**: Granular permissions system with role hierarchies
- **Multi-Factor Authentication**: TOTP-based MFA with QR code generation
- **Password Security**: bcrypt hashing with configurable rounds
- **Session Management**: Secure session handling with expiration
- **Audit Logging**: Comprehensive logging for compliance and security monitoring
- **Enterprise Integration**: Support for LDAP/Active Directory integration

## Quick Start

```python
from auth_manager import AuthenticationManager, UserRegistrationRequest, LoginRequest

# Initialize authentication manager
auth_manager = AuthenticationManager(
    database_url="postgresql://user:pass@localhost:5432/lixil_ai_hub",
    jwt_secret="your-secret-key"
)

await auth_manager.initialize()

# Register a new user
registration = UserRegistrationRequest(
    email="user@lixil.com",
    password="SecurePassword123!",
    first_name="John",
    last_name="Doe",
    department="IT",
    region="US"
)

user_id = await auth_manager.register_user(registration)

# Authenticate user
login = LoginRequest(
    email="user@lixil.com",
    password="SecurePassword123!"
)

result = await auth_manager.authenticate_user(login, "127.0.0.1", "user-agent")

if result.success:
    print(f"Access token: {result.access_token}")
    print(f"User ID: {result.user_id}")
```

## User Roles and Permissions

### Role Hierarchy

- **Super Admin**: Full system access
- **AI Council Member**: Content and knowledge base management
- **Regional Admin**: Regional user and content management
- **Department Admin**: Department-level management
- **Content Manager**: Content creation and editing
- **Policy Reviewer**: Content approval capabilities
- **Standard User**: Basic AI chat and document upload
- **Read Only User**: AI chat access only

### Permission System

The system uses granular permissions that can be assigned to roles:

- Content Management: `CREATE_CONTENT`, `UPDATE_CONTENT`, `DELETE_CONTENT`, `APPROVE_CONTENT`, `PUBLISH_CONTENT`
- User Management: `CREATE_USER`, `UPDATE_USER`, `DELETE_USER`, `ASSIGN_ROLES`
- System Administration: `MANAGE_SYSTEM`, `VIEW_AUDIT_LOGS`, `MANAGE_INTEGRATIONS`
- AI Features: `ACCESS_AI_CHAT`, `UPLOAD_DOCUMENTS`, `MANAGE_KNOWLEDGE_BASE`
- Regional/Department: `MANAGE_REGION`, `MANAGE_DEPARTMENT`

## Multi-Factor Authentication

Enable MFA for enhanced security:

```python
from auth_manager import MFAManager

mfa_manager = MFAManager("LIXIL AI Hub")

# Generate secret for user
secret = mfa_manager.generate_secret()

# Generate QR code for setup
qr_code = mfa_manager.generate_qr_code("user@lixil.com", secret)

# Verify TOTP token
is_valid = mfa_manager.verify_totp(secret, "123456")
```

## Database Schema

The module automatically creates the following tables:

- `users`: User profiles and authentication data
- `user_roles`: Role assignments with audit trail
- `user_sessions`: Active session tracking
- `auth_audit_log`: Security event logging

## Security Features

### Password Security
- bcrypt hashing with configurable rounds
- Password strength validation
- Secure password generation

### Session Security
- JWT tokens with configurable expiration
- Refresh token rotation
- Session invalidation and cleanup

### Account Protection
- Failed login attempt tracking
- Account lockout after multiple failures
- Rate limiting support

### Audit and Compliance
- Comprehensive audit logging
- GDPR-compliant data handling
- Security event tracking

## Configuration

### Environment Variables

```bash
# Database configuration
DATABASE_URL=postgresql://user:pass@localhost:5432/lixil_ai_hub

# JWT configuration
JWT_SECRET_KEY=your-very-secure-secret-key
JWT_ACCESS_TOKEN_EXPIRE_HOURS=1
JWT_REFRESH_TOKEN_EXPIRE_DAYS=30

# Password security
PASSWORD_BCRYPT_ROUNDS=12

# MFA configuration
MFA_ISSUER_NAME="LIXIL AI Hub"

# Session configuration
SESSION_TIMEOUT_HOURS=8
REMEMBER_ME_DAYS=30

# Account lockout
MAX_FAILED_ATTEMPTS=5
LOCKOUT_DURATION_MINUTES=30
```

## Testing

Run the comprehensive test suite:

```bash
# Install test dependencies
pip install -r requirements.txt

# Run tests
pytest test_auth_manager.py -v

# Run with coverage
pytest test_auth_manager.py --cov=auth_manager --cov-report=html
```

## Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer

app = FastAPI()
security = HTTPBearer()

async def get_current_user(token: str = Depends(security)):
    payload = auth_manager.jwt_manager.verify_token(token.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    return payload["sub"]

@app.post("/login")
async def login(request: LoginRequest):
    result = await auth_manager.authenticate_user(request, "127.0.0.1", "api")
    if not result.success:
        raise HTTPException(status_code=401, detail=result.error_message)
    return {"access_token": result.access_token, "token_type": "bearer"}
```

### React Frontend Integration

```javascript
// Login component
const login = async (email, password, mfaCode) => {
  const response = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password, mfa_code: mfaCode })
  });
  
  if (response.ok) {
    const data = await response.json();
    localStorage.setItem('access_token', data.access_token);
    return data;
  } else {
    throw new Error('Login failed');
  }
};

// Protected API calls
const apiCall = async (url, options = {}) => {
  const token = localStorage.getItem('access_token');
  return fetch(url, {
    ...options,
    headers: {
      ...options.headers,
      'Authorization': `Bearer ${token}`
    }
  });
};
```

## Best Practices

1. **Use HTTPS**: Always use HTTPS in production
2. **Secure JWT Secrets**: Use strong, randomly generated JWT secrets
3. **Regular Token Rotation**: Implement refresh token rotation
4. **Monitor Failed Attempts**: Set up alerts for suspicious login activity
5. **Regular Audits**: Review audit logs regularly
6. **MFA Enforcement**: Require MFA for privileged accounts
7. **Password Policies**: Enforce strong password requirements
8. **Session Management**: Implement proper session cleanup

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Verify database URL and credentials
   - Ensure database is accessible
   - Check connection pool settings

2. **JWT Token Issues**
   - Verify JWT secret configuration
   - Check token expiration settings
   - Ensure clock synchronization

3. **MFA Problems**
   - Verify TOTP secret generation
   - Check time synchronization
   - Validate QR code generation

4. **Permission Denied**
   - Review role assignments
   - Check permission mappings
   - Verify role hierarchy

## Contributing

1. Follow the established code style
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure security best practices are followed

## License

This module is part of the LIXIL AI Hub Platform and is proprietary software.


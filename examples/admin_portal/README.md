# LIXIL AI Hub Admin Portal

This module provides a comprehensive admin portal for AI Council members to manage policies, permanent statements, and system administration for the LIXIL AI Hub Platform.

## Features

- **Policy Document Management**: Upload, version, approve, and publish policy documents
- **Permanent Statement Creation**: Create and manage curated Q&A responses
- **Content Approval Workflow**: Review and approve content before publication
- **User Management**: Manage user roles and permissions
- **Analytics Dashboard**: Monitor system usage and content performance
- **Audit Trail**: Complete logging of all administrative actions
- **Multi-language Support**: Manage content in multiple languages
- **Regional Configuration**: Support for region-specific policies

## Architecture

### Backend (FastAPI)
- RESTful API for all admin operations
- Async/await for high performance
- Comprehensive input validation
- Role-based access control integration
- File upload and processing
- Database operations with connection pooling

### Frontend (React)
- Modern Material-UI components
- Real-time dashboard updates
- Responsive design for desktop and mobile
- Interactive charts and analytics
- Drag-and-drop file uploads
- Advanced search and filtering

## Quick Start

### Backend Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://user:pass@localhost:5432/lixil_ai_hub"
export JWT_SECRET_KEY="your-secret-key"
export FILE_STORAGE_PATH="/app/storage"

# Run the server
python admin_portal.py
```

### Frontend Setup

```bash
# Install dependencies
npm install @mui/material @emotion/react @emotion/styled
npm install @mui/icons-material
npm install recharts  # For charts

# Start development server
npm start
```

## API Endpoints

### Policy Management

#### Upload Policy Document
```http
POST /api/admin/policies/upload
Content-Type: multipart/form-data
Authorization: Bearer <token>

{
  "file": <file>,
  "title": "AI Usage Policy",
  "category": "AI Policy",
  "language": "en",
  "region": "US",
  "tags": ["ai", "policy"],
  "metadata": {}
}
```

#### Search Policies
```http
POST /api/admin/policies/search
Content-Type: application/json
Authorization: Bearer <token>

{
  "query": "AI policy",
  "category": "AI Policy",
  "status": "published",
  "language": "en",
  "limit": 20,
  "offset": 0
}
```

### Statement Management

#### Create Permanent Statement
```http
POST /api/admin/statements
Content-Type: application/json
Authorization: Bearer <token>

{
  "question": "What is the AI usage policy?",
  "answer": "The AI usage policy defines how employees should use AI tools...",
  "category": "AI Policy",
  "priority": 8,
  "language": "en",
  "tags": ["ai", "policy"]
}
```

### Content Approval

#### Approve/Reject Content
```http
POST /api/admin/content/approve
Content-Type: application/json
Authorization: Bearer <token>

{
  "content_id": "policy_123",
  "content_type": "policy_document",
  "action": "approve",
  "comments": "Looks good",
  "publish_immediately": false
}
```

### Analytics

#### Get Admin Analytics
```http
POST /api/admin/analytics
Content-Type: application/json
Authorization: Bearer <token>

{
  "start_date": "2024-01-01T00:00:00Z",
  "end_date": "2024-01-31T23:59:59Z",
  "granularity": "day",
  "region": "US"
}
```

## Database Schema

### Policy Documents
```sql
CREATE TABLE policy_documents (
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
```

### Permanent Statements
```sql
CREATE TABLE permanent_statements (
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
```

## React Components

### AdminDashboard
Main dashboard component with tabs for different admin functions:

```jsx
import AdminDashboard from './AdminDashboard';

function App() {
  return (
    <div className="App">
      <AdminDashboard />
    </div>
  );
}
```

### Key Features
- **Dashboard Overview**: Summary cards and charts
- **Policy Management**: Upload, search, and manage policies
- **Statement Management**: Create and edit permanent statements
- **Content Approval**: Review and approve pending content
- **Analytics**: Interactive charts and metrics

## Content Workflow

### Policy Document Lifecycle
1. **Upload**: Admin uploads policy document with metadata
2. **Draft**: Document is saved as draft status
3. **Review**: Document status changed to "pending_review"
4. **Approval**: AI Council member approves or rejects
5. **Publication**: Approved documents can be published
6. **Versioning**: New versions increment automatically

### Statement Lifecycle
1. **Creation**: Admin creates permanent statement
2. **Review**: Optional review process
3. **Activation**: Statement becomes active and searchable
4. **Usage Tracking**: Track how often statement is used
5. **Updates**: Modify or deactivate as needed

## Security Features

### Authentication & Authorization
- JWT token-based authentication
- Role-based access control (RBAC)
- Permission validation for all operations
- Session management and timeout

### File Security
- File type validation
- Virus scanning (configurable)
- Secure file storage with checksums
- Access control for file downloads

### Audit Trail
- Complete logging of all admin actions
- IP address and user agent tracking
- Resource-level change tracking
- Compliance reporting

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/lixil_ai_hub

# Authentication
JWT_SECRET_KEY=your-very-secure-secret-key
JWT_ACCESS_TOKEN_EXPIRE_HOURS=8

# File Storage
FILE_STORAGE_PATH=/app/storage
MAX_FILE_SIZE_MB=50
ALLOWED_FILE_TYPES=pdf,docx,doc,txt,md

# Admin Portal
ADMIN_PORTAL_URL=https://admin.lixil-ai-hub.com
CORS_ORIGINS=http://localhost:3000,https://admin.lixil-ai-hub.com

# Analytics
ANALYTICS_RETENTION_DAYS=365
ENABLE_REAL_TIME_ANALYTICS=true

# Notifications
ENABLE_EMAIL_NOTIFICATIONS=true
SMTP_SERVER=smtp.lixil.com
SMTP_PORT=587
```

## Testing

### Backend Tests
```bash
# Run all tests
pytest test_admin_portal.py -v

# Run with coverage
pytest test_admin_portal.py --cov=admin_portal --cov-report=html

# Run specific test categories
pytest test_admin_portal.py::TestPolicyUploadRequest -v
pytest test_admin_portal.py::TestAdminPortalManager -v
```

### Frontend Tests
```bash
# Install testing dependencies
npm install @testing-library/react @testing-library/jest-dom

# Run tests
npm test

# Run with coverage
npm test -- --coverage
```

## Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "admin_portal:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: admin-portal
spec:
  replicas: 3
  selector:
    matchLabels:
      app: admin-portal
  template:
    metadata:
      labels:
        app: admin-portal
    spec:
      containers:
      - name: admin-portal
        image: lixil/ai-hub-admin-portal:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: admin-portal-secrets
              key: database-url
```

## Monitoring and Observability

### Health Checks
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }
```

### Metrics
- Request/response times
- Error rates
- Database connection pool status
- File upload success rates
- User activity metrics

### Logging
```python
import logging

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
```

## Best Practices

### Security
1. **Input Validation**: Validate all inputs using Pydantic models
2. **File Scanning**: Scan uploaded files for malware
3. **Access Control**: Implement proper RBAC for all operations
4. **Audit Logging**: Log all administrative actions
5. **HTTPS Only**: Use HTTPS in production environments

### Performance
1. **Database Indexing**: Proper indexes on frequently queried columns
2. **Connection Pooling**: Use connection pools for database access
3. **Caching**: Cache frequently accessed data
4. **Async Operations**: Use async/await for I/O operations
5. **File Optimization**: Compress and optimize uploaded files

### Usability
1. **Responsive Design**: Support desktop and mobile devices
2. **Progressive Loading**: Load data progressively for better UX
3. **Error Handling**: Provide clear error messages
4. **Keyboard Navigation**: Support keyboard shortcuts
5. **Accessibility**: Follow WCAG guidelines

## Troubleshooting

### Common Issues

1. **File Upload Failures**
   - Check file size limits
   - Verify file type restrictions
   - Ensure storage directory permissions

2. **Database Connection Issues**
   - Verify database URL and credentials
   - Check connection pool settings
   - Monitor connection usage

3. **Authentication Problems**
   - Verify JWT secret configuration
   - Check token expiration settings
   - Validate user permissions

4. **Performance Issues**
   - Monitor database query performance
   - Check file storage I/O
   - Review connection pool usage

## Contributing

1. Follow the established code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure security best practices are followed
5. Test with multiple browsers and devices

## License

This module is part of the LIXIL AI Hub Platform and is proprietary software.


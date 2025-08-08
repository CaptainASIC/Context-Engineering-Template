# Enhanced Claude Code & WARP ADE Guide

This guide provides comprehensive best practices for developing with Claude Code and the WARP ADE, integrating detailed insights for robust, maintainable, and efficient Python development. Adhering to these guidelines ensures consistency, reliability, and high quality across all projects.

## Core Development Philosophy

Effective software development is rooted in foundational principles that guide design and implementation decisions. For Claude Code and WARP ADE, these philosophies emphasize simplicity, necessity, and sound architectural patterns.

### KISS (Keep It Simple, Stupid)

Simplicity is paramount in software design. The principle of KISS advocates for choosing the most straightforward solutions over overly complex ones. Simple solutions are inherently easier to understand, maintain, debug, and extend. This reduces cognitive load for developers and minimizes the likelihood of introducing subtle bugs.

### YAGNI (You Aren't Gonna Need It)

YAGNI is a principle from Extreme Programming that advises against adding functionality until it is truly needed. Building features on speculation often leads to wasted effort, increased complexity, and code that may never be used. Instead, focus on implementing only the functionality required by current needs, allowing the design to evolve organically as requirements become clearer.

### Design Principles

Adherence to established software design principles is crucial for building scalable and resilient systems:

-   **Dependency Inversion Principle (DIP)**: High-level modules should not depend on low-level modules; both should depend on abstractions. This promotes loose coupling and makes the system more flexible and testable.
-   **Open/Closed Principle (OCP)**: Software entities (classes, modules, functions, etc.) should be open for extension but closed for modification. This means that new functionality should be added by extending existing code rather than altering it, thereby preventing regressions and simplifying maintenance.
-   **Single Responsibility Principle (SRP)**: Each function, class, and module should have one clear, well-defined purpose or responsibility. This enhances cohesion, reduces complexity, and makes components easier to understand, test, and reuse.
-   **Fail Fast**: This principle dictates that errors should be detected and reported as early as possible. Instead of attempting to recover from an invalid state, the system should immediately raise an exception or terminate, preventing further corruption and simplifying debugging.

## 🔄 Project Awareness & Context

Understanding the project's overarching context and current state is fundamental for effective collaboration and development. This section outlines practices to ensure all developers are aligned with project goals and progress.

-   **Always read `PLANNING.md`**: At the start of any new conversation or task, it is imperative to review `PLANNING.md`. This document serves as the single source of truth for the project's architecture, strategic goals, coding style, and critical constraints. A thorough understanding of `PLANNING.md` prevents misaligned efforts and ensures adherence to the project's vision.
-   **Check `TASK.md`**: Before commencing any new development task, consult `TASK.md`. This file tracks ongoing and pending work. If the task you intend to work on is not listed, add it with a brief, descriptive title and the current date. This practice maintains transparency in project progress and helps in organizing individual contributions.
-   **Use consistent naming conventions, file structure, and architecture patterns**: As detailed in `PLANNING.md`, maintaining consistency across the codebase is vital. This includes consistent naming for variables, functions, classes, and files, as well as adhering to predefined directory structures and architectural patterns. Consistency significantly improves code readability, navigability, and maintainability, especially in collaborative environments.
-   **Use `uv run` for Python commands**: All Python commands, including script execution and unit tests, must be run within the `backend/.venv` environment using `uv run`. For example, `cd backend && uv run python script.py` or `cd backend && uv run pytest`. This ensures that all developers operate within the same isolated and correctly configured virtual environment, preventing dependency conflicts and ensuring reproducible builds.

## 🧱 Code Structure & Modularity

Well-structured and modular code is easier to understand, test, and maintain. This section provides guidelines for organizing Python code within Claude Code and WARP ADE projects.

### File and Function Limits

To promote readability and manageability, strict limits are placed on code file and function sizes:

-   **Never create a file longer than 500 lines of code**: If a file approaches or exceeds this limit, it indicates a need for refactoring. Break down the file into smaller, more focused modules or helper files, each with a single, clear responsibility. This practice enhances readability and simplifies debugging.
-   **Functions should be under 50 lines**: Each function should ideally perform one specific task. Functions exceeding 50 lines often indicate multiple responsibilities and should be refactored into smaller, more granular functions.
-   **Classes should be under 100 lines**: Classes should represent a single concept or entity. Large classes often violate the Single Responsibility Principle and should be broken down into smaller, more cohesive classes.
-   **Line length should be max 100 characters**: This is enforced by `ruff` (a linter and formatter) in `pyproject.toml`. Adhering to a consistent line length improves code readability, especially when reviewing code side-by-side.

### Project Architecture

Projects should follow a strict vertical slice architecture, where features are organized end-to-end, and tests reside alongside the code they validate. This promotes clear ownership and simplifies navigation. An example structure is as follows:

```
src/project/
    __init__.py
    main.py
    tests/
        test_main.py
    conftest.py

    # Core modules
    database/
        __init__.py
        connection.py
        models.py
        tests/
            test_connection.py
            test_models.py

    auth/
        __init__.py
        authentication.py
        authorization.py
        tests/
            test_authentication.py
            test_authorization.py

    # Feature slices
    features/
        user_management/
            __init__.py
            handlers.py
            validators.py
            tests/
                test_handlers.py
                test_validators.py

        payment_processing/
            __init__.py
            processor.py
            gateway.py
            tests/
                test_processor.py
                test_gateway.py
```

### Modularity and Imports

-   **Organize code into clearly separated modules**: Group modules by feature or responsibility. For agent-based systems, a typical structure includes:
    -   `agent.py`: Contains the main agent definition and its execution logic.
    -   `tools.py`: Houses tool functions utilized by the agent.
    -   `prompts.py`: Stores system prompts and other prompt-related configurations.
-   **Use clear, consistent imports**: Prefer relative imports within packages to maintain modularity and avoid potential circular dependencies. This also makes the codebase more portable and easier to refactor.
-   **Environment Variables**: Utilize `python-dotenv` and `load_dotenv()` for managing environment variables. Sensitive information and configuration values should never be hardcoded directly into the codebase. This practice enhances security and simplifies deployment across different environments.

## 🛠️ Development Environment

Efficient development relies on a well-configured and consistent environment. This section details the tools and commands necessary for setting up and managing your development workflow.

### UV Package Management

This project leverages `uv` for its speed and efficiency in Python package and environment management. `uv` streamlines dependency resolution and virtual environment operations.

To get started with `uv`:

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Sync dependencies (install/update packages based on pyproject.toml)
uv sync

# Add a package (NEVER UPDATE A DEPENDENCY DIRECTLY IN PYPROJECT.toml)
# ALWAYS USE UV ADD to ensure pyproject.toml is updated correctly
uv add requests

# Add development dependency
uv add --dev pytest ruff mypy

# Remove a package
uv remove requests

# Run commands in the environment
uv run python script.py
uv run pytest
uv run ruff check .

# Install specific Python version (if needed)
uv python install 3.12
```

### Development Commands

Standardized commands are used for common development tasks to ensure consistency across the team:

```bash
# Run all tests
uv run pytest

# Run specific tests with verbose output
uv run pytest tests/test_module.py -v

# Run tests with coverage report (HTML output)
uv run pytest --cov=src --cov-report=html

# Format code using ruff
uv run ruff format .

# Check linting issues using ruff
uv run ruff check .

# Fix linting issues automatically using ruff
uv run ruff check --fix .

# Perform static type checking using mypy
uv run mypy src/

# Run pre-commit hooks (if configured)
uv run pre-commit run --all-files
```

## 📋 Style & Conventions

Consistent coding style and conventions are crucial for readability, maintainability, and collaborative development. This section outlines the specific style guidelines to be followed.

### Python Style Guide

-   **Follow PEP 8**: Adhere to the official Python style guide. Specific deviations or preferences are noted below.
    -   **Line length**: Maximum 100 characters, enforced by `ruff` in `pyproject.toml`.
    -   **Quotes**: Use double quotes for strings consistently.
    -   **Trailing commas**: Use trailing commas in multi-line structures (e.g., lists, dictionaries, function calls) to simplify diffs and reordering.
-   **Always use type hints**: Provide type hints for all function signatures and class attributes. This improves code clarity, enables static analysis, and helps catch type-related errors early.
-   **Format with `ruff format`**: Use `ruff format` as the primary code formatter. It is a faster alternative to Black and ensures consistent formatting across the codebase.
-   **Data Validation**: Utilize `pydantic` v2 for all data validation and settings management. Pydantic provides robust, type-hint-based data validation, which is essential for API request/response models and configuration.
-   **API Framework**: Use `FastAPI` for building RESTful APIs. FastAPI offers high performance and automatic interactive API documentation.
-   **ORM/Database**: When a relational database is required, use `SQLAlchemy` or `SQLModel` (which builds on SQLAlchemy and Pydantic) for Object-Relational Mapping. Prefer `PostgreSQL` over other SQL engines due to its robustness, advanced features, and widespread adoption.
-   **Caching**: Implement `Redis` for caching mechanisms, especially when interacting with SQL databases. This significantly improves application performance by reducing database load.

### Docstring Standards

Every public function, class, and module must include a docstring following the Google style. Docstrings are vital for code documentation and auto-generation of API documentation.

```python
def calculate_discount(
    price: Decimal,
    discount_percent: float,
    min_amount: Decimal = Decimal("0.01")
) -> Decimal:
    """
    Calculate the discounted price for a product.

    Args:
        price: Original price of the product
        discount_percent: Discount percentage (0-100)
        min_amount: Minimum allowed final price

    Returns:
        Final price after applying discount

    Raises:
        ValueError: If discount_percent is not between 0 and 100
        ValueError: If final price would be below min_amount

    Example:
        >>> calculate_discount(Decimal("100"), 20)
        Decimal(\'80.00\')
    """
```

### Naming Conventions

Adhere to the following naming conventions for clarity and consistency:

-   **Variables and functions**: `snake_case` (e.g., `user_name`, `get_data`)
-   **Classes**: `PascalCase` (e.g., `UserManager`, `ProductModel`)
-   **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`, `DEFAULT_TIMEOUT`)
-   **Private attributes/methods**: `_leading_underscore` (e.g., `_internal_method`)
-   **Type aliases**: `PascalCase` (e.g., `UserId`, `ProductList`)
-   **Enum values**: `UPPER_SNAKE_CASE` (e.g., `STATUS_ACTIVE`, `TYPE_ADMIN`)

## 🧪 Testing Strategy

Comprehensive testing is integral to ensuring the reliability and correctness of the codebase. This section outlines the testing philosophy and best practices.

### Test-Driven Development (TDD)

TDD is the preferred development methodology. It involves writing tests before writing the code that fulfills those tests. The TDD cycle is as follows:

1.  **Write the test first**: Define the expected behavior of a new feature or fix by writing a failing test case.
2.  **Watch it fail**: Run the test to confirm that it fails as expected. This validates that the test itself is correctly written and that the functionality does not yet exist.
3.  **Write minimal code**: Implement just enough code to make the failing test pass. Focus solely on satisfying the test requirements.
4.  **Refactor**: Once the test passes, refactor the newly written code to improve its design, readability, and efficiency, ensuring all tests remain green.
5.  **Repeat**: Continue this cycle, adding one test at a time, until the feature is complete.

### Testing Best Practices

-   **Always create Pytest unit tests for new features**: Every new function, class, API route, or significant logic block must be accompanied by unit tests. Untested code is considered unstable and unreliable.
-   **Update existing unit tests**: After modifying any existing logic, review and update relevant unit tests to reflect the changes in behavior or expected outcomes. This ensures the test suite remains accurate and effective.
-   **Test organization**: Tests should reside in a `/tests` folder that mirrors the main application structure. This makes it intuitive to locate tests for specific modules or features.
-   **Comprehensive test cases**: For each feature, include at least three types of test cases:
    -   **Expected Use**: A test for the primary, intended use case with valid inputs.
    -   **Edge Case**: A test for unusual or boundary conditions, such as empty inputs, maximum/minimum values, or specific scenarios that might break the logic.
    -   **Failure Case**: A test for invalid inputs or error conditions, ensuring the system handles errors gracefully and provides appropriate feedback.

```python
# Always use pytest fixtures for setup to ensure test isolation and reusability
import pytest
from datetime import datetime
from pydantic import ValidationError # Assuming pydantic is used for validation

@pytest.fixture
def sample_user():
    """Provide a sample user for testing."""
    # Example User model (replace with actual model definition)
    class User:
        def __init__(self, id, name, email, created_at):
            self.id = id
            self.name = name
            self.email = email
            self.created_at = created_at

        def update_email(self, new_email):
            # Simple email validation for example
            if "@" not in new_email or "." not in new_email:
                raise ValidationError("Invalid email format")
            self.email = new_email

    return User(
        id=123,
        name="Test User",
        email="test@example.com",
        created_at=datetime.now()
    )

# Use descriptive test names that clearly indicate what is being tested
def test_user_can_update_email_when_valid(sample_user):
    """Test that users can update their email with valid input."""
    new_email = "newemail@example.com"
    sample_user.update_email(new_email)
    assert sample_user.email == new_email

# Test edge cases and error conditions to ensure robustness
def test_user_update_email_fails_with_invalid_format(sample_user):
    """Test that invalid email formats are rejected."""
    with pytest.raises(ValidationError) as exc_info:
        sample_user.update_email("not-an-email")
    assert "Invalid email format" in str(exc_info.value)
```

### Test Organization

-   **Unit tests**: Focus on testing individual functions or methods in isolation, mocking external dependencies.
-   **Integration tests**: Verify the interactions between different components or modules of the system.
-   **End-to-end tests**: Simulate complete user workflows to ensure the entire system functions correctly from start to finish.
-   **Co-location of tests**: Keep test files (`test_*.py`) next to the code they test within the `src/project/` structure. This improves discoverability and relevance.
-   **Shared fixtures**: Use `conftest.py` files within test directories to define and share Pytest fixtures across multiple test files.
-   **Code coverage**: Aim for 80%+ code coverage, but prioritize testing critical paths and complex logic over achieving 100% coverage on trivial code.

## 🚨 Error Handling

Robust error handling is essential for building resilient applications. This section outlines best practices for managing exceptions and logging within Claude Code and WARP ADE projects.

### Exception Best Practices

-   **Create custom exceptions**: Define custom exception classes for domain-specific errors. This provides more granular control over error types and improves the clarity of error messages.

```python
# Example of custom exceptions for a payment system
class PaymentError(Exception):
    """Base exception for payment-related errors."""
    pass

class InsufficientFundsError(PaymentError):
    """Raised when account has insufficient funds."""



    def __init__(self, required: Decimal, available: Decimal):
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient funds: required {required}, available {available}"
        )

# Use specific exception handling to differentiate between error types
try:
    process_payment(amount)
except InsufficientFundsError as e:
    logger.warning(f"Payment failed: {e}")
    return PaymentResult(success=False, reason="insufficient_funds")
except PaymentError as e:
    logger.error(f"Payment error: {e}")
    return PaymentResult(success=False, reason="payment_error")

# Use context managers for resource management to ensure proper cleanup
from contextlib import contextmanager

@contextmanager
def database_transaction():
    """Provide a transactional scope for database operations."""
    conn = get_connection()
    trans = conn.begin_transaction()
    try:
        yield conn
        trans.commit()
    except Exception:
        trans.rollback()
        raise
    finally:
        conn.close()
```

### Logging Strategy

Effective logging is critical for monitoring application behavior, debugging issues, and understanding system performance. Implement a structured logging strategy:

```python
import logging
from functools import wraps

# Configure structured logging for clarity and ease of parsing
logging.basicConfig(
    level=logging.INFO,
    format=\'%(asctime)s - %(name)s - %(levelname)s - %(message)s\'
)

logger = logging.getLogger(__name__)

# Log function entry/exit for debugging and performance tracing
def log_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Entering {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func.__name__} successfully")
            return result
        except Exception as e:
            logger.exception(f"Error in {func.__name__}: {e}")
            raise
    return wrapper
```

## 🔧 Configuration Management

Proper configuration management ensures that applications can be easily deployed and adapted to different environments without code changes. This section outlines the use of Pydantic for managing application settings.

### Environment Variables and Settings

Utilize Pydantic for defining and validating application settings, drawing values from environment variables or `.env` files. This provides a robust and type-safe way to manage configurations.

```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with validation."""
    app_name: str = "MyApp"
    debug: bool = False
    database_url: str
    redis_url: str = "redis://localhost:6379"
    api_key: str
    max_connections: int = 100

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance to avoid repeated loading."""
    return Settings()

# Usage example:
settings = get_settings()
print(f"Application Name: {settings.app_name}")
```

## 🏗️ Data Models and Validation

Data integrity and consistency are critical. Pydantic models are used extensively for data validation, ensuring that data conforms to expected schemas, especially for API inputs and outputs.

### Example Pydantic Models (strict with Pydantic v2)

Define clear and strict data models using Pydantic, including validation rules and configuration for serialization/deserialization.

```python
from pydantic import BaseModel, Field, validator, EmailStr, ConfigDict
from datetime import datetime
from typing import Optional, List
from decimal import Decimal

class ProductBase(BaseModel):
    """Base product model with common fields."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    price: Decimal = Field(..., gt=0, decimal_places=2)
    category: str
    tags: List[str] = []

    @validator(\'price\')
    def validate_price(cls, v):
        if v > Decimal(\'1000000\'):
            raise ValueError(\'Price cannot exceed 1,000,000\')
        return v

    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }

class ProductCreate(ProductBase):
    """Model for creating new products."""
    pass

class ProductUpdate(BaseModel):
    """Model for updating products - all fields optional."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    price: Optional[Decimal] = Field(None, gt=0, decimal_places=2)
    category: Optional[str] = None
    tags: Optional[List[str]] = None

class Product(ProductBase):
    """Complete product model with database fields."""
    id: int
    created_at: datetime
    updated_at: datetime
    is_active: bool = True

    model_config = ConfigDict(
        from_attributes=True  # Enable ORM mode for compatibility with ORMs
    )
```

## 🔄 Git Workflow

A consistent Git workflow is essential for collaborative development, version control, and maintaining a clean commit history.

### Branch Strategy

Adopt a clear branching strategy to manage development, features, fixes, and releases:

-   `main`: Production-ready code. Only stable, tested code should be merged here.
-   `develop`: Integration branch for ongoing feature development. All new features are merged into `develop` before being released to `main`.
-   `feature/*`: Branches for new features (e.g., `feature/add-user-profile`).
-   `fix/*`: Branches for bug fixes (e.g., `fix/login-issue`).
-   `docs/*`: Branches for documentation updates (e.g., `docs/update-api-guide`).
-   `refactor/*`: Branches for code refactoring efforts (e.g., `refactor/database-layer`).
-   `test/*`: Branches for test additions or fixes (e.g., `test/add-payment-tests`).

### Commit Message Format

Follow a standardized commit message format to ensure clarity, traceability, and automated changelog generation. **Crucially, never include 


Claude code, or written by Claude code in commit messages.**

```
<type>(<scope>): <subject>

<body>

<footer>
```

-   **Types**: `feat` (new feature), `fix` (bug fix), `docs` (documentation only changes), `style` (formatting, missing semicolons, etc.; no code change), `refactor` (a code change that neither fixes a bug nor adds a feature), `test` (adding missing tests or correcting existing tests), `chore` (other changes that don't modify src or test files)
-   **Scope**: Optional, specifies the part of the codebase affected (e.g., `auth`, `database`, `user-management`).
-   **Subject**: Concise, imperative mood, 50 characters max, starts with lowercase.
-   **Body**: Optional, provides a more detailed explanation of the commit.
-   **Footer**: Optional, for breaking changes, references to issues (e.g., `Closes #123`).

**Example Commit Message:**

```
feat(auth): add two-factor authentication

- Implement TOTP generation and validation
- Add QR code generation for authenticator apps
- Update user model with 2FA fields

Closes #123
```

## 🗄️ Database Naming Standards

Consistent database naming conventions are vital for schema readability, maintainability, and effective collaboration. These standards apply to table names, primary keys, foreign keys, and field names.

### Entity-Specific Primary Keys

All database tables must use entity-specific primary keys for clarity and consistency. This means the primary key column name should reflect the entity it identifies (e.g., `session_id` for the `sessions` table, `lead_id` for the `leads` table).

```sql
-- ✅ STANDARDIZED: Entity-specific primary keys
sessions.session_id UUID PRIMARY KEY
leads.lead_id UUID PRIMARY KEY
messages.message_id UUID PRIMARY KEY
daily_metrics.daily_metric_id UUID PRIMARY KEY
agencies.agency_id UUID PRIMARY KEY
```

### Field Naming Conventions

Adhere to the following conventions for all database field names:

-   **Primary keys**: `{entity}_id` (e.g., `session_id`, `lead_id`, `message_id`)
-   **Foreign keys**: `{referenced_entity}_id` (e.g., `session_id` REFERENCES `sessions(session_id)`, `agency_id` REFERENCES `agencies(agency_id)`)
-   **Timestamps**: `{action}_at` (e.g., `created_at`, `updated_at`, `started_at`, `expires_at`)
-   **Booleans**: `is_{state}` (e.g., `is_connected`, `is_active`, `is_qualified`)
-   **Counts**: `{entity}_count` (e.g., `message_count`, `lead_count`, `notification_count`)
-   **Durations**: `{property}_{unit}` (e.g., `duration_seconds`, `timeout_minutes`)

### Repository Pattern Auto-Derivation

The enhanced `BaseRepository` automatically derives table names and primary keys based on the entity model, promoting convention over configuration.

```python
# ✅ STANDARDIZED: Convention-based repositories
class LeadRepository(BaseRepository[Lead]):
    def __init__(self):
        super().__init__()  # Auto-derives "leads" and "lead_id"

class SessionRepository(BaseRepository[AvatarSession]):
    def __init__(self):
        super().__init__()  # Auto-derives "sessions" and "session_id"
```

**Benefits of this approach**:

-   ✅ **Self-documenting schema**: Database structure is intuitive and easy to understand.
-   ✅ **Clear foreign key relationships**: Relationships between tables are explicitly defined and easily identifiable.
-   ✅ **Eliminates repository method overrides**: Reduces boilerplate code by leveraging conventions.
-   ✅ **Consistent with entity naming patterns**: Ensures uniformity across the application.

### Model-Database Alignment

Pydantic models should mirror database fields exactly to eliminate field mapping complexity and ensure a single source of truth for data structure.

```python
# ✅ STANDARDIZED: Models mirror database exactly
from uuid import UUID, uuid4
from datetime import datetime, UTC
from pydantic import BaseModel, Field, ConfigDict

class Lead(BaseModel):
    lead_id: UUID = Field(default_factory=uuid4)  # Matches database field
    session_id: UUID                               # Matches database field
    agency_id: str                                 # Matches database field
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = ConfigDict(
        use_enum_values=True,
        populate_by_name=True,
        alias_generator=None  # Use exact field names
    )
```

### API Route Standards

API routes should be RESTful and follow consistent naming conventions for endpoints and parameters.

```python
# ✅ STANDARDIZED: RESTful with consistent parameter naming
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/leads", tags=["leads"])

@router.get("/{lead_id}")           # GET /api/v1/leads/{lead_id}
@router.put("/{lead_id}")           # PUT /api/v1/leads/{lead_id}
@router.delete("/{lead_id}")        # DELETE /api/v1/leads/{lead_id}

# Sub-resources
@router.get("/{lead_id}/messages")  # GET /api/v1/leads/{lead_id}/messages
@router.get("/agency/{agency_id}")  # GET /api/v1/leads/agency/{agency_id}
```

For complete naming standards, refer to `NAMING_CONVENTIONS.md`.

## 📝 Documentation Standards

Comprehensive and up-to-date documentation is crucial for onboarding new team members, facilitating collaboration, and ensuring long-term maintainability. This section outlines the standards for both code and API documentation.

### Code Documentation

-   **Module Docstrings**: Every Python module (`.py` file) should have a docstring at the top explaining its purpose, contents, and any important considerations.
-   **Public Functions and Classes**: All public functions, methods, and classes must have complete docstrings following the Google style, as detailed in the 


Docstring Standards section. These docstrings should clearly describe their purpose, arguments, return values, and any exceptions raised.
-   **Inline Comments**: Use inline comments (`#`) to explain non-obvious code logic or complex algorithms. If the *why* behind a piece of code is not immediately apparent, add an inline `# Reason:` comment to clarify the design decision or rationale.
-   **`README.md`**: The `README.md` file in the project root must be kept updated. It should include comprehensive setup instructions, how to run tests, how to contribute, and a high-level overview of the project's features and architecture. Update it whenever new features are added, dependencies change, or setup steps are modified.
-   **`CHANGELOG.md`**: Maintain a `CHANGELOG.md` file to document all significant changes, new features, bug fixes, and breaking changes for each version release. This provides a clear version history for the project.

### API Documentation

API endpoints should be self-documenting and provide clear, concise descriptions for consumers. FastAPI automatically generates OpenAPI documentation from type hints and docstrings, but additional details can be provided.

```python
from fastapi import APIRouter, HTTPException, status
from typing import List, Optional

router = APIRouter(prefix="/products", tags=["products"])

@router.get(
    "/",
    response_model=List[Product],
    summary="List all products",
    description="Retrieve a paginated list of all active products with optional filtering by category."
)
async def list_products(
    skip: int = 0,
    limit: int = 100,
    category: Optional[str] = None
) -> List[Product]:
    """
    Retrieve products with optional filtering and pagination.

    - **skip**: Number of products to skip (for pagination).
    - **limit**: Maximum number of products to return.
    - **category**: Optional filter to retrieve products belonging to a specific category.
    """
    # Implementation here to fetch products from the database
    # Example: return await product_service.get_products(skip=skip, limit=limit, category=category)
```

## 🚀 Performance Considerations

Optimizing application performance is crucial for responsiveness and scalability. This section outlines strategies and tools for identifying and addressing performance bottlenecks.

### Optimization Guidelines

-   **Profile before optimizing**: Never optimize prematurely. Use profiling tools like `cProfile` (built-in Python profiler) or `py-spy` (sampling profiler for Python programs) to identify actual performance bottlenecks before attempting any optimizations. Focus on optimizing the slowest parts of the application.
-   **`lru_cache` for expensive computations**: For functions with expensive computations and frequently repeated inputs, use `functools.lru_cache` to memoize results. This can significantly reduce redundant calculations.
-   **Prefer generators for large datasets**: When processing large datasets, use generators instead of loading the entire dataset into memory. Generators yield items one by one, reducing memory footprint and improving performance for streaming data.
-   **`asyncio` for I/O-bound operations**: For operations that involve waiting for external resources (e.g., network requests, database queries, file I/O), use `asyncio` for asynchronous programming. This allows the application to perform other tasks while waiting, improving concurrency and responsiveness.
-   **`multiprocessing` for CPU-bound tasks**: For tasks that heavily utilize the CPU (e.g., complex calculations, data processing), use the `multiprocessing` module to distribute the workload across multiple CPU cores. This can lead to significant speedups for CPU-intensive operations.
-   **Cache database queries appropriately**: Implement caching strategies for frequently accessed data from the database using tools like Redis. This reduces the number of database round-trips and improves data retrieval times.

### Example Optimization

```python
from functools import lru_cache
import asyncio
from typing import AsyncIterator
import json
import aiofiles # Assuming aiofiles is installed for async file operations

@lru_cache(maxsize=1000)
def expensive_calculation(n: int) -> int:
    """
    Cache results of expensive calculations.
    This function simulates a CPU-bound task.
    """
    # Example: Fibonacci sequence calculation
    if n <= 1:
        return n
    return expensive_calculation(n-1) + expensive_calculation(n-2)

async def process_large_dataset() -> AsyncIterator[dict]:
    """
    Process large dataset without loading all into memory.
    This function simulates reading and processing a large JSONL file.
    """
    try:
        async with aiofiles.open(\'large_file.jsonl\', mode=\'r\') as f:
            async for line in f:
                data = json.loads(line)
                # Process each item (e.g., perform some transformation)
                yield data
    except FileNotFoundError:
        print("Error: large_file.jsonl not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from line: {e}")
        return

# Example usage of the optimized functions:
async def main():
    # Using lru_cache
    print(f"Expensive calculation for 30: {expensive_calculation(30)}")
    print(f"Expensive calculation for 30 (cached): {expensive_calculation(30)}")

    # Processing large dataset with generator
    print("\nProcessing large dataset:")
    async for item in process_large_dataset():
        print(f"Processed item: {item}")

if __name__ == "__main__":
    # To run this example, you might need to create a dummy large_file.jsonl
    # and install aiofiles: pip install aiofiles
    # Example large_file.jsonl content:
    # {"id": 1, "value": "a"}
    # {"id": 2, "value": "b"}
    # {"id": 3, "value": "c"}
    asyncio.run(main())
```

This concludes the Enhanced Claude Code & WARP ADE Guide. By adhering to these principles and practices, developers can contribute to a codebase that is robust, maintainable, scalable, and a pleasure to work with.


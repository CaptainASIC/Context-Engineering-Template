## FEATURE:

- Pydantic AI agent that has another Pydantic AI agent as a tool.
- Research Agent for the primary agent and then an email draft Agent for the subagent.
- CLI to interact with the agent.
- Gmail for the email draft agent, Brave API for the research agent.

## EXAMPLES:

In the `examples/` folder, reference materials and templates will be provided:

- `examples/react_frontend/` - React application structure with modern UI components and responsive design
- `examples/fastapi_backend/` - FastAPI application template with proper routing, middleware, and API documentation
- `examples/admin_portal/` - AI Council admin interface for policy management, uploads, and permanent statements
- `examples/auth_rbac/` - Role-based access control implementation with user authentication
- `examples/rag_agent/` - PydanticAI agent implementation for RAG functionality with multi-LLM support (Gemini, Anthropic, OpenAI, OpenRouter, Ollama)
- `examples/knowledge_graph/` - Neo4j integration patterns for connecting policies, use cases, and people
- `examples/vector_db/` - Neon database setup and pgvector semantic search implementation
- `examples/memory_management/` - mem0 integration for conversation context and user preference storage
- `examples/multilingual/` - Language detection and response generation patterns
- `examples/content_management/` - Policy upload, versioning, and permanent statement management
- `examples/use_case_radar/` - Status tracking and visualization components using React
- `examples/llm_providers/` - Multi-LLM provider integration with fallback mechanisms
- `examples/global_deployment/` - Multi-region deployment patterns and configuration

Use these examples as templates and best practices, adapting them specifically for this projects requirements.

## DOCUMENTATION:

Pydantic AI documentation: https://ai.pydantic.dev/

## OTHER CONSIDERATIONS:

- Include a .env.example, README with instructions for setup including how to configure Gmail and Brave.
- Include the project structure in the README.
- Virtual environment has already been set up with the necessary dependencies.
- Use python_dotenv and load_env() for environment variables

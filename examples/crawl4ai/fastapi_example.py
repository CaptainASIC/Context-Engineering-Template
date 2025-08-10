#!/usr/bin/env python3
"""
FastAPI Integration Example for Crawl4AI

This example demonstrates how to integrate Crawl4AI with FastAPI using
a modular architecture. The implementation is split across multiple files
to maintain clean, manageable code under 500 lines per file.

Files:
- models.py: Pydantic models for request/response validation
- crawler_service.py: Crawler lifecycle and utility functions
- routes.py: API endpoint definitions
- background_tasks.py: Asynchronous task management

Usage:
    pip install fastapi uvicorn crawl4ai
    python fastapi_example.py
    
    Then visit: http://localhost:8000/docs for API documentation

Example requests:
    # Simple crawl
    curl -X POST "http://localhost:8000/crawl" \
         -H "Content-Type: application/json" \
         -d '{"url": "https://github.com/CaptainASIC"}'
    
    # Crawl with screenshot
    curl -X POST "http://localhost:8000/crawl" \
         -H "Content-Type: application/json" \
         -d '{"url": "https://github.com/CaptainASIC", "screenshot": true}'
    
    # Batch crawl
    curl -X POST "http://localhost:8000/batch-crawl" \
         -H "Content-Type: application/json" \
         -d '{"urls": ["https://github.com/CaptainASIC", "https://github.com/CaptainASIC/repositories"]}'
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi_integration import router, get_crawler_lifespan


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="Crawl4AI API Service",
        description="Web crawling API powered by Crawl4AI - Modular Implementation",
        version="1.0.0",
        lifespan=get_crawler_lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(router)
    
    return app


# Create the FastAPI app
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Crawl4AI FastAPI Service (Modular)")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🎯 Example target: https://github.com/CaptainASIC")
    print("📁 Modular structure:")
    print("  - models.py: Request/Response models")
    print("  - crawler_service.py: Crawler management")
    print("  - routes.py: API endpoints")
    print("  - background_tasks.py: Async task handling")
    
    uvicorn.run(
        "fastapi_example:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


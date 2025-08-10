"""
Crawl4AI FastAPI Integration Package

This package provides a modular FastAPI integration for Crawl4AI.
"""

from .models import *
from .crawler_service import crawler_service, get_crawler_lifespan
from .routes import router
from .background_tasks import task_manager

__all__ = [
    'crawler_service',
    'get_crawler_lifespan', 
    'router',
    'task_manager'
]


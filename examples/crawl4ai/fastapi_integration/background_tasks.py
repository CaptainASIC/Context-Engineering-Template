#!/usr/bin/env python3
"""
Background Tasks Module for Crawl4AI FastAPI Integration

This module handles asynchronous task management and execution.
"""

from datetime import datetime
from typing import Dict, Optional

from .models import TaskStatus, CrawlRequest
from .crawler_service import crawler_service


class TaskManager:
    """Manages background crawling tasks"""
    
    def __init__(self):
        self.tasks: Dict[str, TaskStatus] = {}
    
    def create_task(self, task_id: str) -> TaskStatus:
        """Create a new task"""
        task = TaskStatus(
            task_id=task_id,
            status="pending",
            progress=0.0,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        self.tasks[task_id] = task
        return task
    
    def get_task(self, task_id: str) -> Optional[TaskStatus]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def update_task(self, task_id: str, **kwargs) -> Optional[TaskStatus]:
        """Update task properties"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        for key, value in kwargs.items():
            if hasattr(task, key):
                setattr(task, key, value)
        
        task.updated_at = datetime.now().isoformat()
        return task
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a task"""
        if task_id in self.tasks:
            del self.tasks[task_id]
            return True
        return False
    
    def get_task_summary(self) -> Dict:
        """Get summary of all tasks"""
        status_counts = {}
        for task in self.tasks.values():
            status_counts[task.status] = status_counts.get(task.status, 0) + 1
        
        return {
            "tasks": list(self.tasks.keys()),
            "total": len(self.tasks),
            "by_status": status_counts
        }


# Global task manager instance
task_manager = TaskManager()


async def perform_async_crawl(task_id: str, request: CrawlRequest):
    """Background task to perform crawling"""
    task = task_manager.get_task(task_id)
    if not task:
        return
    
    try:
        # Update task status
        task_manager.update_task(task_id, status="running", progress=0.1)
        
        if not crawler_service.is_ready():
            raise Exception("Crawler not initialized")
        
        # Update progress
        task_manager.update_task(task_id, progress=0.3)
        
        # Perform crawl
        result = await crawler_service.crawl_single_url(request)
        
        # Update progress
        task_manager.update_task(task_id, progress=0.9)
        
        # Complete task
        task_manager.update_task(
            task_id,
            status="completed",
            progress=1.0,
            result=result.dict()
        )
        
    except Exception as e:
        # Mark task as failed
        task_manager.update_task(
            task_id,
            status="failed",
            error=str(e)
        )


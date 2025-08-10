#!/usr/bin/env python3
"""
API Routes for Crawl4AI FastAPI Integration

This module contains all the API endpoint definitions.
"""

import uuid
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, HTTPException, BackgroundTasks

from .models import (
    CrawlRequest, BatchCrawlRequest, CrawlWithExtractionRequest,
    CrawlResponse, BatchCrawlResponse, TaskStatus
)
from .crawler_service import crawler_service
from .background_tasks import task_manager, perform_async_crawl


# Create API router
router = APIRouter()


@router.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Crawl4AI API Service",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "crawl": "/crawl",
            "batch_crawl": "/batch-crawl",
            "extract": "/extract",
            "status": "/status/{task_id}",
            "health": "/health"
        }
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "crawler_ready": crawler_service.is_ready(),
        "timestamp": datetime.now().isoformat()
    }


@router.post("/crawl", response_model=CrawlResponse)
async def crawl_url(request: CrawlRequest):
    """Crawl a single URL with specified configuration"""
    if not crawler_service.is_ready():
        raise HTTPException(status_code=503, detail="Crawler not initialized")
    
    try:
        return await crawler_service.crawl_single_url(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crawl failed: {str(e)}")


@router.post("/batch-crawl", response_model=BatchCrawlResponse)
async def batch_crawl_urls(request: BatchCrawlRequest):
    """Crawl multiple URLs in batch"""
    if not crawler_service.is_ready():
        raise HTTPException(status_code=503, detail="Crawler not initialized")
    
    start_time = datetime.now()
    results = []
    
    try:
        # Process URLs sequentially for simplicity
        for url in request.urls:
            url_start_time = datetime.now()
            try:
                # Create individual crawl request
                crawl_request = CrawlRequest(
                    url=url,
                    css_selector=request.css_selector,
                    word_count_threshold=request.word_count_threshold,
                    screenshot=request.screenshot,
                    extract_media=request.extract_media,
                    cache_mode=request.cache_mode
                )
                
                result = await crawler_service.crawl_single_url(crawl_request)
                results.append(result)
                
            except Exception as e:
                # Create error response for failed URL
                error_response = CrawlResponse(
                    success=False,
                    url=str(url),
                    content_length=0,
                    markdown_length=0,
                    has_screenshot=False,
                    has_pdf=False,
                    links_count={"internal": 0, "external": 0},
                    media_count=0,
                    error_message=str(e),
                    crawl_time=0.0,
                    timestamp=datetime.now().isoformat()
                )
                results.append(error_response)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        successful_crawls = sum(1 for r in results if r.success)
        failed_crawls = len(results) - successful_crawls
        
        return BatchCrawlResponse(
            total_urls=len(request.urls),
            successful_crawls=successful_crawls,
            failed_crawls=failed_crawls,
            results=results,
            total_time=total_time,
            timestamp=end_time.isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch crawl failed: {str(e)}")


@router.post("/extract")
async def crawl_with_extraction(request: CrawlWithExtractionRequest):
    """Crawl URL with structured content extraction"""
    if not crawler_service.is_ready():
        raise HTTPException(status_code=503, detail="Crawler not initialized")
    
    try:
        return await crawler_service.crawl_with_extraction(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router.post("/crawl-async")
async def crawl_url_async(request: CrawlRequest, background_tasks: BackgroundTasks):
    """Start an asynchronous crawl job and return task ID"""
    task_id = str(uuid.uuid4())
    
    # Create and register task
    task_manager.create_task(task_id)
    
    # Add background task
    background_tasks.add_task(perform_async_crawl, task_id, request)
    
    return {"task_id": task_id, "status": "pending"}


@router.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get status of an asynchronous crawl task"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task


@router.delete("/status/{task_id}")
async def delete_task(task_id: str):
    """Delete a completed or failed task"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status in ["pending", "running"]:
        raise HTTPException(status_code=400, detail="Cannot delete running task")
    
    task_manager.delete_task(task_id)
    return {"message": "Task deleted successfully"}


@router.get("/tasks")
async def list_tasks():
    """List all background tasks"""
    return task_manager.get_task_summary()


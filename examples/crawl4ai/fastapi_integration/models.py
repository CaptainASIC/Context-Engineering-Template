#!/usr/bin/env python3
"""
Pydantic Models for Crawl4AI FastAPI Integration

This module contains all the request and response models used in the FastAPI service.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, HttpUrl, Field, validator


class CrawlRequest(BaseModel):
    """Request model for crawling operations"""
    url: HttpUrl = Field(..., description="URL to crawl")
    css_selector: Optional[str] = Field(None, description="CSS selector for content extraction")
    word_count_threshold: int = Field(10, ge=1, le=1000, description="Minimum word count threshold")
    screenshot: bool = Field(False, description="Capture screenshot")
    pdf: bool = Field(False, description="Generate PDF")
    extract_media: bool = Field(True, description="Extract media information")
    wait_for: Optional[str] = Field(None, description="CSS selector to wait for")
    delay: float = Field(0.0, ge=0.0, le=10.0, description="Delay before returning HTML")
    cache_mode: str = Field("enabled", description="Cache mode: enabled, bypass, read_only, write_only")
    
    @validator('cache_mode')
    def validate_cache_mode(cls, v):
        valid_modes = ['enabled', 'bypass', 'read_only', 'write_only']
        if v not in valid_modes:
            raise ValueError(f'cache_mode must be one of: {valid_modes}')
        return v


class BatchCrawlRequest(BaseModel):
    """Request model for batch crawling operations"""
    urls: List[HttpUrl] = Field(..., min_items=1, max_items=10, description="List of URLs to crawl")
    css_selector: Optional[str] = Field(None, description="CSS selector for content extraction")
    word_count_threshold: int = Field(10, ge=1, le=1000, description="Minimum word count threshold")
    screenshot: bool = Field(False, description="Capture screenshots")
    extract_media: bool = Field(True, description="Extract media information")
    cache_mode: str = Field("enabled", description="Cache mode")


class ExtractionSchema(BaseModel):
    """Model for CSS extraction schema"""
    name: str = Field(..., description="Schema name")
    base_selector: str = Field(..., description="Base CSS selector")
    fields: List[Dict[str, Any]] = Field(..., description="Extraction fields")


class CrawlWithExtractionRequest(BaseModel):
    """Request model for crawling with structured extraction"""
    url: HttpUrl = Field(..., description="URL to crawl")
    extraction_schema: ExtractionSchema = Field(..., description="Extraction schema")
    wait_for: Optional[str] = Field(None, description="CSS selector to wait for")
    delay: float = Field(2.0, ge=0.0, le=10.0, description="Delay before extraction")


class CrawlResponse(BaseModel):
    """Response model for crawling operations"""
    success: bool
    url: str
    title: Optional[str] = None
    content_length: int
    markdown_length: int
    has_screenshot: bool
    has_pdf: bool
    links_count: Dict[str, int]
    media_count: int
    extracted_content: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    crawl_time: float
    timestamp: str


class BatchCrawlResponse(BaseModel):
    """Response model for batch crawling operations"""
    total_urls: int
    successful_crawls: int
    failed_crawls: int
    results: List[CrawlResponse]
    total_time: float
    timestamp: str


class TaskStatus(BaseModel):
    """Model for background task status"""
    task_id: str
    status: str  # pending, running, completed, failed
    progress: float  # 0.0 to 1.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


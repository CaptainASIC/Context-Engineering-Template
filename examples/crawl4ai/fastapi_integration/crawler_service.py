#!/usr/bin/env python3
"""
Crawler Service Module for FastAPI Integration

This module handles the crawler lifecycle and provides utility functions
for crawling operations.
"""

import json
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

from .models import CrawlResponse, CrawlRequest, CrawlWithExtractionRequest


class CrawlerService:
    """Service class to manage crawler operations"""
    
    def __init__(self):
        self.crawler: Optional[AsyncWebCrawler] = None
        self.browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True,
            verbose=False
        )
    
    async def start(self):
        """Initialize and start the crawler"""
        if not self.crawler:
            self.crawler = AsyncWebCrawler(config=self.browser_config)
            await self.crawler.start()
    
    async def stop(self):
        """Stop and cleanup the crawler"""
        if self.crawler:
            await self.crawler.close()
            self.crawler = None
    
    def is_ready(self) -> bool:
        """Check if crawler is ready"""
        return self.crawler is not None
    
    def _get_cache_mode(self, cache_mode_str: str) -> CacheMode:
        """Convert cache mode string to enum"""
        cache_mode_map = {
            "enabled": CacheMode.ENABLED,
            "bypass": CacheMode.BYPASS,
            "read_only": CacheMode.READ_ONLY,
            "write_only": CacheMode.WRITE_ONLY
        }
        return cache_mode_map.get(cache_mode_str, CacheMode.ENABLED)
    
    def create_crawl_response(self, result, start_time: datetime) -> CrawlResponse:
        """Create standardized crawl response"""
        end_time = datetime.now()
        crawl_time = (end_time - start_time).total_seconds()
        
        return CrawlResponse(
            success=result.success,
            url=str(result.url),
            title=result.metadata.get("title"),
            content_length=len(result.cleaned_html) if result.cleaned_html else 0,
            markdown_length=len(result.markdown) if result.markdown else 0,
            has_screenshot=bool(result.screenshot),
            has_pdf=bool(result.pdf),
            links_count={
                "internal": len(result.links.get("internal", [])) if result.links else 0,
                "external": len(result.links.get("external", [])) if result.links else 0
            },
            media_count=len(result.media.get("images", [])) if result.media else 0,
            error_message=result.error_message if not result.success else None,
            crawl_time=crawl_time,
            timestamp=end_time.isoformat()
        )
    
    async def crawl_single_url(self, request: CrawlRequest) -> CrawlResponse:
        """Crawl a single URL"""
        if not self.crawler:
            raise RuntimeError("Crawler not initialized")
        
        start_time = datetime.now()
        
        # Create run configuration
        run_config = CrawlerRunConfig(
            css_selector=request.css_selector,
            word_count_threshold=request.word_count_threshold,
            screenshot=request.screenshot,
            pdf=request.pdf,
            extract_media=request.extract_media,
            wait_for=request.wait_for,
            delay_before_return_html=request.delay,
            cache_mode=self._get_cache_mode(request.cache_mode)
        )
        
        # Perform crawl
        result = await self.crawler.arun(
            url=str(request.url),
            config=run_config
        )
        
        return self.create_crawl_response(result, start_time)
    
    async def crawl_with_extraction(self, request: CrawlWithExtractionRequest) -> CrawlResponse:
        """Crawl URL with structured content extraction"""
        if not self.crawler:
            raise RuntimeError("Crawler not initialized")
        
        start_time = datetime.now()
        
        # Create extraction schema
        extraction_schema = {
            "name": request.extraction_schema.name,
            "baseSelector": request.extraction_schema.base_selector,
            "fields": request.extraction_schema.fields
        }
        
        # Create run configuration with extraction strategy
        run_config = CrawlerRunConfig(
            extraction_strategy=JsonCssExtractionStrategy(extraction_schema),
            wait_for=request.wait_for,
            delay_before_return_html=request.delay
        )
        
        # Perform crawl with extraction
        result = await self.crawler.arun(
            url=str(request.url),
            config=run_config
        )
        
        # Create response with extracted content
        response = self.create_crawl_response(result, start_time)
        
        if result.success and result.extracted_content:
            try:
                response.extracted_content = json.loads(result.extracted_content)
            except json.JSONDecodeError:
                response.extracted_content = {"raw": result.extracted_content}
        
        return response


# Global crawler service instance
crawler_service = CrawlerService()


@asynccontextmanager
async def get_crawler_lifespan(app):
    """Context manager for crawler lifecycle in FastAPI"""
    # Startup
    await crawler_service.start()
    print("🚀 Crawler service initialized")
    
    yield
    
    # Shutdown
    await crawler_service.stop()
    print("🛑 Crawler service stopped")


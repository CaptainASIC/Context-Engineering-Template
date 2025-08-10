#!/usr/bin/env python3
"""
Batch Processing Example for Crawl4AI

This example demonstrates how to process multiple URLs efficiently using Crawl4AI's
batch processing capabilities with resource monitoring and rate limiting.

Features:
- Multiple URL processing
- Resource-aware crawling
- Progress monitoring
- Rate limiting and retry logic
- Error handling and recovery
- Results aggregation and analysis

Target URLs: Various GitHub pages related to CaptainASIC
"""

import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode


class BatchCrawler:
    """Batch crawler with resource management and monitoring"""
    
    def __init__(self, max_concurrent: int = 3, delay_between_requests: float = 1.0):
        self.max_concurrent = max_concurrent
        self.delay_between_requests = delay_between_requests
        self.results: List[Dict[str, Any]] = []
        self.stats = {
            "total_urls": 0,
            "successful": 0,
            "failed": 0,
            "total_time": 0.0,
            "start_time": None,
            "end_time": None
        }
    
    async def crawl_single_url(self, crawler: AsyncWebCrawler, url: str, config: CrawlerRunConfig) -> Dict[str, Any]:
        """Crawl a single URL and return result summary"""
        start_time = time.time()
        
        try:
            result = await crawler.arun(url=url, config=config)
            end_time = time.time()
            
            if result.success:
                return {
                    "url": url,
                    "success": True,
                    "duration": end_time - start_time,
                    "content_length": len(result.cleaned_html),
                    "markdown_length": len(result.markdown),
                    "title": result.metadata.get("title", "N/A"),
                    "links_count": {
                        "internal": len(result.links.get("internal", [])) if result.links else 0,
                        "external": len(result.links.get("external", [])) if result.links else 0
                    },
                    "media_count": len(result.media.get("images", [])) if result.media else 0,
                    "has_screenshot": bool(result.screenshot),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "url": url,
                    "success": False,
                    "duration": end_time - start_time,
                    "error": result.error_message,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            end_time = time.time()
            return {
                "url": url,
                "success": False,
                "duration": end_time - start_time,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def crawl_batch(self, urls: List[str], config: CrawlerRunConfig) -> List[Dict[str, Any]]:
        """Crawl multiple URLs with concurrency control"""
        self.stats["total_urls"] = len(urls)
        self.stats["start_time"] = datetime.now().isoformat()
        
        browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True,
            verbose=False
        )
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Process URLs in batches to control concurrency
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def crawl_with_semaphore(url: str) -> Dict[str, Any]:
                async with semaphore:
                    result = await self.crawl_single_url(crawler, url, config)
                    
                    # Update stats
                    if result["success"]:
                        self.stats["successful"] += 1
                    else:
                        self.stats["failed"] += 1
                    
                    # Progress reporting
                    completed = self.stats["successful"] + self.stats["failed"]
                    progress = (completed / self.stats["total_urls"]) * 100
                    print(f"📊 Progress: {completed}/{self.stats['total_urls']} ({progress:.1f}%) - {url}")
                    
                    # Rate limiting
                    if self.delay_between_requests > 0:
                        await asyncio.sleep(self.delay_between_requests)
                    
                    return result
            
            # Execute all crawls concurrently (but limited by semaphore)
            start_time = time.time()
            self.results = await asyncio.gather(*[crawl_with_semaphore(url) for url in urls])
            end_time = time.time()
            
            self.stats["total_time"] = end_time - start_time
            self.stats["end_time"] = datetime.now().isoformat()
        
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get batch processing summary"""
        if not self.results:
            return {"error": "No results available"}
        
        successful_results = [r for r in self.results if r["success"]]
        failed_results = [r for r in self.results if not r["success"]]
        
        summary = {
            "batch_stats": self.stats.copy(),
            "performance": {
                "average_duration": sum(r["duration"] for r in self.results) / len(self.results),
                "fastest_crawl": min(self.results, key=lambda x: x["duration"])["duration"],
                "slowest_crawl": max(self.results, key=lambda x: x["duration"])["duration"],
                "success_rate": (len(successful_results) / len(self.results)) * 100
            },
            "content_stats": {
                "total_content_length": sum(r.get("content_length", 0) for r in successful_results),
                "average_content_length": sum(r.get("content_length", 0) for r in successful_results) / max(len(successful_results), 1),
                "total_links": sum(r.get("links_count", {}).get("internal", 0) + r.get("links_count", {}).get("external", 0) for r in successful_results),
                "total_media": sum(r.get("media_count", 0) for r in successful_results)
            },
            "errors": [{"url": r["url"], "error": r["error"]} for r in failed_results]
        }
        
        return summary


async def basic_batch_example():
    """Basic batch processing example"""
    print("=== Basic Batch Processing Example ===")
    
    # URLs to crawl
    urls = [
        "https://github.com/CaptainASIC",
        "https://github.com/CaptainASIC?tab=repositories",
        "https://github.com/CaptainASIC?tab=stars",
        "https://github.com/CaptainASIC?tab=followers"
    ]
    
    # Basic configuration
    config = CrawlerRunConfig(
        word_count_threshold=10,
        extract_media=True,
        cache_mode=CacheMode.ENABLED
    )
    
    # Create batch crawler
    batch_crawler = BatchCrawler(max_concurrent=2, delay_between_requests=0.5)
    
    print(f"🚀 Starting batch crawl of {len(urls)} URLs...")
    results = await batch_crawler.crawl_batch(urls, config)
    
    # Display results
    summary = batch_crawler.get_summary()
    print(f"\n📊 Batch Processing Summary:")
    print(f"  Total URLs: {summary['batch_stats']['total_urls']}")
    print(f"  Successful: {summary['batch_stats']['successful']}")
    print(f"  Failed: {summary['batch_stats']['failed']}")
    print(f"  Total time: {summary['batch_stats']['total_time']:.2f}s")
    print(f"  Success rate: {summary['performance']['success_rate']:.1f}%")
    print(f"  Average duration: {summary['performance']['average_duration']:.2f}s")
    
    # Save results
    with open("basic_batch_results.json", "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)
    print("💾 Basic batch results saved")
    
    return results


async def advanced_batch_example():
    """Advanced batch processing with screenshots and extraction"""
    print("\n=== Advanced Batch Processing Example ===")
    
    # More comprehensive URL list
    urls = [
        "https://github.com/CaptainASIC",
        "https://github.com/CaptainASIC?tab=repositories",
        "https://github.com/CaptainASIC?tab=stars",
        "https://github.com/CaptainASIC?tab=followers",
        "https://github.com/CaptainASIC?tab=following"
    ]
    
    # Advanced configuration with screenshots
    config = CrawlerRunConfig(
        word_count_threshold=5,
        screenshot=True,
        extract_media=True,
        remove_overlay_elements=True,
        cache_mode=CacheMode.BYPASS,
        wait_for="css:body",
        delay_before_return_html=1.0
    )
    
    # Create batch crawler with higher concurrency
    batch_crawler = BatchCrawler(max_concurrent=3, delay_between_requests=1.0)
    
    print(f"🚀 Starting advanced batch crawl of {len(urls)} URLs...")
    results = await batch_crawler.crawl_batch(urls, config)
    
    # Display detailed results
    summary = batch_crawler.get_summary()
    print(f"\n📊 Advanced Batch Processing Summary:")
    print(f"  Total URLs: {summary['batch_stats']['total_urls']}")
    print(f"  Successful: {summary['batch_stats']['successful']}")
    print(f"  Failed: {summary['batch_stats']['failed']}")
    print(f"  Total time: {summary['batch_stats']['total_time']:.2f}s")
    print(f"  Success rate: {summary['performance']['success_rate']:.1f}%")
    print(f"  Total content: {summary['content_stats']['total_content_length']:,} chars")
    print(f"  Total links: {summary['content_stats']['total_links']}")
    print(f"  Total media: {summary['content_stats']['total_media']}")
    
    # Show individual results
    print(f"\n📄 Individual Results:")
    for result in results:
        if result["success"]:
            print(f"  ✅ {result['url']}: {result['duration']:.2f}s ({result['content_length']:,} chars)")
        else:
            print(f"  ❌ {result['url']}: {result['error']}")
    
    # Save screenshots if available
    screenshot_count = 0
    for i, result in enumerate(results):
        if result["success"] and result.get("has_screenshot"):
            screenshot_count += 1
    
    if screenshot_count > 0:
        print(f"📸 {screenshot_count} screenshots captured")
    
    # Save advanced results
    with open("advanced_batch_results.json", "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)
    print("💾 Advanced batch results saved")
    
    return results


async def performance_comparison_example():
    """Compare different batch processing strategies"""
    print("\n=== Batch Performance Comparison ===")
    
    urls = [
        "https://github.com/CaptainASIC",
        "https://github.com/CaptainASIC?tab=repositories",
        "https://github.com/CaptainASIC?tab=stars"
    ]
    
    strategies = [
        {
            "name": "Sequential (No Concurrency)",
            "max_concurrent": 1,
            "delay": 0.0,
            "config": CrawlerRunConfig(word_count_threshold=10, screenshot=False)
        },
        {
            "name": "Low Concurrency",
            "max_concurrent": 2,
            "delay": 0.5,
            "config": CrawlerRunConfig(word_count_threshold=10, screenshot=False)
        },
        {
            "name": "High Concurrency",
            "max_concurrent": 3,
            "delay": 0.0,
            "config": CrawlerRunConfig(word_count_threshold=10, screenshot=False)
        }
    ]
    
    comparison_results = {}
    
    for strategy in strategies:
        print(f"\n🔄 Testing {strategy['name']}...")
        
        batch_crawler = BatchCrawler(
            max_concurrent=strategy["max_concurrent"],
            delay_between_requests=strategy["delay"]
        )
        
        start_time = time.time()
        results = await batch_crawler.crawl_batch(urls, strategy["config"])
        end_time = time.time()
        
        summary = batch_crawler.get_summary()
        comparison_results[strategy["name"]] = {
            "total_time": end_time - start_time,
            "success_rate": summary["performance"]["success_rate"],
            "average_duration": summary["performance"]["average_duration"],
            "successful_crawls": summary["batch_stats"]["successful"]
        }
        
        print(f"  ⏱️  Total time: {end_time - start_time:.2f}s")
        print(f"  ✅ Success rate: {summary['performance']['success_rate']:.1f}%")
    
    # Display comparison
    print(f"\n📊 Performance Comparison Summary:")
    fastest_strategy = min(comparison_results.items(), key=lambda x: x[1]["total_time"])
    print(f"🏆 Fastest strategy: {fastest_strategy[0]} ({fastest_strategy[1]['total_time']:.2f}s)")
    
    for name, stats in comparison_results.items():
        print(f"  {name}: {stats['total_time']:.2f}s (Success: {stats['success_rate']:.1f}%)")
    
    # Save comparison results
    with open("batch_performance_comparison.json", "w") as f:
        json.dump(comparison_results, f, indent=2)
    print("💾 Performance comparison results saved")
    
    return comparison_results


async def main():
    """Run all batch processing examples"""
    print("🚀 Starting Crawl4AI Batch Processing Examples")
    print("🎯 Target: GitHub pages related to CaptainASIC")
    print("=" * 60)
    
    # Create output directory
    Path("batch_output").mkdir(exist_ok=True)
    import os
    os.chdir("batch_output")
    
    # Run batch processing examples
    await basic_batch_example()
    await advanced_batch_example()
    await performance_comparison_example()
    
    print("\n🎉 All batch processing examples completed!")
    print("📁 Check the 'batch_output' directory for generated files")


if __name__ == "__main__":
    asyncio.run(main())


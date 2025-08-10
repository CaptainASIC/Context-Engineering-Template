#!/usr/bin/env python3
"""
Cache Management and Performance Examples for Crawl4AI

This module demonstrates:
- Different cache modes and their performance impact
- Batch processing with resource management
- Performance optimization techniques
- Memory and time monitoring
"""

import asyncio
import time
import json
from datetime import datetime
from typing import List, Dict, Any

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode


async def cache_modes_comparison():
    """Compare performance of different cache modes"""
    print("=== Cache Modes Performance Comparison ===")
    
    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=True
    )
    
    # Test different cache modes
    cache_modes = [
        (CacheMode.ENABLED, "Cache Enabled"),
        (CacheMode.BYPASS, "Cache Bypassed"),
        (CacheMode.READ_ONLY, "Read-Only Cache"),
        (CacheMode.WRITE_ONLY, "Write-Only Cache")
    ]
    
    results = {}
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        for cache_mode, description in cache_modes:
            print(f"🔄 Testing {description}...")
            
            run_config = CrawlerRunConfig(
                cache_mode=cache_mode,
                word_count_threshold=10
            )
            
            try:
                start_time = time.time()
                result = await crawler.arun(
                    url="https://github.com/CaptainASIC",
                    config=run_config
                )
                end_time = time.time()
                
                if result.success:
                    duration = end_time - start_time
                    results[cache_mode.value] = {
                        "success": True,
                        "duration": duration,
                        "content_length": len(result.cleaned_html),
                        "description": description
                    }
                    print(f"  ✅ Success in {duration:.2f}s ({len(result.cleaned_html)} chars)")
                else:
                    results[cache_mode.value] = {
                        "success": False,
                        "error": result.error_message,
                        "description": description
                    }
                    print(f"  ❌ Failed: {result.error_message}")
                    
            except Exception as e:
                results[cache_mode.value] = {
                    "success": False,
                    "error": str(e),
                    "description": description
                }
                print(f"  💥 Exception: {str(e)}")
    
    # Display performance summary
    print("\n📊 Cache Performance Summary:")
    successful_results = {k: v for k, v in results.items() if v["success"]}
    
    if successful_results:
        fastest = min(successful_results.items(), key=lambda x: x[1]["duration"])
        slowest = max(successful_results.items(), key=lambda x: x[1]["duration"])
        
        print(f"🏆 Fastest: {fastest[1]['description']} ({fastest[1]['duration']:.2f}s)")
        print(f"🐌 Slowest: {slowest[1]['description']} ({slowest[1]['duration']:.2f}s)")
        
        for mode, data in results.items():
            if data["success"]:
                print(f"  {data['description']}: {data['duration']:.2f}s")
            else:
                print(f"  {data['description']}: Failed - {data['error']}")
    
    # Save results
    with open("cache_performance_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("💾 Cache performance results saved")
    
    return results


async def batch_processing_example():
    """Demonstrate batch processing with multiple URLs"""
    print("\n=== Batch Processing Example ===")
    
    # URLs to crawl (GitHub related)
    urls = [
        "https://github.com/CaptainASIC",
        "https://github.com/CaptainASIC?tab=repositories",
        "https://github.com/CaptainASIC?tab=stars",
        "https://github.com/CaptainASIC?tab=followers",
        "https://github.com/CaptainASIC?tab=following"
    ]
    
    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=True
    )
    
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED,
        word_count_threshold=5,
        screenshot=False  # Disable screenshots for faster processing
    )
    
    results = []
    start_time = time.time()
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        print(f"🚀 Processing {len(urls)} URLs...")
        
        for i, url in enumerate(urls, 1):
            print(f"📄 Processing {i}/{len(urls)}: {url}")
            
            try:
                url_start_time = time.time()
                result = await crawler.arun(url=url, config=run_config)
                url_end_time = time.time()
                
                url_duration = url_end_time - url_start_time
                
                if result.success:
                    results.append({
                        "url": url,
                        "success": True,
                        "duration": url_duration,
                        "content_length": len(result.cleaned_html),
                        "title": result.metadata.get("title", "N/A")
                    })
                    print(f"  ✅ Success in {url_duration:.2f}s ({len(result.cleaned_html)} chars)")
                else:
                    results.append({
                        "url": url,
                        "success": False,
                        "duration": url_duration,
                        "error": result.error_message
                    })
                    print(f"  ❌ Failed in {url_duration:.2f}s: {result.error_message}")
                    
            except Exception as e:
                results.append({
                    "url": url,
                    "success": False,
                    "duration": 0,
                    "error": str(e)
                })
                print(f"  💥 Exception: {str(e)}")
    
    total_time = time.time() - start_time
    successful_crawls = sum(1 for r in results if r["success"])
    
    print(f"\n📊 Batch Processing Summary:")
    print(f"  Total URLs: {len(urls)}")
    print(f"  Successful: {successful_crawls}")
    print(f"  Failed: {len(urls) - successful_crawls}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time per URL: {total_time/len(urls):.2f}s")
    
    # Save batch results
    batch_summary = {
        "timestamp": datetime.now().isoformat(),
        "total_urls": len(urls),
        "successful_crawls": successful_crawls,
        "failed_crawls": len(urls) - successful_crawls,
        "total_time": total_time,
        "average_time_per_url": total_time / len(urls),
        "results": results
    }
    
    with open("batch_processing_results.json", "w") as f:
        json.dump(batch_summary, f, indent=2)
    print("💾 Batch processing results saved")
    
    return results


async def performance_optimization_example():
    """Demonstrate performance optimization techniques"""
    print("\n=== Performance Optimization Example ===")
    
    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=True,
        verbose=False  # Reduce logging overhead
    )
    
    # Test different optimization strategies
    optimization_configs = [
        {
            "name": "Basic Configuration",
            "config": CrawlerRunConfig(
                word_count_threshold=10,
                screenshot=False,
                pdf=False,
                extract_media=True
            )
        },
        {
            "name": "Minimal Processing",
            "config": CrawlerRunConfig(
                word_count_threshold=5,
                screenshot=False,
                pdf=False,
                extract_media=False,
                remove_overlay_elements=False
            )
        },
        {
            "name": "Fast Cache Mode",
            "config": CrawlerRunConfig(
                cache_mode=CacheMode.READ_ONLY,
                word_count_threshold=5,
                screenshot=False,
                pdf=False,
                extract_media=False
            )
        },
        {
            "name": "Content Focused",
            "config": CrawlerRunConfig(
                css_selector="main, article, .content",
                word_count_threshold=10,
                screenshot=False,
                pdf=False,
                extract_media=False
            )
        }
    ]
    
    results = {}
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        for opt_config in optimization_configs:
            name = opt_config["name"]
            config = opt_config["config"]
            
            print(f"🔧 Testing {name}...")
            
            try:
                start_time = time.time()
                result = await crawler.arun(
                    url="https://github.com/CaptainASIC",
                    config=config
                )
                end_time = time.time()
                
                duration = end_time - start_time
                
                if result.success:
                    results[name] = {
                        "success": True,
                        "duration": duration,
                        "content_length": len(result.cleaned_html),
                        "markdown_length": len(result.markdown),
                        "has_media": bool(result.media),
                        "links_count": len(result.links.get("internal", [])) + len(result.links.get("external", [])) if result.links else 0
                    }
                    print(f"  ✅ {duration:.2f}s - {len(result.cleaned_html)} chars")
                else:
                    results[name] = {
                        "success": False,
                        "error": result.error_message
                    }
                    print(f"  ❌ Failed: {result.error_message}")
                    
            except Exception as e:
                results[name] = {
                    "success": False,
                    "error": str(e)
                }
                print(f"  💥 Exception: {str(e)}")
    
    # Find best performing configuration
    successful_results = {k: v for k, v in results.items() if v.get("success")}
    
    if successful_results:
        fastest = min(successful_results.items(), key=lambda x: x[1]["duration"])
        print(f"\n🏆 Best Performance: {fastest[0]} ({fastest[1]['duration']:.2f}s)")
        
        print("\n📊 Performance Comparison:")
        for name, data in results.items():
            if data.get("success"):
                print(f"  {name}: {data['duration']:.2f}s ({data['content_length']} chars)")
            else:
                print(f"  {name}: Failed - {data['error']}")
    
    # Save optimization results
    with open("performance_optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("💾 Performance optimization results saved")
    
    return results


async def memory_monitoring_example():
    """Example with basic memory usage monitoring"""
    print("\n=== Memory Monitoring Example ===")
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=True
    )
    
    run_config = CrawlerRunConfig(
        screenshot=True,
        pdf=True,
        extract_media=True,
        word_count_threshold=10
    )
    
    # Monitor memory before crawling
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    print(f"📊 Memory before crawling: {memory_before:.2f} MB")
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        try:
            start_time = time.time()
            result = await crawler.arun(
                url="https://github.com/CaptainASIC",
                config=run_config
            )
            end_time = time.time()
            
            # Monitor memory after crawling
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            if result.success:
                print(f"✅ Crawl completed in {end_time - start_time:.2f}s")
                print(f"📊 Memory after crawling: {memory_after:.2f} MB")
                print(f"📈 Memory used: {memory_used:.2f} MB")
                print(f"📄 Content size: {len(result.cleaned_html)} chars")
                print(f"📸 Screenshot: {'Yes' if result.screenshot else 'No'}")
                print(f"📄 PDF: {'Yes' if result.pdf else 'No'}")
                
                # Calculate efficiency metrics
                chars_per_mb = len(result.cleaned_html) / max(memory_used, 0.1)
                print(f"⚡ Efficiency: {chars_per_mb:.0f} chars/MB")
                
                memory_stats = {
                    "memory_before_mb": memory_before,
                    "memory_after_mb": memory_after,
                    "memory_used_mb": memory_used,
                    "content_length": len(result.cleaned_html),
                    "efficiency_chars_per_mb": chars_per_mb,
                    "duration_seconds": end_time - start_time
                }
                
                with open("memory_monitoring_results.json", "w") as f:
                    json.dump(memory_stats, f, indent=2)
                print("💾 Memory monitoring results saved")
                
                return memory_stats
            else:
                print(f"❌ Crawl failed: {result.error_message}")
                
        except Exception as e:
            print(f"💥 Exception: {str(e)}")
    
    return None


async def main():
    """Run all cache and performance examples"""
    print("🚀 Starting Cache Management and Performance Examples")
    print("🎯 Target: https://github.com/CaptainASIC")
    print("=" * 60)
    
    # Create output directory
    from pathlib import Path
    Path("performance_output").mkdir(exist_ok=True)
    import os
    os.chdir("performance_output")
    
    # Run performance examples
    await cache_modes_comparison()
    await batch_processing_example()
    await performance_optimization_example()
    await memory_monitoring_example()
    
    print("\n🎉 All cache and performance examples completed!")
    print("📁 Check the 'performance_output' directory for generated files")


if __name__ == "__main__":
    asyncio.run(main())


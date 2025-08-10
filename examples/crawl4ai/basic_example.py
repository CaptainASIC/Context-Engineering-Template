#!/usr/bin/env python3
"""
Basic Crawl4AI Integration Example

This example demonstrates the fundamental usage of Crawl4AI for web crawling.
It shows how to:
- Set up AsyncWebCrawler
- Perform basic web crawling
- Handle results and errors
- Extract markdown content

Target: https://github.com/CaptainASIC
"""

import asyncio
import json
from datetime import datetime
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig


async def basic_crawl_example():
    """
    Basic crawling example with minimal configuration
    """
    print("=== Basic Crawl Example ===")
    
    # Create a basic browser configuration
    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=True,
        verbose=True
    )
    
    # Use the crawler in an async context manager
    async with AsyncWebCrawler(config=browser_config) as crawler:
        try:
            # Perform the crawl
            result = await crawler.arun(url="https://github.com/CaptainASIC")
            
            if result.success:
                print(f"✅ Successfully crawled: {result.url}")
                print(f"📄 Page title: {result.metadata.get('title', 'N/A')}")
                print(f"📊 Content length: {len(result.cleaned_html)} characters")
                print(f"🔗 Links found: {len(result.links.get('internal', []))} internal, {len(result.links.get('external', []))} external")
                
                # Display first 500 characters of markdown
                print("\n📝 Content preview (first 500 chars):")
                print("-" * 50)
                print(result.markdown[:500])
                print("-" * 50)
                
                return result
            else:
                print(f"❌ Crawl failed: {result.error_message}")
                return None
                
        except Exception as e:
            print(f"💥 Exception occurred: {str(e)}")
            return None


async def crawl_with_custom_config():
    """
    Crawling with custom configuration options
    """
    print("\n=== Custom Configuration Example ===")
    
    # Custom browser configuration
    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=True,
        verbose=False,
        user_agent="Crawl4AI-Example/1.0"
    )
    
    # Custom crawler run configuration
    run_config = CrawlerRunConfig(
        word_count_threshold=10,
        screenshot=True,
        pdf=False,
        remove_overlay_elements=True,
        wait_for="css:body"
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        try:
            result = await crawler.arun(
                url="https://github.com/CaptainASIC",
                config=run_config
            )
            
            if result.success:
                print(f"✅ Custom crawl successful")
                print(f"📸 Screenshot captured: {'Yes' if result.screenshot else 'No'}")
                print(f"📄 PDF generated: {'Yes' if result.pdf else 'No'}")
                print(f"⏱️ Crawl time: {result.metadata.get('crawl_time', 'N/A')}")
                
                # Save screenshot if available
                if result.screenshot:
                    import base64
                    screenshot_data = base64.b64decode(result.screenshot)
                    with open("github_screenshot.png", "wb") as f:
                        f.write(screenshot_data)
                    print("📸 Screenshot saved as 'github_screenshot.png'")
                
                return result
            else:
                print(f"❌ Custom crawl failed: {result.error_message}")
                return None
                
        except Exception as e:
            print(f"💥 Exception in custom crawl: {str(e)}")
            return None


async def extract_specific_content():
    """
    Example of extracting specific content using CSS selectors
    """
    print("\n=== Content Extraction Example ===")
    
    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=True
    )
    
    # Configure to extract specific elements
    run_config = CrawlerRunConfig(
        css_selector="article, .repository-content, .readme",
        word_count_threshold=5,
        remove_overlay_elements=True
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        try:
            result = await crawler.arun(
                url="https://github.com/CaptainASIC",
                config=run_config
            )
            
            if result.success:
                print(f"✅ Content extraction successful")
                print(f"🎯 Extracted content length: {len(result.cleaned_html)} characters")
                
                # Display extracted content
                print("\n🎯 Extracted content preview:")
                print("-" * 50)
                print(result.markdown[:300])
                print("-" * 50)
                
                return result
            else:
                print(f"❌ Content extraction failed: {result.error_message}")
                return None
                
        except Exception as e:
            print(f"💥 Exception in content extraction: {str(e)}")
            return None


def save_results_to_file(result, filename="crawl_results.json"):
    """
    Save crawl results to a JSON file
    """
    if not result:
        return
    
    # Prepare data for JSON serialization
    data = {
        "timestamp": datetime.now().isoformat(),
        "url": result.url,
        "success": result.success,
        "title": result.metadata.get("title", ""),
        "content_length": len(result.cleaned_html),
        "markdown_length": len(result.markdown),
        "links_internal": len(result.links.get("internal", [])),
        "links_external": len(result.links.get("external", [])),
        "has_screenshot": bool(result.screenshot),
        "has_pdf": bool(result.pdf),
        "markdown_preview": result.markdown[:1000],  # First 1000 chars
        "metadata": result.metadata
    }
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved to {filename}")
    except Exception as e:
        print(f"❌ Failed to save results: {str(e)}")


async def main():
    """
    Main function to run all examples
    """
    print("🚀 Starting Crawl4AI Basic Examples")
    print("🎯 Target: https://github.com/CaptainASIC")
    print("=" * 60)
    
    # Run basic crawl example
    basic_result = await basic_crawl_example()
    
    # Run custom configuration example
    custom_result = await crawl_with_custom_config()
    
    # Run content extraction example
    extraction_result = await extract_specific_content()
    
    # Save results if any were successful
    if basic_result:
        save_results_to_file(basic_result, "basic_crawl_results.json")
    
    if custom_result:
        save_results_to_file(custom_result, "custom_crawl_results.json")
    
    if extraction_result:
        save_results_to_file(extraction_result, "extraction_results.json")
    
    print("\n🎉 All examples completed!")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())


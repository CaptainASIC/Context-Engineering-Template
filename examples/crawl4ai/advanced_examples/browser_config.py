#!/usr/bin/env python3
"""
Advanced Browser Configuration Examples for Crawl4AI

This module demonstrates advanced browser configuration options including:
- Custom user agents and headers
- Viewport settings
- Cookie management
- JavaScript execution
- Screenshot and PDF generation
"""

import asyncio
import base64
from datetime import datetime
from pathlib import Path

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode


async def basic_browser_config_example():
    """Example with basic browser configuration"""
    print("=== Basic Browser Configuration ===")
    
    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=True,
        verbose=True,
        user_agent="Crawl4AI-Example/1.0"
    )
    
    run_config = CrawlerRunConfig(
        screenshot=True,
        word_count_threshold=10
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        try:
            result = await crawler.arun(
                url="https://github.com/CaptainASIC",
                config=run_config
            )
            
            if result.success:
                print(f"✅ Basic config crawl successful")
                print(f"📊 Content length: {len(result.cleaned_html)} chars")
                print(f"📸 Screenshot: {'✓' if result.screenshot else '✗'}")
                
                # Save screenshot
                if result.screenshot:
                    await save_screenshot(result.screenshot, "basic_config_screenshot.png")
                
                return result
            else:
                print(f"❌ Basic config crawl failed: {result.error_message}")
                
        except Exception as e:
            print(f"💥 Exception: {str(e)}")
    
    return None


async def advanced_browser_config_example():
    """Example with advanced browser configuration options"""
    print("\n=== Advanced Browser Configuration ===")
    
    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=True,
        verbose=False,
        user_agent="Mozilla/5.0 (compatible; Crawl4AI-Advanced/1.0)",
        viewport_width=1920,
        viewport_height=1080,
        accept_downloads=True,
        java_script_enabled=True,
        cookies=[
            {"name": "session", "value": "example", "domain": "github.com"}
        ],
        headers={
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1"
        }
    )
    
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        screenshot=True,
        pdf=True,
        word_count_threshold=10,
        remove_overlay_elements=True,
        simulate_user=True,
        override_navigator=True,
        wait_for="css:.repository-content",
        delay_before_return_html=2.0,
        js_code=[
            "window.scrollTo(0, document.body.scrollHeight/2);",
            "await new Promise(resolve => setTimeout(resolve, 1000));"
        ]
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        try:
            result = await crawler.arun(
                url="https://github.com/CaptainASIC",
                config=run_config
            )
            
            if result.success:
                print(f"✅ Advanced config crawl successful")
                print(f"📊 Final URL: {result.url}")
                print(f"📄 Content length: {len(result.cleaned_html)} chars")
                print(f"📸 Screenshot: {'✓' if result.screenshot else '✗'}")
                print(f"📄 PDF: {'✓' if result.pdf else '✗'}")
                
                # Save media files
                if result.screenshot:
                    await save_screenshot(result.screenshot, "advanced_config_screenshot.png")
                
                if result.pdf:
                    await save_pdf(result.pdf, "advanced_config_page.pdf")
                
                return result
            else:
                print(f"❌ Advanced config crawl failed: {result.error_message}")
                
        except Exception as e:
            print(f"💥 Exception: {str(e)}")
    
    return None


async def mobile_browser_config_example():
    """Example simulating mobile browser"""
    print("\n=== Mobile Browser Configuration ===")
    
    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=True,
        user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15",
        viewport_width=375,
        viewport_height=812,
        device_scale_factor=3.0
    )
    
    run_config = CrawlerRunConfig(
        screenshot=True,
        word_count_threshold=5,
        wait_for="css:body"
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        try:
            result = await crawler.arun(
                url="https://github.com/CaptainASIC",
                config=run_config
            )
            
            if result.success:
                print(f"✅ Mobile config crawl successful")
                print(f"📱 Mobile viewport: 375x812")
                print(f"📊 Content length: {len(result.cleaned_html)} chars")
                
                if result.screenshot:
                    await save_screenshot(result.screenshot, "mobile_config_screenshot.png")
                
                return result
            else:
                print(f"❌ Mobile config crawl failed: {result.error_message}")
                
        except Exception as e:
            print(f"💥 Exception: {str(e)}")
    
    return None


async def custom_javascript_example():
    """Example with custom JavaScript execution"""
    print("\n=== Custom JavaScript Execution ===")
    
    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=True,
        java_script_enabled=True
    )
    
    # Custom JavaScript to extract specific data
    custom_js = [
        """
        // Scroll to load more content
        window.scrollTo(0, document.body.scrollHeight);
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Add custom data attribute
        document.body.setAttribute('data-crawled-by', 'Crawl4AI');
        
        // Log some information
        console.log('Custom JS executed successfully');
        """,
        """
        // Highlight all links
        const links = document.querySelectorAll('a');
        links.forEach(link => {
            link.style.border = '2px solid red';
        });
        """
    ]
    
    run_config = CrawlerRunConfig(
        js_code=custom_js,
        screenshot=True,
        delay_before_return_html=3.0,
        word_count_threshold=10
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        try:
            result = await crawler.arun(
                url="https://github.com/CaptainASIC",
                config=run_config
            )
            
            if result.success:
                print(f"✅ Custom JS crawl successful")
                print(f"📊 Content length: {len(result.cleaned_html)} chars")
                
                # Check if our custom attribute was added
                if 'data-crawled-by="Crawl4AI"' in result.cleaned_html:
                    print("🎯 Custom JavaScript executed successfully!")
                
                if result.screenshot:
                    await save_screenshot(result.screenshot, "custom_js_screenshot.png")
                
                return result
            else:
                print(f"❌ Custom JS crawl failed: {result.error_message}")
                
        except Exception as e:
            print(f"💥 Exception: {str(e)}")
    
    return None


async def save_screenshot(screenshot_b64: str, filename: str):
    """Save base64 screenshot to file"""
    try:
        screenshot_data = base64.b64decode(screenshot_b64)
        with open(filename, "wb") as f:
            f.write(screenshot_data)
        print(f"📸 Screenshot saved as '{filename}'")
    except Exception as e:
        print(f"❌ Failed to save screenshot: {str(e)}")


async def save_pdf(pdf_b64: str, filename: str):
    """Save base64 PDF to file"""
    try:
        pdf_data = base64.b64decode(pdf_b64)
        with open(filename, "wb") as f:
            f.write(pdf_data)
        print(f"📄 PDF saved as '{filename}'")
    except Exception as e:
        print(f"❌ Failed to save PDF: {str(e)}")


async def main():
    """Run all browser configuration examples"""
    print("🚀 Starting Advanced Browser Configuration Examples")
    print("🎯 Target: https://github.com/CaptainASIC")
    print("=" * 60)
    
    # Create output directory
    Path("browser_config_output").mkdir(exist_ok=True)
    import os
    os.chdir("browser_config_output")
    
    # Run examples
    await basic_browser_config_example()
    await advanced_browser_config_example()
    await mobile_browser_config_example()
    await custom_javascript_example()
    
    print("\n🎉 All browser configuration examples completed!")
    print("📁 Check the 'browser_config_output' directory for generated files")


if __name__ == "__main__":
    asyncio.run(main())


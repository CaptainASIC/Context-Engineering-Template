#!/usr/bin/env python3
"""
Content Extraction Strategies Examples for Crawl4AI

This module demonstrates different content extraction strategies including:
- CSS selector-based extraction
- XPath-based extraction
- LLM-powered extraction
- Custom extraction patterns
"""

import asyncio
import json
import os
from datetime import datetime

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy, LLMExtractionStrategy


async def css_extraction_example():
    """Example using CSS-based content extraction"""
    print("=== CSS Extraction Strategy ===")
    
    # Define extraction schema for GitHub profile
    extraction_schema = {
        "name": "GitHubProfile",
        "baseSelector": "body",
        "fields": [
            {
                "name": "username",
                "selector": ".p-nickname",
                "type": "text"
            },
            {
                "name": "full_name", 
                "selector": ".p-name",
                "type": "text"
            },
            {
                "name": "bio",
                "selector": ".p-note",
                "type": "text"
            },
            {
                "name": "repositories",
                "selector": ".js-pinned-item-list-item",
                "type": "nested",
                "fields": [
                    {
                        "name": "name",
                        "selector": ".repo",
                        "type": "text"
                    },
                    {
                        "name": "description",
                        "selector": ".pinned-item-desc",
                        "type": "text"
                    },
                    {
                        "name": "language",
                        "selector": "[itemprop='programmingLanguage']",
                        "type": "text"
                    }
                ]
            }
        ]
    }
    
    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=True
    )
    
    run_config = CrawlerRunConfig(
        extraction_strategy=JsonCssExtractionStrategy(extraction_schema),
        wait_for="css:.p-nickname",
        delay_before_return_html=3.0
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        try:
            result = await crawler.arun(
                url="https://github.com/CaptainASIC",
                config=run_config
            )
            
            if result.success and result.extracted_content:
                print("✅ CSS extraction successful")
                
                # Parse extracted content
                extracted_data = json.loads(result.extracted_content)
                print(f"🎯 Extracted data:")
                print(json.dumps(extracted_data, indent=2))
                
                # Save extracted data
                with open("css_extracted_data.json", "w") as f:
                    json.dump(extracted_data, f, indent=2)
                print("💾 CSS extracted data saved to 'css_extracted_data.json'")
                
                return result
            else:
                print(f"❌ CSS extraction failed: {result.error_message}")
                return None
                
        except Exception as e:
            print(f"💥 Exception in CSS extraction: {str(e)}")
            return None


async def simple_css_extraction_example():
    """Simple CSS extraction for basic content"""
    print("\n=== Simple CSS Extraction ===")
    
    # Simple extraction schema
    simple_schema = {
        "name": "BasicInfo",
        "baseSelector": "body",
        "fields": [
            {
                "name": "title",
                "selector": "title",
                "type": "text"
            },
            {
                "name": "headings",
                "selector": "h1, h2, h3",
                "type": "text",
                "multiple": True
            },
            {
                "name": "links",
                "selector": "a[href]",
                "type": "attribute",
                "attribute": "href",
                "multiple": True
            }
        ]
    }
    
    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=True
    )
    
    run_config = CrawlerRunConfig(
        extraction_strategy=JsonCssExtractionStrategy(simple_schema),
        delay_before_return_html=2.0
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        try:
            result = await crawler.arun(
                url="https://github.com/CaptainASIC",
                config=run_config
            )
            
            if result.success and result.extracted_content:
                print("✅ Simple CSS extraction successful")
                
                extracted_data = json.loads(result.extracted_content)
                print(f"📄 Page title: {extracted_data.get('title', 'N/A')}")
                print(f"📝 Headings found: {len(extracted_data.get('headings', []))}")
                print(f"🔗 Links found: {len(extracted_data.get('links', []))}")
                
                # Save data
                with open("simple_extracted_data.json", "w") as f:
                    json.dump(extracted_data, f, indent=2)
                print("💾 Simple extracted data saved")
                
                return result
            else:
                print(f"❌ Simple CSS extraction failed: {result.error_message}")
                
        except Exception as e:
            print(f"💥 Exception: {str(e)}")
    
    return None


async def llm_extraction_example():
    """Example using LLM-based content extraction"""
    print("\n=== LLM Extraction Strategy ===")
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipping LLM extraction - OPENAI_API_KEY not found")
        print("   Set OPENAI_API_KEY environment variable to enable this feature")
        return None
    
    # Define extraction prompt
    extraction_prompt = """
    Extract the following information from this GitHub profile page:
    1. Username
    2. Full name (if available)
    3. Bio/description
    4. Number of public repositories
    5. Programming languages used (from pinned repositories)
    6. Company/organization (if mentioned)
    7. Location (if available)
    
    Return the information in JSON format with clear field names.
    """
    
    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=True
    )
    
    run_config = CrawlerRunConfig(
        extraction_strategy=LLMExtractionStrategy(
            provider="openai",
            api_token=os.getenv("OPENAI_API_KEY"),
            instruction=extraction_prompt
        ),
        wait_for="css:.p-nickname",
        delay_before_return_html=2.0
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        try:
            result = await crawler.arun(
                url="https://github.com/CaptainASIC",
                config=run_config
            )
            
            if result.success and result.extracted_content:
                print("✅ LLM extraction successful")
                
                # Parse and display extracted content
                try:
                    extracted_data = json.loads(result.extracted_content)
                    print("🤖 LLM extracted data:")
                    print(json.dumps(extracted_data, indent=2))
                    
                    # Save LLM extracted data
                    with open("llm_extracted_data.json", "w") as f:
                        json.dump(extracted_data, f, indent=2)
                    print("💾 LLM data saved to 'llm_extracted_data.json'")
                    
                except json.JSONDecodeError:
                    print("📝 LLM extracted content (raw):")
                    print(result.extracted_content)
                    
                    # Save raw content
                    with open("llm_extracted_raw.txt", "w") as f:
                        f.write(result.extracted_content)
                    print("💾 Raw LLM data saved")
                
                return result
            else:
                print(f"❌ LLM extraction failed: {result.error_message}")
                return None
                
        except Exception as e:
            print(f"💥 Exception in LLM extraction: {str(e)}")
            return None


async def repository_extraction_example():
    """Extract repository information from GitHub profile"""
    print("\n=== Repository Information Extraction ===")
    
    # Schema focused on repository data
    repo_schema = {
        "name": "GitHubRepositories",
        "baseSelector": "body",
        "fields": [
            {
                "name": "profile_info",
                "selector": ".js-profile-editable-area",
                "type": "nested",
                "fields": [
                    {
                        "name": "repositories_count",
                        "selector": "a[href*='repositories'] .Counter",
                        "type": "text"
                    },
                    {
                        "name": "followers_count",
                        "selector": "a[href*='followers'] .text-bold",
                        "type": "text"
                    },
                    {
                        "name": "following_count",
                        "selector": "a[href*='following'] .text-bold",
                        "type": "text"
                    }
                ]
            },
            {
                "name": "pinned_repositories",
                "selector": ".js-pinned-item-list-item",
                "type": "nested",
                "multiple": True,
                "fields": [
                    {
                        "name": "name",
                        "selector": ".repo",
                        "type": "text"
                    },
                    {
                        "name": "description",
                        "selector": ".pinned-item-desc",
                        "type": "text"
                    },
                    {
                        "name": "language",
                        "selector": "[itemprop='programmingLanguage']",
                        "type": "text"
                    },
                    {
                        "name": "stars",
                        "selector": ".octicon-star + span",
                        "type": "text"
                    },
                    {
                        "name": "forks",
                        "selector": ".octicon-repo-forked + span",
                        "type": "text"
                    }
                ]
            }
        ]
    }
    
    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=True
    )
    
    run_config = CrawlerRunConfig(
        extraction_strategy=JsonCssExtractionStrategy(repo_schema),
        wait_for="css:.js-pinned-item-list-item",
        delay_before_return_html=3.0
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        try:
            result = await crawler.arun(
                url="https://github.com/CaptainASIC",
                config=run_config
            )
            
            if result.success and result.extracted_content:
                print("✅ Repository extraction successful")
                
                extracted_data = json.loads(result.extracted_content)
                
                # Display summary
                profile_info = extracted_data.get("profile_info", {})
                pinned_repos = extracted_data.get("pinned_repositories", [])
                
                print(f"📊 Profile Statistics:")
                print(f"  - Repositories: {profile_info.get('repositories_count', 'N/A')}")
                print(f"  - Followers: {profile_info.get('followers_count', 'N/A')}")
                print(f"  - Following: {profile_info.get('following_count', 'N/A')}")
                
                print(f"\n📌 Pinned Repositories ({len(pinned_repos)}):")
                for repo in pinned_repos[:3]:  # Show first 3
                    name = repo.get('name', 'N/A')
                    lang = repo.get('language', 'N/A')
                    desc = repo.get('description', 'No description')[:50]
                    print(f"  - {name} ({lang}): {desc}...")
                
                # Save repository data
                with open("repository_data.json", "w") as f:
                    json.dump(extracted_data, f, indent=2)
                print("💾 Repository data saved")
                
                return result
            else:
                print(f"❌ Repository extraction failed: {result.error_message}")
                
        except Exception as e:
            print(f"💥 Exception: {str(e)}")
    
    return None


async def main():
    """Run all extraction strategy examples"""
    print("🚀 Starting Content Extraction Strategy Examples")
    print("🎯 Target: https://github.com/CaptainASIC")
    print("=" * 60)
    
    # Create output directory
    from pathlib import Path
    Path("extraction_output").mkdir(exist_ok=True)
    import os
    os.chdir("extraction_output")
    
    # Run extraction examples
    await css_extraction_example()
    await simple_css_extraction_example()
    await llm_extraction_example()
    await repository_extraction_example()
    
    print("\n🎉 All extraction strategy examples completed!")
    print("📁 Check the 'extraction_output' directory for generated files")


if __name__ == "__main__":
    asyncio.run(main())


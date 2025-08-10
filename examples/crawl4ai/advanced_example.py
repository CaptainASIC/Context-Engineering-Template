#!/usr/bin/env python3
"""
Advanced Crawl4AI Integration Example - Main Runner

This example demonstrates advanced features of Crawl4AI using a modular approach.
The implementation is split across multiple files to maintain clean, manageable code.

Modules:
- browser_config.py: Advanced browser configuration examples
- extraction_strategies.py: Content extraction strategies
- cache_and_performance.py: Cache management and performance optimization

Usage:
    python advanced_example.py [module_name]
    
    Available modules:
    - browser_config: Run browser configuration examples
    - extraction: Run extraction strategy examples  
    - performance: Run cache and performance examples
    - all: Run all examples (default)

Target: https://github.com/CaptainASIC
"""

import asyncio
import sys
from pathlib import Path

# Import the modular examples
from advanced_examples import browser_config, extraction_strategies, cache_and_performance


async def run_browser_config_examples():
    """Run browser configuration examples"""
    print("🔧 Running Browser Configuration Examples")
    print("=" * 50)
    await browser_config.main()


async def run_extraction_examples():
    """Run extraction strategy examples"""
    print("\n🎯 Running Extraction Strategy Examples")
    print("=" * 50)
    await extraction_strategies.main()


async def run_performance_examples():
    """Run cache and performance examples"""
    print("\n⚡ Running Cache and Performance Examples")
    print("=" * 50)
    await cache_and_performance.main()


async def run_all_examples():
    """Run all advanced examples"""
    print("🚀 Starting All Advanced Crawl4AI Examples")
    print("🎯 Target: https://github.com/CaptainASIC")
    print("📁 Modular structure:")
    print("  - browser_config.py: Browser configuration options")
    print("  - extraction_strategies.py: Content extraction methods")
    print("  - cache_and_performance.py: Performance optimization")
    print("=" * 60)
    
    # Create main output directory
    Path("advanced_examples_output").mkdir(exist_ok=True)
    import os
    original_dir = os.getcwd()
    os.chdir("advanced_examples_output")
    
    try:
        # Run all example modules
        await run_browser_config_examples()
        await run_extraction_examples()
        await run_performance_examples()
        
        print("\n🎉 All advanced examples completed successfully!")
        print("📁 Check the 'advanced_examples_output' directory for generated files")
        
    finally:
        # Return to original directory
        os.chdir(original_dir)


def print_usage():
    """Print usage information"""
    print("Advanced Crawl4AI Examples")
    print("=" * 30)
    print("Usage: python advanced_example.py [module]")
    print()
    print("Available modules:")
    print("  browser_config  - Browser configuration examples")
    print("  extraction      - Content extraction strategies")
    print("  performance     - Cache and performance optimization")
    print("  all            - Run all examples (default)")
    print()
    print("Examples:")
    print("  python advanced_example.py")
    print("  python advanced_example.py browser_config")
    print("  python advanced_example.py extraction")
    print("  python advanced_example.py performance")


async def main():
    """Main function to handle command line arguments and run examples"""
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        module = sys.argv[1].lower()
    else:
        module = "all"
    
    # Handle help requests
    if module in ["-h", "--help", "help"]:
        print_usage()
        return
    
    # Run the requested module
    try:
        if module == "browser_config":
            await run_browser_config_examples()
        elif module == "extraction":
            await run_extraction_examples()
        elif module == "performance":
            await run_performance_examples()
        elif module == "all":
            await run_all_examples()
        else:
            print(f"❌ Unknown module: {module}")
            print()
            print_usage()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n💥 Error running examples: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())


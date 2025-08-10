# Crawl4AI Integration Examples

This directory contains comprehensive examples demonstrating how to integrate Crawl4AI into your applications with various architectures and use cases.

## 📁 Project Structure

```
examples/crawl4ai/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
├── basic_example.py            # Simple crawling example
├── batch_example.py            # Batch processing example
├── advanced_example.py         # Advanced features runner
├── fastapi_example.py          # FastAPI integration
├── advanced_examples/          # Modular advanced examples
│   ├── browser_config.py       # Browser configuration
│   ├── extraction_strategies.py # Content extraction
│   └── cache_and_performance.py # Performance optimization
├── fastapi_integration/        # Modular FastAPI components
│   ├── models.py              # Pydantic models
│   ├── crawler_service.py     # Crawler management
│   ├── routes.py              # API endpoints
│   └── background_tasks.py    # Async task handling
└── react-crawl4ai-demo/       # React frontend
    ├── src/
    │   ├── components/
    │   │   ├── CrawlForm.jsx   # Crawling form
    │   │   └── CrawlResults.jsx # Results display
    │   └── App.jsx             # Main application
    └── package.json
```

## 🚀 Examples Overview

### 1. Basic Example (`basic_example.py`)
Simple introduction to Crawl4AI with essential features:
- Basic web crawling setup
- Content extraction and display
- Screenshot and PDF generation
- Error handling and logging

### 2. Batch Processing (`batch_example.py`)
Efficient processing of multiple URLs:
- Concurrent crawling with rate limiting
- Resource monitoring and management
- Progress tracking and reporting
- Performance comparison strategies

### 3. Advanced Examples (`advanced_example.py`)
Comprehensive feature demonstration split into modules:
- **Browser Configuration**: Custom user agents, viewports, JavaScript execution
- **Extraction Strategies**: CSS selectors, XPath, LLM-powered extraction
- **Cache & Performance**: Cache modes, optimization techniques, memory monitoring

### 4. FastAPI Integration (`fastapi_example.py`)
Production-ready API service with modular architecture:
- RESTful API endpoints with validation
- Async request handling and background tasks
- Comprehensive error handling and logging
- Interactive API documentation

### 5. React Frontend (`react-crawl4ai-demo/`)
Modern web interface for crawling operations:
- Interactive form with all crawling options
- Real-time results display and analysis
- Content viewing, copying, and downloading
- Responsive design with Tailwind CSS

## 🛠️ Quick Start

### Prerequisites
```bash
# Install Python dependencies
pip install -r requirements.txt

# For React frontend (optional)
cd react-crawl4ai-demo
npm install  # or pnpm install
```

### Running Examples

#### 1. Basic Crawling
```bash
python basic_example.py
```

#### 2. Batch Processing
```bash
python batch_example.py
```

#### 3. Advanced Features
```bash
# Run all advanced examples
python advanced_example.py

# Run specific modules
python advanced_example.py browser_config
python advanced_example.py extraction
python advanced_example.py performance
```

#### 4. FastAPI Backend
```bash
# Start the API server
python fastapi_example.py

# Visit API documentation
open http://localhost:8000/docs
```

#### 5. React Frontend
```bash
# Start development server
cd react-crawl4ai-demo
npm run dev

# Visit the application
open http://localhost:5173
```

## 🎯 Target URLs

All examples use GitHub-related URLs for demonstration:
- Primary: `https://github.com/CaptainASIC`
- Additional: Profile tabs, repositories, etc.

## 📊 Features Demonstrated

### Core Crawling Features
- ✅ Basic web page crawling
- ✅ Content extraction (HTML, Markdown)
- ✅ Screenshot capture
- ✅ PDF generation
- ✅ Media extraction
- ✅ Link analysis

### Advanced Configuration
- ✅ Custom browser settings
- ✅ User agent customization
- ✅ Viewport configuration
- ✅ JavaScript execution
- ✅ Cookie management
- ✅ Header customization

### Extraction Strategies
- ✅ CSS selector-based extraction
- ✅ XPath-based extraction
- ✅ LLM-powered extraction (with OpenAI)
- ✅ Custom extraction schemas
- ✅ Structured data extraction

### Performance & Optimization
- ✅ Cache management (multiple modes)
- ✅ Batch processing with concurrency control
- ✅ Rate limiting and retry logic
- ✅ Memory usage monitoring
- ✅ Performance benchmarking

### Integration Patterns
- ✅ FastAPI REST API
- ✅ React frontend integration
- ✅ Background task processing
- ✅ Real-time progress tracking
- ✅ Error handling and recovery

## 🔧 Configuration Options

### Browser Configuration
```python
browser_config = BrowserConfig(
    browser_type="chromium",
    headless=True,
    user_agent="Custom-Agent/1.0",
    viewport_width=1920,
    viewport_height=1080,
    java_script_enabled=True
)
```

### Crawling Configuration
```python
run_config = CrawlerRunConfig(
    css_selector="main, article",
    word_count_threshold=10,
    screenshot=True,
    pdf=True,
    extract_media=True,
    cache_mode=CacheMode.ENABLED
)
```

## 🌐 API Endpoints (FastAPI)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| POST | `/crawl` | Single URL crawling |
| POST | `/batch-crawl` | Multiple URL crawling |
| POST | `/extract` | Structured content extraction |
| POST | `/crawl-async` | Asynchronous crawling |
| GET | `/status/{task_id}` | Task status |
| GET | `/tasks` | List all tasks |

## 📱 Frontend Features

- **Interactive Form**: Configure all crawling parameters
- **Real-time Results**: View extracted content immediately
- **Content Analysis**: Statistics, links, and media breakdown
- **Export Options**: Copy or download results
- **Responsive Design**: Works on desktop and mobile
- **Error Handling**: Graceful error display and recovery

## 🔍 Example Use Cases

1. **Content Migration**: Extract content from old websites
2. **SEO Analysis**: Analyze page structure and links
3. **Data Collection**: Gather structured data from web pages
4. **Monitoring**: Track changes in web content
5. **Research**: Collect information from multiple sources
6. **Testing**: Validate web page rendering and content

## 🚨 Important Notes

- **API Keys**: Set `OPENAI_API_KEY` environment variable for LLM extraction
- **Rate Limiting**: Be respectful of target websites
- **Error Handling**: All examples include comprehensive error handling
- **Performance**: Monitor resource usage for large-scale operations
- **Security**: Validate all inputs in production environments

## 📚 Additional Resources

- [Crawl4AI Documentation](https://docs.crawl4ai.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [GitHub Repository](https://github.com/CaptainASIC)

## 🤝 Contributing

Feel free to extend these examples with additional features or improvements. Each module is designed to be self-contained and easily modifiable.


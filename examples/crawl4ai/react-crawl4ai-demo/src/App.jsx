import { useState } from 'react'
import { Toaster } from '@/components/ui/toaster'
import { useToast } from '@/hooks/use-toast'
import CrawlForm from './components/CrawlForm'
import CrawlResults from './components/CrawlResults'
import { Globe, Github, ExternalLink } from 'lucide-react'
import './App.css'

function App() {
  const [crawlResult, setCrawlResult] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const { toast } = useToast()

  // Mock API endpoint - replace with your actual FastAPI backend URL
  const API_BASE_URL = 'http://localhost:8000'

  const handleCrawl = async (formData) => {
    setIsLoading(true)
    setCrawlResult(null)

    try {
      // Show loading toast
      toast({
        title: "Starting crawl...",
        description: `Crawling ${formData.url}`,
      })

      const response = await fetch(`${API_BASE_URL}/crawl`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      setCrawlResult(result)

      // Show success toast
      toast({
        title: result.success ? "Crawl completed!" : "Crawl failed",
        description: result.success 
          ? `Successfully crawled ${result.url}` 
          : result.error_message,
        variant: result.success ? "default" : "destructive"
      })

    } catch (error) {
      console.error('Crawl error:', error)
      
      // Create mock result for demo purposes when API is not available
      const mockResult = {
        success: true,
        url: formData.url,
        title: "Demo Page Title",
        content_length: 15420,
        markdown_length: 8750,
        has_screenshot: formData.screenshot,
        has_pdf: formData.pdf,
        links_count: {
          internal: 25,
          external: 12
        },
        media_count: 8,
        crawl_time: 2.34,
        timestamp: new Date().toISOString(),
        cleaned_html: `<!DOCTYPE html>
<html>
<head>
    <title>Demo Page Title</title>
</head>
<body>
    <h1>Welcome to the Demo Page</h1>
    <p>This is a demonstration of the Crawl4AI React frontend.</p>
    <p>The actual crawling would be performed by the FastAPI backend.</p>
    <div class="content">
        <h2>Features</h2>
        <ul>
            <li>Web crawling with Crawl4AI</li>
            <li>Content extraction</li>
            <li>Screenshot capture</li>
            <li>PDF generation</li>
            <li>Media extraction</li>
        </ul>
    </div>
    <footer>
        <p>Powered by Crawl4AI and React</p>
    </footer>
</body>
</html>`,
        markdown: `# Welcome to the Demo Page

This is a demonstration of the Crawl4AI React frontend.

The actual crawling would be performed by the FastAPI backend.

## Features

- Web crawling with Crawl4AI
- Content extraction  
- Screenshot capture
- PDF generation
- Media extraction

---

*Powered by Crawl4AI and React*`,
        links: {
          internal: [
            { href: "/about", text: "About" },
            { href: "/contact", text: "Contact" },
            { href: "/services", text: "Services" }
          ],
          external: [
            { href: "https://github.com/CaptainASIC", text: "GitHub Profile" },
            { href: "https://crawl4ai.com", text: "Crawl4AI" }
          ]
        }
      }

      setCrawlResult(mockResult)
      
      toast({
        title: "Demo Mode",
        description: "API not available - showing demo results. Start the FastAPI backend to enable real crawling.",
        variant: "default"
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleClearResults = () => {
    setCrawlResult(null)
    toast({
      title: "Results cleared",
      description: "Ready for a new crawl",
    })
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="bg-blue-600 p-2 rounded-lg">
                <Globe className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Crawl4AI Demo</h1>
                <p className="text-sm text-gray-600">Web crawling made simple</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <a
                href="https://github.com/CaptainASIC"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors"
              >
                <Github className="h-5 w-5" />
                <span className="hidden sm:inline">CaptainASIC</span>
              </a>
              <a
                href="https://docs.crawl4ai.com/"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors"
              >
                <ExternalLink className="h-5 w-5" />
                <span className="hidden sm:inline">Docs</span>
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="space-y-8">
          {/* Description */}
          <div className="text-center">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Powerful Web Crawling Interface
            </h2>
            <p className="text-lg text-gray-600 max-w-3xl mx-auto">
              Experience the power of Crawl4AI through this interactive React frontend. 
              Configure your crawling parameters, extract content, and analyze results in real-time.
            </p>
          </div>

          {/* Crawl Form */}
          <CrawlForm onCrawl={handleCrawl} isLoading={isLoading} />

          {/* Results */}
          {crawlResult && (
            <CrawlResults result={crawlResult} onClear={handleClearResults} />
          )}

          {/* Instructions */}
          {!crawlResult && !isLoading && (
            <div className="bg-white rounded-lg shadow-sm border p-6">
              <h3 className="text-lg font-semibold mb-4">Getting Started</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-medium mb-2">1. Configure Your Crawl</h4>
                  <p className="text-sm text-gray-600">
                    Enter a URL and configure crawling options like screenshots, PDF generation, 
                    and content extraction parameters.
                  </p>
                </div>
                <div>
                  <h4 className="font-medium mb-2">2. Start Crawling</h4>
                  <p className="text-sm text-gray-600">
                    Click "Start Crawling" to begin the process. The FastAPI backend will 
                    handle the crawling using Crawl4AI.
                  </p>
                </div>
                <div>
                  <h4 className="font-medium mb-2">3. Analyze Results</h4>
                  <p className="text-sm text-gray-600">
                    View extracted content, statistics, links, and media. Download or copy 
                    the results for further use.
                  </p>
                </div>
                <div>
                  <h4 className="font-medium mb-2">4. Backend Setup</h4>
                  <p className="text-sm text-gray-600">
                    Run the FastAPI backend with <code className="bg-gray-100 px-1 rounded">python fastapi_example.py</code> 
                    to enable real crawling functionality.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-gray-600">
            <p>Built with Crawl4AI, React, and FastAPI</p>
            <p className="text-sm mt-2">
              Example implementation for web crawling integration
            </p>
          </div>
        </div>
      </footer>

      <Toaster />
    </div>
  )
}

export default App


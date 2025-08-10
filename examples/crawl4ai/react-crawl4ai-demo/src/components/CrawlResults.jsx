import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { 
  CheckCircle, 
  XCircle, 
  Clock, 
  FileText, 
  Camera, 
  Link, 
  Image, 
  Download,
  Copy,
  Eye,
  BarChart3
} from 'lucide-react'

const CrawlResults = ({ result, onClear }) => {
  const [copiedField, setCopiedField] = useState(null)

  if (!result) {
    return null
  }

  const copyToClipboard = async (text, field) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopiedField(field)
      setTimeout(() => setCopiedField(null), 2000)
    } catch (err) {
      console.error('Failed to copy text: ', err)
    }
  }

  const downloadContent = (content, filename, type = 'text/plain') => {
    const blob = new Blob([content], { type })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <div className="w-full max-w-6xl mx-auto space-y-6">
      {/* Status Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {result.success ? (
                <CheckCircle className="h-5 w-5 text-green-500" />
              ) : (
                <XCircle className="h-5 w-5 text-red-500" />
              )}
              <CardTitle>
                {result.success ? 'Crawl Successful' : 'Crawl Failed'}
              </CardTitle>
            </div>
            <Button variant="outline" onClick={onClear}>
              Clear Results
            </Button>
          </div>
          <CardDescription>
            {result.url} • {new Date(result.timestamp).toLocaleString()}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold">{result.crawl_time?.toFixed(2)}s</div>
              <div className="text-sm text-muted-foreground flex items-center justify-center gap-1">
                <Clock className="h-3 w-3" />
                Duration
              </div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold">{result.content_length?.toLocaleString()}</div>
              <div className="text-sm text-muted-foreground flex items-center justify-center gap-1">
                <FileText className="h-3 w-3" />
                Characters
              </div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold">
                {(result.links_count?.internal || 0) + (result.links_count?.external || 0)}
              </div>
              <div className="text-sm text-muted-foreground flex items-center justify-center gap-1">
                <Link className="h-3 w-3" />
                Links
              </div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold">{result.media_count || 0}</div>
              <div className="text-sm text-muted-foreground flex items-center justify-center gap-1">
                <Image className="h-3 w-3" />
                Media
              </div>
            </div>
          </div>

          {/* Features */}
          <div className="flex flex-wrap gap-2 mt-4">
            {result.has_screenshot && (
              <Badge variant="secondary">
                <Camera className="h-3 w-3 mr-1" />
                Screenshot
              </Badge>
            )}
            {result.has_pdf && (
              <Badge variant="secondary">
                <FileText className="h-3 w-3 mr-1" />
                PDF
              </Badge>
            )}
            {result.title && (
              <Badge variant="outline">
                {result.title}
              </Badge>
            )}
          </div>

          {/* Error Message */}
          {!result.success && result.error_message && (
            <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-700 text-sm">{result.error_message}</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Content Tabs */}
      {result.success && (
        <Card>
          <CardContent className="p-0">
            <Tabs defaultValue="content" className="w-full">
              <div className="border-b px-6 pt-6">
                <TabsList className="grid w-full grid-cols-4">
                  <TabsTrigger value="content">Content</TabsTrigger>
                  <TabsTrigger value="markdown">Markdown</TabsTrigger>
                  <TabsTrigger value="links">Links</TabsTrigger>
                  <TabsTrigger value="stats">Statistics</TabsTrigger>
                </TabsList>
              </div>

              {/* HTML Content */}
              <TabsContent value="content" className="p-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold">Cleaned HTML Content</h3>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => copyToClipboard(result.cleaned_html || '', 'html')}
                      >
                        {copiedField === 'html' ? (
                          <CheckCircle className="h-4 w-4 mr-1" />
                        ) : (
                          <Copy className="h-4 w-4 mr-1" />
                        )}
                        Copy
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => downloadContent(result.cleaned_html || '', 'content.html', 'text/html')}
                      >
                        <Download className="h-4 w-4 mr-1" />
                        Download
                      </Button>
                    </div>
                  </div>
                  <ScrollArea className="h-96 w-full border rounded-lg p-4">
                    <pre className="text-sm whitespace-pre-wrap">
                      {result.cleaned_html || 'No content available'}
                    </pre>
                  </ScrollArea>
                </div>
              </TabsContent>

              {/* Markdown Content */}
              <TabsContent value="markdown" className="p-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold">Markdown Content</h3>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => copyToClipboard(result.markdown || '', 'markdown')}
                      >
                        {copiedField === 'markdown' ? (
                          <CheckCircle className="h-4 w-4 mr-1" />
                        ) : (
                          <Copy className="h-4 w-4 mr-1" />
                        )}
                        Copy
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => downloadContent(result.markdown || '', 'content.md', 'text/markdown')}
                      >
                        <Download className="h-4 w-4 mr-1" />
                        Download
                      </Button>
                    </div>
                  </div>
                  <ScrollArea className="h-96 w-full border rounded-lg p-4">
                    <pre className="text-sm whitespace-pre-wrap">
                      {result.markdown || 'No markdown content available'}
                    </pre>
                  </ScrollArea>
                </div>
              </TabsContent>

              {/* Links */}
              <TabsContent value="links" className="p-6">
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Extracted Links</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-base">Internal Links</CardTitle>
                        <CardDescription>
                          {result.links_count?.internal || 0} links found
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <ScrollArea className="h-48">
                          <div className="space-y-2">
                            {result.links?.internal?.slice(0, 10).map((link, index) => (
                              <div key={index} className="text-sm p-2 bg-muted rounded">
                                <a 
                                  href={link.href} 
                                  target="_blank" 
                                  rel="noopener noreferrer"
                                  className="text-blue-600 hover:underline"
                                >
                                  {link.text || link.href}
                                </a>
                              </div>
                            )) || <p className="text-muted-foreground">No internal links found</p>}
                          </div>
                        </ScrollArea>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader>
                        <CardTitle className="text-base">External Links</CardTitle>
                        <CardDescription>
                          {result.links_count?.external || 0} links found
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <ScrollArea className="h-48">
                          <div className="space-y-2">
                            {result.links?.external?.slice(0, 10).map((link, index) => (
                              <div key={index} className="text-sm p-2 bg-muted rounded">
                                <a 
                                  href={link.href} 
                                  target="_blank" 
                                  rel="noopener noreferrer"
                                  className="text-blue-600 hover:underline"
                                >
                                  {link.text || link.href}
                                </a>
                              </div>
                            )) || <p className="text-muted-foreground">No external links found</p>}
                          </div>
                        </ScrollArea>
                      </CardContent>
                    </Card>
                  </div>
                </div>
              </TabsContent>

              {/* Statistics */}
              <TabsContent value="stats" className="p-6">
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold flex items-center gap-2">
                    <BarChart3 className="h-5 w-5" />
                    Crawl Statistics
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <Card>
                      <CardContent className="p-4">
                        <div className="text-2xl font-bold">{result.content_length?.toLocaleString()}</div>
                        <div className="text-sm text-muted-foreground">HTML Characters</div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="p-4">
                        <div className="text-2xl font-bold">{result.markdown_length?.toLocaleString()}</div>
                        <div className="text-sm text-muted-foreground">Markdown Characters</div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="p-4">
                        <div className="text-2xl font-bold">{result.crawl_time?.toFixed(2)}s</div>
                        <div className="text-sm text-muted-foreground">Processing Time</div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="p-4">
                        <div className="text-2xl font-bold">{result.links_count?.internal || 0}</div>
                        <div className="text-sm text-muted-foreground">Internal Links</div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="p-4">
                        <div className="text-2xl font-bold">{result.links_count?.external || 0}</div>
                        <div className="text-sm text-muted-foreground">External Links</div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="p-4">
                        <div className="text-2xl font-bold">{result.media_count || 0}</div>
                        <div className="text-sm text-muted-foreground">Media Items</div>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Extracted Content */}
                  {result.extracted_content && (
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-base">Extracted Structured Data</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <ScrollArea className="h-48">
                          <pre className="text-sm whitespace-pre-wrap">
                            {JSON.stringify(result.extracted_content, null, 2)}
                          </pre>
                        </ScrollArea>
                      </CardContent>
                    </Card>
                  )}
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

export default CrawlResults


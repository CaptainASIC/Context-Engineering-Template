import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Switch } from '@/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Loader2, Globe, Camera, FileText, Image, Link } from 'lucide-react'

const CrawlForm = ({ onCrawl, isLoading }) => {
  const [formData, setFormData] = useState({
    url: 'https://github.com/CaptainASIC',
    css_selector: '',
    word_count_threshold: 10,
    screenshot: false,
    pdf: false,
    extract_media: true,
    wait_for: '',
    delay: 0,
    cache_mode: 'enabled'
  })

  const handleSubmit = (e) => {
    e.preventDefault()
    onCrawl(formData)
  }

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }))
  }

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Globe className="h-5 w-5" />
          Web Crawling Configuration
        </CardTitle>
        <CardDescription>
          Configure your web crawling parameters and start extracting content
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* URL Input */}
          <div className="space-y-2">
            <Label htmlFor="url">Target URL</Label>
            <Input
              id="url"
              type="url"
              value={formData.url}
              onChange={(e) => handleInputChange('url', e.target.value)}
              placeholder="https://example.com"
              required
            />
          </div>

          {/* CSS Selector */}
          <div className="space-y-2">
            <Label htmlFor="css_selector">CSS Selector (Optional)</Label>
            <Input
              id="css_selector"
              value={formData.css_selector}
              onChange={(e) => handleInputChange('css_selector', e.target.value)}
              placeholder="main, article, .content"
            />
            <p className="text-sm text-muted-foreground">
              Specify CSS selector to extract specific content
            </p>
          </div>

          {/* Word Count Threshold */}
          <div className="space-y-2">
            <Label htmlFor="word_count_threshold">Word Count Threshold</Label>
            <Input
              id="word_count_threshold"
              type="number"
              min="1"
              max="1000"
              value={formData.word_count_threshold}
              onChange={(e) => handleInputChange('word_count_threshold', parseInt(e.target.value))}
            />
          </div>

          {/* Options Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Screenshot Option */}
            <div className="flex items-center justify-between space-x-2 p-3 border rounded-lg">
              <div className="flex items-center space-x-2">
                <Camera className="h-4 w-4" />
                <Label htmlFor="screenshot">Capture Screenshot</Label>
              </div>
              <Switch
                id="screenshot"
                checked={formData.screenshot}
                onCheckedChange={(checked) => handleInputChange('screenshot', checked)}
              />
            </div>

            {/* PDF Option */}
            <div className="flex items-center justify-between space-x-2 p-3 border rounded-lg">
              <div className="flex items-center space-x-2">
                <FileText className="h-4 w-4" />
                <Label htmlFor="pdf">Generate PDF</Label>
              </div>
              <Switch
                id="pdf"
                checked={formData.pdf}
                onCheckedChange={(checked) => handleInputChange('pdf', checked)}
              />
            </div>

            {/* Extract Media Option */}
            <div className="flex items-center justify-between space-x-2 p-3 border rounded-lg">
              <div className="flex items-center space-x-2">
                <Image className="h-4 w-4" />
                <Label htmlFor="extract_media">Extract Media</Label>
              </div>
              <Switch
                id="extract_media"
                checked={formData.extract_media}
                onCheckedChange={(checked) => handleInputChange('extract_media', checked)}
              />
            </div>

            {/* Cache Mode */}
            <div className="space-y-2">
              <Label>Cache Mode</Label>
              <Select
                value={formData.cache_mode}
                onValueChange={(value) => handleInputChange('cache_mode', value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="enabled">Enabled</SelectItem>
                  <SelectItem value="bypass">Bypass</SelectItem>
                  <SelectItem value="read_only">Read Only</SelectItem>
                  <SelectItem value="write_only">Write Only</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Advanced Options */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium">Advanced Options</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Wait For */}
              <div className="space-y-2">
                <Label htmlFor="wait_for">Wait For (CSS Selector)</Label>
                <Input
                  id="wait_for"
                  value={formData.wait_for}
                  onChange={(e) => handleInputChange('wait_for', e.target.value)}
                  placeholder="css:.content"
                />
              </div>

              {/* Delay */}
              <div className="space-y-2">
                <Label htmlFor="delay">Delay (seconds)</Label>
                <Input
                  id="delay"
                  type="number"
                  min="0"
                  max="10"
                  step="0.1"
                  value={formData.delay}
                  onChange={(e) => handleInputChange('delay', parseFloat(e.target.value) || 0)}
                />
              </div>
            </div>
          </div>

          {/* Submit Button */}
          <Button type="submit" className="w-full" disabled={isLoading}>
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Crawling...
              </>
            ) : (
              <>
                <Globe className="mr-2 h-4 w-4" />
                Start Crawling
              </>
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}

export default CrawlForm


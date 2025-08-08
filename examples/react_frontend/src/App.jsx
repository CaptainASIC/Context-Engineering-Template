import { useState } from 'react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { Input } from '@/components/ui/input.jsx'
import { Textarea } from '@/components/ui/textarea.jsx'
import { 
  Brain, 
  Database, 
  Network, 
  MessageSquare, 
  Search, 
  Zap,
  Code,
  GitBranch,
  Settings,
  Users,
  BarChart3,
  FileText,
  Bot,
  Layers
} from 'lucide-react'
import './App.css'

function App() {
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I\'m your context-aware AI assistant. How can I help you today?' }
  ])
  const [isLoading, setIsLoading] = useState(false)

  const handleSendMessage = async () => {
    if (!query.trim()) return
    
    setIsLoading(true)
    const newMessage = { role: 'user', content: query }
    setMessages(prev => [...prev, newMessage])
    setQuery('')
    
    // Simulate AI response
    setTimeout(() => {
      const response = {
        role: 'assistant',
        content: `I understand you're asking about "${query}". Based on my knowledge graph and vector search capabilities, I can provide contextual information while maintaining conversation memory.`
      }
      setMessages(prev => [...prev, response])
      setIsLoading(false)
    }, 1500)
  }

  const technologies = [
    {
      name: 'PydanticAI',
      description: 'Multi-LLM RAG agent with type-safe responses',
      icon: <Bot className="h-6 w-6" />,
      features: ['OpenAI', 'Anthropic', 'Gemini', 'Ollama'],
      color: 'bg-blue-500'
    },
    {
      name: 'Neo4j',
      description: 'Knowledge graph for relationship modeling',
      icon: <Network className="h-6 w-6" />,
      features: ['Graph Queries', 'Relationships', 'Vector Search'],
      color: 'bg-green-500'
    },
    {
      name: 'Neon + pgvector',
      description: 'PostgreSQL with vector similarity search',
      icon: <Database className="h-6 w-6" />,
      features: ['Vector Storage', 'Semantic Search', 'ACID Compliance'],
      color: 'bg-purple-500'
    },
    {
      name: 'mem0',
      description: 'Intelligent memory management system',
      icon: <Brain className="h-6 w-6" />,
      features: ['Context Memory', 'User Preferences', 'Smart Retrieval'],
      color: 'bg-orange-500'
    }
  ]

  const metrics = [
    { label: 'Active Conversations', value: '1,247', change: '+12%' },
    { label: 'Knowledge Entities', value: '45,892', change: '+8%' },
    { label: 'Vector Embeddings', value: '2.3M', change: '+15%' },
    { label: 'Memory Contexts', value: '8,934', change: '+22%' }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm dark:bg-slate-900/80 sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="h-10 w-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                <Layers className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Context Engineering Platform
                </h1>
                <p className="text-sm text-muted-foreground">Advanced AI Context Management</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Button variant="outline" size="sm">
                <Settings className="h-4 w-4 mr-2" />
                Settings
              </Button>
              <Button size="sm">
                <Users className="h-4 w-4 mr-2" />
                Team
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        {/* Metrics Dashboard */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {metrics.map((metric, index) => (
            <Card key={index} className="hover:shadow-lg transition-shadow">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">{metric.label}</p>
                    <p className="text-2xl font-bold">{metric.value}</p>
                  </div>
                  <div className="flex items-center space-x-1">
                    <BarChart3 className="h-4 w-4 text-green-500" />
                    <span className="text-sm text-green-500 font-medium">{metric.change}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        <Tabs defaultValue="chat" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="chat" className="flex items-center space-x-2">
              <MessageSquare className="h-4 w-4" />
              <span>AI Chat</span>
            </TabsTrigger>
            <TabsTrigger value="technologies" className="flex items-center space-x-2">
              <Code className="h-4 w-4" />
              <span>Technologies</span>
            </TabsTrigger>
            <TabsTrigger value="knowledge" className="flex items-center space-x-2">
              <Network className="h-4 w-4" />
              <span>Knowledge Graph</span>
            </TabsTrigger>
            <TabsTrigger value="analytics" className="flex items-center space-x-2">
              <BarChart3 className="h-4 w-4" />
              <span>Analytics</span>
            </TabsTrigger>
          </TabsList>

          {/* AI Chat Interface */}
          <TabsContent value="chat" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <Card className="h-[600px] flex flex-col">
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Bot className="h-5 w-5" />
                      <span>Context-Aware AI Assistant</span>
                    </CardTitle>
                    <CardDescription>
                      Powered by PydanticAI with multi-LLM support and persistent memory
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="flex-1 flex flex-col">
                    <div className="flex-1 overflow-y-auto space-y-4 mb-4">
                      {messages.map((message, index) => (
                        <div
                          key={index}
                          className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                          <div
                            className={`max-w-[80%] rounded-lg px-4 py-2 ${
                              message.role === 'user'
                                ? 'bg-blue-500 text-white'
                                : 'bg-muted text-foreground'
                            }`}
                          >
                            {message.content}
                          </div>
                        </div>
                      ))}
                      {isLoading && (
                        <div className="flex justify-start">
                          <div className="bg-muted rounded-lg px-4 py-2">
                            <div className="flex space-x-1">
                              <div className="w-2 h-2 bg-current rounded-full animate-bounce"></div>
                              <div className="w-2 h-2 bg-current rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                              <div className="w-2 h-2 bg-current rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                    <div className="flex space-x-2">
                      <Input
                        placeholder="Ask me anything about context engineering..."
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                        disabled={isLoading}
                      />
                      <Button onClick={handleSendMessage} disabled={isLoading || !query.trim()}>
                        <Search className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </div>
              
              <div className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Memory Context</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Active Memories</span>
                      <Badge variant="secondary">24</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">User Preferences</span>
                      <Badge variant="secondary">8</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Context Depth</span>
                      <Badge variant="secondary">High</Badge>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Recent Queries</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <div className="text-sm p-2 bg-muted rounded">
                      "How does vector search work?"
                    </div>
                    <div className="text-sm p-2 bg-muted rounded">
                      "Explain knowledge graphs"
                    </div>
                    <div className="text-sm p-2 bg-muted rounded">
                      "Best practices for RAG"
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* Technologies Overview */}
          <TabsContent value="technologies" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {technologies.map((tech, index) => (
                <Card key={index} className="hover:shadow-lg transition-all duration-300 hover:scale-105">
                  <CardHeader>
                    <div className="flex items-center space-x-3">
                      <div className={`p-2 rounded-lg ${tech.color} text-white`}>
                        {tech.icon}
                      </div>
                      <div>
                        <CardTitle className="text-xl">{tech.name}</CardTitle>
                        <CardDescription>{tech.description}</CardDescription>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="flex flex-wrap gap-2">
                      {tech.features.map((feature, idx) => (
                        <Badge key={idx} variant="outline">
                          {feature}
                        </Badge>
                      ))}
                    </div>
                    <Button className="w-full mt-4" variant="outline">
                      <Code className="h-4 w-4 mr-2" />
                      View Implementation
                    </Button>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* Knowledge Graph Visualization */}
          <TabsContent value="knowledge" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Network className="h-5 w-5" />
                  <span>Knowledge Graph Explorer</span>
                </CardTitle>
                <CardDescription>
                  Visualize relationships between entities, policies, and use cases
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-96 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-950 dark:to-purple-950 rounded-lg flex items-center justify-center border-2 border-dashed border-muted-foreground/25">
                  <div className="text-center space-y-4">
                    <Network className="h-16 w-16 mx-auto text-muted-foreground" />
                    <div>
                      <h3 className="text-lg font-semibold">Interactive Graph Visualization</h3>
                      <p className="text-muted-foreground">
                        Connect to Neo4j to explore your knowledge graph
                      </p>
                    </div>
                    <Button>
                      <GitBranch className="h-4 w-4 mr-2" />
                      Connect to Neo4j
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Analytics Dashboard */}
          <TabsContent value="analytics" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Vector Search Performance</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64 bg-gradient-to-br from-green-50 to-blue-50 dark:from-green-950 dark:to-blue-950 rounded-lg flex items-center justify-center">
                    <div className="text-center">
                      <BarChart3 className="h-12 w-12 mx-auto text-muted-foreground mb-2" />
                      <p className="text-muted-foreground">Performance metrics visualization</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Memory Usage Patterns</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64 bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-950 dark:to-red-950 rounded-lg flex items-center justify-center">
                    <div className="text-center">
                      <Brain className="h-12 w-12 mx-auto text-muted-foreground mb-2" />
                      <p className="text-muted-foreground">Memory analytics dashboard</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>

      {/* Footer */}
      <footer className="border-t bg-white/80 backdrop-blur-sm dark:bg-slate-900/80 mt-16">
        <div className="container mx-auto px-4 py-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center space-x-2 mb-4 md:mb-0">
              <Layers className="h-5 w-5 text-muted-foreground" />
              <span className="text-muted-foreground">Context Engineering Platform</span>
            </div>
            <div className="flex items-center space-x-4 text-sm text-muted-foreground">
              <span>Built with React, FastAPI, PydanticAI, Neo4j, Neon & mem0</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App


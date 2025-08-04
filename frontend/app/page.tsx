"use client"

import { useState, useEffect, useRef } from "react"
import type { SetStateAction, Key } from "react"
import {
  Search,
  Clock,
  Database,
  Zap,
  TrendingUp,
  Globe,
  CheckCircle,
  AlertCircle,
  Sparkles,
  Activity,
  Loader2,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Separator } from "@/components/ui/separator"

// Define a clear type for our search results
type SearchResult = {
  cached: boolean
  query: string
  summary: string
  sources?: Array<{ url: string; title: string }>
  timestamp: string
  selector_used?: string
  similar_query?: string
}

// Define the type for our statistics object
type QueryStats = {
  totalQueries: number
  cachedHits: number
  avgResponseTime: number
  totalResponseTime: number // This is needed to correctly calculate the average
}

export default function QueryAgentApp() {
  const [query, setQuery] = useState("")
  const [results, setResults] = useState<SearchResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [stats, setStats] = useState<QueryStats>({
    totalQueries: 0,
    cachedHits: 0,
    avgResponseTime: 0,
    totalResponseTime: 0,
  })
  const [responseTime, setResponseTime] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)

  const API_BASE_URL = "http://localhost:8000/api"

  // Load stats from localStorage on component mount
  useEffect(() => {
    try {
      const savedStats = localStorage.getItem("queryStats")
      if (savedStats) {
        setStats(JSON.parse(savedStats))
      }
    } catch (e) {
      console.error("Failed to parse stats from localStorage", e)
    }
  }, [])

  const updateStats = (currentResponseTime: number, cached: boolean) => {
    const newTotalQueries = stats.totalQueries + 1
    const newTotalResponseTime = stats.totalResponseTime + currentResponseTime
    
    const newStats: QueryStats = {
      totalQueries: newTotalQueries,
      cachedHits: stats.cachedHits + (cached ? 1 : 0),
      totalResponseTime: newTotalResponseTime,
      avgResponseTime: newTotalResponseTime / newTotalQueries,
    }
    
    setStats(newStats)
    localStorage.setItem("queryStats", JSON.stringify(newStats))
  }

  const handleSearch = async (e?: React.FormEvent<HTMLFormElement>) => {
    if (e) e.preventDefault()
    if (!query.trim()) return

    setIsLoading(true)
    setError(null)
    setResults(null)
    const startTime = Date.now()

    try {
      // Validate query first
      const validationResponse = await fetch(`${API_BASE_URL}/validate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query.trim() }),
      })
      const validation = await validationResponse.json()
      if (!validationResponse.ok || !validation.is_valid) {
        throw new Error(`Invalid query: ${validation.reason || "Validation failed"}`)
      }

      // Perform search
      const searchResponse = await fetch(`${API_BASE_URL}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query.trim(), use_cache: true }),
      })
      if (!searchResponse.ok) {
        throw new Error(`Search failed: ${searchResponse.statusText}`)
      }

      const searchData: SearchResult = await searchResponse.json()
      const endTime = Date.now()
      const totalResponseTime = (endTime - startTime) / 1000

      setResults(searchData)
      setResponseTime(totalResponseTime)
      updateStats(totalResponseTime, searchData.cached)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "An unknown error occurred."
      setError(errorMessage)
      console.error("Search error:", err)
    } finally {
      setIsLoading(false)
    }
  }

  const handleInputFocus = () => {
    if (results || error) {
      setResults(null)
      setError(null)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleSearch()
    }
  }

  const setExampleQuery = (exampleQuery: string) => {
    setQuery(exampleQuery)
    setResults(null)
    setError(null)
    inputRef.current?.focus()
  }

  const exampleQueries = [
    "Best restaurants in Mumbai",
    "Latest AI technology trends 2024",
    "How to learn Python programming",
    "Top tourist attractions in Delhi",
    "Best practices for web development",
    "Climate change effects on environment",
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      {/* Animated background */}
      <div className="absolute inset-0 bg-[url('/grid.svg')] bg-center [mask-image:linear-gradient(180deg,white,rgba(255,255,255,0))]" />
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-0 -left-4 w-72 h-72 bg-purple-300 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob" />
        <div className="absolute top-0 -right-4 w-72 h-72 bg-yellow-300 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-2000" />
        <div className="absolute -bottom-8 left-20 w-72 h-72 bg-pink-300 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-4000" />
      </div>

      <div className="relative z-10 container mx-auto px-4 py-8 lg:py-16">
        {/* Header */}
        <div className="text-center mb-16">
          <div className="flex items-center justify-center mb-6">
            <div className="p-4 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl shadow-2xl animate-pulse">
              <Search className="w-8 h-8 text-white" />
            </div>
          </div>
          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold bg-gradient-to-r from-white via-purple-200 to-purple-400 bg-clip-text text-transparent mb-4 leading-normal">
            Web Browser Query Agent
          </h1>
          <p className="text-lg md:text-xl text-purple-200 max-w-2xl mx-auto leading-normal">
            Intelligent web search with similarity matching and caching
          </p>
        </div>

        {/* Search Section */}
        <Card className="mb-8 bg-white/5 backdrop-blur-sm border-white/20 shadow-2xl">
          <CardContent className="p-6 md:p-8">
            <form onSubmit={handleSearch} className="flex flex-col sm:flex-row gap-4">
              <div className="flex-1">
                <Input
                  ref={inputRef}
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onFocus={handleInputFocus}
                  onKeyDown={handleKeyDown}
                  placeholder="Enter your search query... (e.g., 'Best places to visit in Delhi')"
                  disabled={isLoading}
                  className="h-14 text-lg px-6 bg-slate-800 text-white border-2 border-slate-700 focus:border-purple-500 focus:bg-slate-900 transition-all duration-300"
                />
              </div>
              <Button
                type="submit"
                disabled={isLoading || !query.trim()}
                size="lg"
                className="h-14 px-8 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 transition-all duration-300 shadow-lg hover:shadow-xl disabled:opacity-50"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                    Searching...
                  </>
                ) : (
                  <>
                    <Search className="w-5 h-5 mr-2" />
                    Search
                  </>
                )}
              </Button>
            </form>
          </CardContent>
        </Card>
        
        {isLoading && (
          <div className="text-center py-12 flex flex-col items-center">
             <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full mb-4">
               <Loader2 className="w-8 h-8 text-white animate-spin" />
             </div>
             <p className="text-lg text-purple-200 font-medium">Processing your query... This may take 5-15 seconds</p>
          </div>
        )}

        {/* Results Section */}
        {(results || error) && !isLoading && (
          <Card className="mb-8 bg-white/95 backdrop-blur-sm border-0 shadow-2xl animate-fade-in">
            <CardContent className="p-6 md:p-8">
              {error ? (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription className="font-medium">{error}</AlertDescription>
                </Alert>
              ) : results ? (
                <>
                  {/* Result Header */}
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-8 gap-4">
                    <div className="flex items-center gap-3">
                      <Badge
                        variant={results.cached ? "secondary" : "default"}
                        className={`px-3 py-1 text-xs font-bold ${
                          results.cached
                            ? "bg-green-100 text-green-800 hover:bg-green-200"
                            : "bg-blue-100 text-blue-800 hover:bg-blue-200"
                        }`}
                      >
                        {results.cached ? (
                          <>
                            <Database className="w-3 h-3 mr-1" />
                            CACHED
                          </>
                        ) : (
                          <>
                            <Sparkles className="w-3 h-3 mr-1" />
                            NEW SEARCH
                          </>
                        )}
                      </Badge>
                      <span className="text-lg font-semibold text-gray-800">{results.query}</span>
                    </div>
                    <div className="flex items-center text-sm text-gray-500">
                      <Clock className="w-4 h-4 mr-1" />
                      {responseTime.toFixed(2)}s
                    </div>
                  </div>

                  {/* Similar Query Notice */}
                  {results.cached && results.similar_query && (
                    <Alert className="mb-6 border-green-200 bg-green-50">
                      <Activity className="h-4 w-4 text-green-600" />
                      <AlertDescription className="text-green-800">
                        <span className="font-medium">Similar to:</span> "{results.similar_query}"
                      </AlertDescription>
                    </Alert>
                  )}

                  {/* Summary */}
                  <div className="prose prose-lg max-w-none mb-8">
                    <div
                      className="text-gray-700 leading-relaxed"
                      dangerouslySetInnerHTML={{ __html: results.summary }}
                    />
                  </div>

                  {/* Sources */}
                  {results.sources && results.sources.length > 0 && (
                    <>
                      <Separator className="my-6" />
                      <div>
                        <h4 className="text-xl font-semibold mb-4 flex items-center text-gray-900">
                          <Globe className="w-5 h-5 mr-2 text-purple-600" />
                          Sources
                        </h4>
                        <div className="grid gap-3">
                          {results.sources.map((source, index) => (
                            <a
                              key={index}
                              href={source.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="flex items-center p-3 rounded-lg border border-gray-200 hover:border-purple-300 hover:bg-purple-50 transition-all duration-200 group"
                            >
                              <CheckCircle className="w-4 h-4 mr-3 text-green-500 flex-shrink-0" />
                              <span className="text-purple-700 group-hover:text-purple-800 font-medium">
                                {source.title}
                              </span>
                            </a>
                          ))}
                        </div>
                      </div>
                    </>
                  )}

                  {/* Metadata */}
                  <Separator className="my-6" />
                  <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 text-sm text-gray-500">
                    <span>Processed at: {new Date(results.timestamp).toLocaleString()}</span>
                    {results.selector_used && <span>Selector: {results.selector_used}</span>}
                  </div>
                </>
              ) : null}
            </CardContent>
          </Card>
        )}

        {/* Example Queries */}
        <Card className="mb-8 bg-white/10 backdrop-blur-sm border-white/20 shadow-2xl">
          <CardHeader>
            <CardTitle className="text-2xl font-bold text-white flex items-center">
              <Zap className="w-6 h-6 mr-3 text-yellow-400 leading-normal" />
              Example Queries
            </CardTitle>
            <CardDescription className="text-purple-200 leading-normal">Click on any example to try it out</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {exampleQueries.map((example, index) => (
                <Button
                  key={index}
                  variant="ghost"
                  onClick={() => setExampleQuery(example)}
                  className="h-auto p-4 text-left justify-start bg-white/10 hover:bg-white/20 text-white border border-white/20 hover:border-white/30 transition-all duration-300 hover:scale-105"
                >
                  {example}
                </Button>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Stats Section */}
        <Card className="bg-white/10 backdrop-blur-sm border-white/20 shadow-2xl">
          <CardHeader>
            <CardTitle className="text-2xl font-bold text-white flex items-center">
              <TrendingUp className="w-6 h-6 mr-3 text-green-400 leading-normal" />
              Query Statistics
            </CardTitle>
            <CardDescription className="text-purple-200 leading-normal">Your search activity overview</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center p-6 bg-white/10 rounded-xl border border-white/20 hover:bg-white/20 transition-all duration-300 hover:scale-105">
                <div className="text-4xl font-bold text-white mb-2">{stats.totalQueries}</div>
                <div className="text-purple-200 text-sm font-medium uppercase tracking-wide leading-normal">Total Queries</div>
              </div>
              <div className="text-center p-6 bg-white/10 rounded-xl border border-white/20 hover:bg-white/20 transition-all duration-300 hover:scale-105">
                <div className="text-4xl font-bold text-white mb-2">{stats.cachedHits}</div>
                <div className="text-purple-200 text-sm font-medium uppercase tracking-wide leading-normal">Cache Hits</div>
              </div>
              <div className="text-center p-6 bg-white/10 rounded-xl border border-white/20 hover:bg-white/20 transition-all duration-300 hover:scale-105">
                <div className="text-4xl font-bold text-white mb-2 leading-normal">
                  {stats.avgResponseTime.toFixed(1)}s
                </div>
                <div className="text-purple-200 text-sm font-medium uppercase tracking-wide">Avg Response Time</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
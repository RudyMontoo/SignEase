/**
 * Request Batcher
 * ===============
 * 
 * Advanced request batching and optimization using TanStack Pacer patterns
 * Implements debouncing, throttling, rate limiting, and intelligent batching
 */

import { throttle, debounce } from './performanceOptimizer'

export interface BatchRequest {
  id: string
  endpoint: string
  method: 'GET' | 'POST' | 'PUT' | 'DELETE'
  data?: any
  headers?: Record<string, string>
  priority: 'low' | 'medium' | 'high' | 'critical'
  timestamp: number
  retryCount: number
  maxRetries: number
  timeout: number
}

export interface BatchResponse {
  id: string
  success: boolean
  data?: any
  error?: string
  statusCode?: number
  duration: number
}

export interface BatchMetrics {
  totalRequests: number
  batchedRequests: number
  successRate: number
  averageLatency: number
  cacheHitRate: number
  rateLimitHits: number
  queueSize: number
  activeRequests: number
}

export interface RateLimitConfig {
  limit: number
  window: number // milliseconds
  windowType: 'fixed' | 'sliding'
}

export interface BatchConfig {
  maxBatchSize: number
  batchTimeout: number // milliseconds
  rateLimits: Record<string, RateLimitConfig>
  retryDelays: number[] // exponential backoff delays
  cacheTimeout: number // milliseconds
  priorityWeights: Record<string, number>
}

class RequestBatcher {
  private config: BatchConfig
  private requestQueue: BatchRequest[] = []
  private activeRequests: Map<string, Promise<BatchResponse>> = new Map()
  private responseCache: Map<string, { response: BatchResponse; timestamp: number }> = new Map()
  private rateLimiters: Map<string, { count: number; windowStart: number; requests: number[] }> = new Map()
  private metrics: BatchMetrics = {
    totalRequests: 0,
    batchedRequests: 0,
    successRate: 0,
    averageLatency: 0,
    cacheHitRate: 0,
    rateLimitHits: 0,
    queueSize: 0,
    activeRequests: 0
  }
  private isProcessing: boolean = false
  private processingTimer: number | null = null

  constructor(config: Partial<BatchConfig> = {}) {
    this.config = {
      maxBatchSize: 10,
      batchTimeout: 100,
      rateLimits: {
        '/api/predict': { limit: 30, window: 1000, windowType: 'sliding' },
        '/api/health': { limit: 10, window: 1000, windowType: 'fixed' },
        default: { limit: 50, window: 1000, windowType: 'sliding' }
      },
      retryDelays: [1000, 2000, 4000, 8000], // exponential backoff
      cacheTimeout: 60000, // 1 minute
      priorityWeights: { critical: 4, high: 3, medium: 2, low: 1 },
      ...config
    }

    // Start periodic processing
    this.startPeriodicProcessing()
    
    // Cleanup expired cache entries
    setInterval(() => this.cleanupCache(), 30000)
  }

  // Rate-limited request execution with intelligent batching
  async executeRequest(
    endpoint: string,
    options: {
      method?: 'GET' | 'POST' | 'PUT' | 'DELETE'
      data?: any
      headers?: Record<string, string>
      priority?: 'low' | 'medium' | 'high' | 'critical'
      timeout?: number
      maxRetries?: number
      cacheKey?: string
    } = {}
  ): Promise<BatchResponse> {
    const requestId = this.generateRequestId()
    const cacheKey = options.cacheKey || this.generateCacheKey(endpoint, options.method || 'GET', options.data)
    
    // Check cache first
    const cachedResponse = this.getCachedResponse(cacheKey)
    if (cachedResponse) {
      this.metrics.cacheHitRate = (this.metrics.cacheHitRate * this.metrics.totalRequests + 1) / (this.metrics.totalRequests + 1)
      this.metrics.totalRequests++
      return cachedResponse
    }

    // Check rate limits
    if (!this.checkRateLimit(endpoint)) {
      this.metrics.rateLimitHits++
      throw new Error(`Rate limit exceeded for ${endpoint}`)
    }

    const request: BatchRequest = {
      id: requestId,
      endpoint,
      method: options.method || 'GET',
      data: options.data,
      headers: options.headers,
      priority: options.priority || 'medium',
      timestamp: Date.now(),
      retryCount: 0,
      maxRetries: options.maxRetries || 3,
      timeout: options.timeout || 5000
    }

    // Check if request is already in progress
    const activeRequest = this.activeRequests.get(cacheKey)
    if (activeRequest) {
      return activeRequest
    }

    // Add to queue and process
    this.requestQueue.push(request)
    this.metrics.queueSize = this.requestQueue.length
    this.metrics.totalRequests++

    // Create promise for this request
    const requestPromise = new Promise<BatchResponse>((resolve, reject) => {
      request.resolve = resolve
      request.reject = reject
    }) as Promise<BatchResponse> & { resolve?: (value: BatchResponse) => void; reject?: (reason: any) => void }

    this.activeRequests.set(cacheKey, requestPromise)

    // Trigger immediate processing for high priority requests
    if (request.priority === 'critical' || request.priority === 'high') {
      this.processQueue()
    }

    return requestPromise
  }

  // Debounced request execution for search/input scenarios
  executeDebounced = debounce(async (
    endpoint: string,
    options: Parameters<typeof this.executeRequest>[1] = {}
  ) => {
    return this.executeRequest(endpoint, options)
  }, 300)

  // Throttled request execution for frequent updates
  executeThrottled = throttle(async (
    endpoint: string,
    options: Parameters<typeof this.executeRequest>[1] = {}
  ) => {
    return this.executeRequest(endpoint, options)
  }, 100)

  private startPeriodicProcessing(): void {
    const processInterval = Math.min(this.config.batchTimeout, 50) // Process at least every 50ms
    
    setInterval(() => {
      if (this.requestQueue.length > 0 && !this.isProcessing) {
        this.processQueue()
      }
    }, processInterval)
  }

  private async processQueue(): Promise<void> {
    if (this.isProcessing || this.requestQueue.length === 0) return

    this.isProcessing = true

    try {
      // Sort queue by priority and timestamp
      this.requestQueue.sort((a, b) => {
        const priorityDiff = this.config.priorityWeights[b.priority] - this.config.priorityWeights[a.priority]
        return priorityDiff !== 0 ? priorityDiff : a.timestamp - b.timestamp
      })

      // Create batches based on endpoint and method
      const batches = this.createBatches()

      // Process batches concurrently
      const batchPromises = batches.map(batch => this.processBatch(batch))
      await Promise.allSettled(batchPromises)

    } finally {
      this.isProcessing = false
      this.metrics.queueSize = this.requestQueue.length
    }
  }

  private createBatches(): BatchRequest[][] {
    const batches: BatchRequest[][] = []
    const batchMap = new Map<string, BatchRequest[]>()

    // Group requests by endpoint and method for batching
    for (const request of this.requestQueue) {
      const batchKey = `${request.method}:${request.endpoint}`
      
      if (!batchMap.has(batchKey)) {
        batchMap.set(batchKey, [])
      }
      
      const batch = batchMap.get(batchKey)!
      batch.push(request)

      // Create new batch if max size reached
      if (batch.length >= this.config.maxBatchSize) {
        batches.push([...batch])
        batchMap.set(batchKey, [])
      }
    }

    // Add remaining partial batches
    for (const batch of batchMap.values()) {
      if (batch.length > 0) {
        batches.push(batch)
      }
    }

    return batches
  }

  private async processBatch(batch: BatchRequest[]): Promise<void> {
    if (batch.length === 0) return

    const startTime = performance.now()

    try {
      // Remove processed requests from queue
      batch.forEach(request => {
        const index = this.requestQueue.indexOf(request)
        if (index > -1) {
          this.requestQueue.splice(index, 1)
        }
      })

      // Execute batch
      if (batch.length === 1) {
        // Single request
        await this.executeSingleRequest(batch[0])
      } else {
        // Batch request
        await this.executeBatchRequest(batch)
      }

      this.metrics.batchedRequests += batch.length
      
    } catch (error) {
      console.error('Batch processing error:', error)
      
      // Handle failed requests
      batch.forEach(request => {
        if (request.reject) {
          request.reject(error)
        }
      })
    }

    const endTime = performance.now()
    const duration = endTime - startTime
    
    // Update metrics
    this.updateLatencyMetrics(duration)
  }

  private async executeSingleRequest(request: BatchRequest): Promise<void> {
    const startTime = performance.now()

    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), request.timeout)

      const response = await fetch(request.endpoint, {
        method: request.method,
        headers: {
          'Content-Type': 'application/json',
          ...request.headers
        },
        body: request.data ? JSON.stringify(request.data) : undefined,
        signal: controller.signal
      })

      clearTimeout(timeoutId)

      const responseData = await response.json()
      const endTime = performance.now()
      const duration = endTime - startTime

      const batchResponse: BatchResponse = {
        id: request.id,
        success: response.ok,
        data: responseData,
        statusCode: response.status,
        duration
      }

      if (!response.ok) {
        batchResponse.error = `HTTP ${response.status}: ${response.statusText}`
      }

      // Cache successful responses
      if (response.ok && request.method === 'GET') {
        const cacheKey = this.generateCacheKey(request.endpoint, request.method, request.data)
        this.cacheResponse(cacheKey, batchResponse)
      }

      // Update success rate
      this.updateSuccessRate(response.ok)

      // Resolve request
      if (request.resolve) {
        request.resolve(batchResponse)
      }

      // Remove from active requests
      const cacheKey = this.generateCacheKey(request.endpoint, request.method, request.data)
      this.activeRequests.delete(cacheKey)

    } catch (error) {
      const endTime = performance.now()
      const duration = endTime - startTime

      // Handle retry logic
      if (request.retryCount < request.maxRetries && this.shouldRetry(error)) {
        request.retryCount++
        const delay = this.config.retryDelays[Math.min(request.retryCount - 1, this.config.retryDelays.length - 1)]
        
        setTimeout(() => {
          this.requestQueue.unshift(request) // Add to front of queue for retry
        }, delay)
        
        return
      }

      const batchResponse: BatchResponse = {
        id: request.id,
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        duration
      }

      this.updateSuccessRate(false)

      if (request.reject) {
        request.reject(batchResponse)
      }

      // Remove from active requests
      const cacheKey = this.generateCacheKey(request.endpoint, request.method, request.data)
      this.activeRequests.delete(cacheKey)
    }
  }

  private async executeBatchRequest(batch: BatchRequest[]): Promise<void> {
    // For endpoints that support batching, combine requests
    const endpoint = batch[0].endpoint
    
    if (this.supportsBatching(endpoint)) {
      const batchData = {
        requests: batch.map(req => ({
          id: req.id,
          data: req.data
        }))
      }

      const startTime = performance.now()

      try {
        const response = await fetch(`${endpoint}/batch`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(batchData)
        })

        const responseData = await response.json()
        const endTime = performance.now()
        const duration = endTime - startTime

        // Distribute responses back to individual requests
        if (response.ok && Array.isArray(responseData.responses)) {
          responseData.responses.forEach((resp: any, index: number) => {
            const request = batch[index]
            const batchResponse: BatchResponse = {
              id: request.id,
              success: resp.success,
              data: resp.data,
              error: resp.error,
              statusCode: resp.statusCode || response.status,
              duration: duration / batch.length // Distribute duration
            }

            if (request.resolve) {
              request.resolve(batchResponse)
            }
          })
        }

        this.updateSuccessRate(response.ok)

      } catch (error) {
        // Fall back to individual requests
        await Promise.all(batch.map(req => this.executeSingleRequest(req)))
      }
    } else {
      // Execute requests individually but concurrently
      await Promise.all(batch.map(req => this.executeSingleRequest(req)))
    }
  }

  private checkRateLimit(endpoint: string): boolean {
    const config = this.config.rateLimits[endpoint] || this.config.rateLimits.default
    const now = Date.now()
    
    let limiter = this.rateLimiters.get(endpoint)
    if (!limiter) {
      limiter = { count: 0, windowStart: now, requests: [] }
      this.rateLimiters.set(endpoint, limiter)
    }

    if (config.windowType === 'fixed') {
      // Fixed window rate limiting
      if (now - limiter.windowStart >= config.window) {
        limiter.count = 0
        limiter.windowStart = now
      }

      if (limiter.count >= config.limit) {
        return false
      }

      limiter.count++
      return true

    } else {
      // Sliding window rate limiting
      const cutoff = now - config.window
      limiter.requests = limiter.requests.filter(time => time > cutoff)

      if (limiter.requests.length >= config.limit) {
        return false
      }

      limiter.requests.push(now)
      return true
    }
  }

  private getCachedResponse(cacheKey: string): BatchResponse | null {
    const cached = this.responseCache.get(cacheKey)
    if (!cached) return null

    const now = Date.now()
    if (now - cached.timestamp > this.config.cacheTimeout) {
      this.responseCache.delete(cacheKey)
      return null
    }

    return cached.response
  }

  private cacheResponse(cacheKey: string, response: BatchResponse): void {
    this.responseCache.set(cacheKey, {
      response,
      timestamp: Date.now()
    })
  }

  private cleanupCache(): void {
    const now = Date.now()
    for (const [key, cached] of this.responseCache.entries()) {
      if (now - cached.timestamp > this.config.cacheTimeout) {
        this.responseCache.delete(key)
      }
    }
  }

  private generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  private generateCacheKey(endpoint: string, method: string, data?: any): string {
    const dataHash = data ? JSON.stringify(data) : ''
    return `${method}:${endpoint}:${dataHash}`
  }

  private supportsBatching(endpoint: string): boolean {
    // Define which endpoints support batching
    const batchableEndpoints = ['/api/predict', '/api/batch']
    return batchableEndpoints.some(ep => endpoint.includes(ep))
  }

  private shouldRetry(error: any): boolean {
    // Retry on network errors, timeouts, and 5xx status codes
    if (error.name === 'AbortError') return false // Don't retry timeouts
    if (error.name === 'TypeError') return true // Network error
    if (error.status >= 500) return true // Server error
    return false
  }

  private updateSuccessRate(success: boolean): void {
    const total = this.metrics.totalRequests
    const currentSuccessRate = this.metrics.successRate
    this.metrics.successRate = (currentSuccessRate * (total - 1) + (success ? 1 : 0)) / total
  }

  private updateLatencyMetrics(duration: number): void {
    const total = this.metrics.batchedRequests
    const currentAverage = this.metrics.averageLatency
    this.metrics.averageLatency = (currentAverage * (total - 1) + duration) / total
  }

  // Public API methods
  getMetrics(): BatchMetrics {
    this.metrics.activeRequests = this.activeRequests.size
    this.metrics.queueSize = this.requestQueue.length
    return { ...this.metrics }
  }

  clearCache(): void {
    this.responseCache.clear()
    console.log('🧹 Request cache cleared')
  }

  updateConfig(newConfig: Partial<BatchConfig>): void {
    this.config = { ...this.config, ...newConfig }
    console.log('⚙️ Request batcher config updated')
  }

  exportReport(): string {
    const report = {
      timestamp: new Date().toISOString(),
      metrics: this.getMetrics(),
      config: this.config,
      queueStatus: {
        queueSize: this.requestQueue.length,
        activeRequests: this.activeRequests.size,
        cacheSize: this.responseCache.size
      },
      rateLimiters: Object.fromEntries(
        Array.from(this.rateLimiters.entries()).map(([key, limiter]) => [
          key,
          { count: limiter.count, requestsInWindow: limiter.requests.length }
        ])
      )
    }

    return JSON.stringify(report, null, 2)
  }
}

// Global request batcher instance
export const requestBatcher = new RequestBatcher()

// React hook for request batching
export function useRequestBatcher() {
  const [metrics, setMetrics] = React.useState<BatchMetrics>(requestBatcher.getMetrics())

  React.useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(requestBatcher.getMetrics())
    }, 1000)

    return () => clearInterval(interval)
  }, [])

  return {
    metrics,
    executeRequest: requestBatcher.executeRequest.bind(requestBatcher),
    executeDebounced: requestBatcher.executeDebounced,
    executeThrottled: requestBatcher.executeThrottled,
    clearCache: requestBatcher.clearCache.bind(requestBatcher),
    exportReport: requestBatcher.exportReport.bind(requestBatcher)
  }
}

export default RequestBatcher
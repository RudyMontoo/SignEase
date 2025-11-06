/**
 * Performance Optimizer
 * ====================
 * 
 * Comprehensive performance optimization utilities for SignEase
 */

// Performance monitoring interface
export interface PerformanceMetrics {
  fps: number
  memoryUsage: {
    used: number
    total: number
    percentage: number
  }
  apiLatency: {
    average: number
    min: number
    max: number
    recent: number[]
  }
  renderTime: {
    average: number
    recent: number[]
  }
  cacheHitRate: number
  activeConnections: number
}

// Performance optimization configuration
export interface OptimizationConfig {
  targetFPS: number
  maxMemoryUsage: number // MB
  apiTimeout: number // ms
  cacheSize: number
  batchSize: number
  throttleMs: number
}

class PerformanceOptimizer {
  private metrics: PerformanceMetrics
  private config: OptimizationConfig
  private frameCount: number = 0
  private lastFrameTime: number = 0
  private apiLatencies: number[] = []
  private renderTimes: number[] = []
  private requestCache: Map<string, { data: any; timestamp: number }> = new Map()
  private requestQueue: Array<{ request: Function; resolve: Function; reject: Function }> = []
  private activeRequests: number = 0
  private isOptimizing: boolean = false

  constructor(config: Partial<OptimizationConfig> = {}) {
    this.config = {
      targetFPS: 30,
      maxMemoryUsage: 512, // 512MB
      apiTimeout: 5000,
      cacheSize: 100,
      batchSize: 5,
      throttleMs: 33, // ~30 FPS
      ...config
    }

    this.metrics = {
      fps: 0,
      memoryUsage: { used: 0, total: 0, percentage: 0 },
      apiLatency: { average: 0, min: 0, max: 0, recent: [] },
      renderTime: { average: 0, recent: [] },
      cacheHitRate: 0,
      activeConnections: 0
    }

    this.startMonitoring()
  }

  // Start performance monitoring
  private startMonitoring(): void {
    // FPS monitoring
    this.monitorFPS()
    
    // Memory monitoring
    setInterval(() => this.updateMemoryMetrics(), 1000)
    
    // Cache cleanup
    setInterval(() => this.cleanupCache(), 30000) // Every 30 seconds
    
    // Performance optimization
    setInterval(() => this.optimizePerformance(), 5000) // Every 5 seconds
  }

  // Monitor FPS
  private monitorFPS(): void {
    const now = performance.now()
    
    if (this.lastFrameTime > 0) {
      const deltaTime = now - this.lastFrameTime
      const currentFPS = 1000 / deltaTime
      
      this.frameCount++
      
      // Calculate rolling average FPS
      if (this.frameCount % 10 === 0) {
        this.metrics.fps = Math.round(currentFPS)
      }
    }
    
    this.lastFrameTime = now
    requestAnimationFrame(() => this.monitorFPS())
  }

  // Update memory metrics
  private updateMemoryMetrics(): void {
    if ('memory' in performance) {
      const memory = (performance as any).memory
      this.metrics.memoryUsage = {
        used: Math.round(memory.usedJSHeapSize / 1024 / 1024), // MB
        total: Math.round(memory.totalJSHeapSize / 1024 / 1024), // MB
        percentage: Math.round((memory.usedJSHeapSize / memory.jsHeapSizeLimit) * 100)
      }
    }
  }

  // Track API request performance
  public trackAPIRequest<T>(requestFn: () => Promise<T>, cacheKey?: string): Promise<T> {
    return new Promise((resolve, reject) => {
      // Check cache first
      if (cacheKey && this.requestCache.has(cacheKey)) {
        const cached = this.requestCache.get(cacheKey)!
        const age = Date.now() - cached.timestamp
        
        if (age < 60000) { // Cache valid for 1 minute
          this.updateCacheHitRate(true)
          resolve(cached.data)
          return
        } else {
          this.requestCache.delete(cacheKey)
        }
      }

      this.updateCacheHitRate(false)

      // Queue request if too many active
      if (this.activeRequests >= this.config.batchSize) {
        this.requestQueue.push({
          request: () => this.executeRequest(requestFn, cacheKey),
          resolve,
          reject
        })
        return
      }

      this.executeRequest(requestFn, cacheKey).then(resolve).catch(reject)
    })
  }

  // Execute API request with performance tracking
  private async executeRequest<T>(requestFn: () => Promise<T>, cacheKey?: string): Promise<T> {
    this.activeRequests++
    this.metrics.activeConnections = this.activeRequests
    
    const startTime = performance.now()
    
    try {
      const result = await Promise.race([
        requestFn(),
        new Promise<never>((_, reject) => 
          setTimeout(() => reject(new Error('Request timeout')), this.config.apiTimeout)
        )
      ])
      
      const latency = performance.now() - startTime
      this.updateAPILatency(latency)
      
      // Cache successful result
      if (cacheKey && this.requestCache.size < this.config.cacheSize) {
        this.requestCache.set(cacheKey, {
          data: result,
          timestamp: Date.now()
        })
      }
      
      return result
    } finally {
      this.activeRequests--
      this.metrics.activeConnections = this.activeRequests
      
      // Process next queued request
      if (this.requestQueue.length > 0) {
        const next = this.requestQueue.shift()!
        next.request().then(next.resolve).catch(next.reject)
      }
    }
  }

  // Update API latency metrics
  private updateAPILatency(latency: number): void {
    this.apiLatencies.push(latency)
    
    // Keep only recent measurements
    if (this.apiLatencies.length > 50) {
      this.apiLatencies = this.apiLatencies.slice(-50)
    }
    
    this.metrics.apiLatency = {
      average: this.apiLatencies.reduce((sum, val) => sum + val, 0) / this.apiLatencies.length,
      min: Math.min(...this.apiLatencies),
      max: Math.max(...this.apiLatencies),
      recent: this.apiLatencies.slice(-10)
    }
  }

  // Track render performance
  public trackRender<T>(renderFn: () => T): T {
    const startTime = performance.now()
    const result = renderFn()
    const renderTime = performance.now() - startTime
    
    this.renderTimes.push(renderTime)
    
    // Keep only recent measurements
    if (this.renderTimes.length > 30) {
      this.renderTimes = this.renderTimes.slice(-30)
    }
    
    this.metrics.renderTime = {
      average: this.renderTimes.reduce((sum, val) => sum + val, 0) / this.renderTimes.length,
      recent: this.renderTimes.slice(-10)
    }
    
    return result
  }

  // Update cache hit rate
  private updateCacheHitRate(hit: boolean): void {
    // Simple rolling average for cache hit rate
    const currentRate = this.metrics.cacheHitRate
    this.metrics.cacheHitRate = currentRate * 0.9 + (hit ? 1 : 0) * 0.1
  }

  // Clean up expired cache entries
  private cleanupCache(): void {
    const now = Date.now()
    const expiredKeys: string[] = []
    
    for (const [key, value] of this.requestCache.entries()) {
      if (now - value.timestamp > 300000) { // 5 minutes
        expiredKeys.push(key)
      }
    }
    
    expiredKeys.forEach(key => this.requestCache.delete(key))
  }

  // Automatic performance optimization
  private optimizePerformance(): void {
    if (this.isOptimizing) return
    
    this.isOptimizing = true
    
    try {
      // Optimize based on FPS
      if (this.metrics.fps < this.config.targetFPS * 0.8) {
        this.optimizeFPS()
      }
      
      // Optimize based on memory usage
      if (this.metrics.memoryUsage.percentage > 80) {
        this.optimizeMemory()
      }
      
      // Optimize based on API latency
      if (this.metrics.apiLatency.average > 200) {
        this.optimizeAPI()
      }
    } finally {
      this.isOptimizing = false
    }
  }

  // Optimize FPS performance
  private optimizeFPS(): void {
    // Increase throttling to reduce render frequency
    this.config.throttleMs = Math.min(this.config.throttleMs * 1.1, 100)
    
    // Reduce batch size to process fewer items at once
    this.config.batchSize = Math.max(this.config.batchSize - 1, 1)
    
    console.log(`🎯 FPS Optimization: throttle=${this.config.throttleMs}ms, batch=${this.config.batchSize}`)
  }

  // Optimize memory usage
  private optimizeMemory(): void {
    // Clear cache aggressively
    const cacheSize = this.requestCache.size
    if (cacheSize > this.config.cacheSize * 0.5) {
      const keysToDelete = Array.from(this.requestCache.keys()).slice(0, Math.floor(cacheSize * 0.3))
      keysToDelete.forEach(key => this.requestCache.delete(key))
    }
    
    // Trigger garbage collection if available
    if ('gc' in window) {
      (window as any).gc()
    }
    
    console.log(`🧹 Memory Optimization: cleared ${cacheSize - this.requestCache.size} cache entries`)
  }

  // Optimize API performance
  private optimizeAPI(): void {
    // Reduce timeout for faster failure detection
    this.config.apiTimeout = Math.max(this.config.apiTimeout * 0.9, 1000)
    
    // Increase cache size to reduce API calls
    this.config.cacheSize = Math.min(this.config.cacheSize * 1.2, 200)
    
    console.log(`🚀 API Optimization: timeout=${this.config.apiTimeout}ms, cache=${this.config.cacheSize}`)
  }

  // Get current performance metrics
  public getMetrics(): PerformanceMetrics {
    return { ...this.metrics }
  }

  // Get optimization suggestions
  public getOptimizationSuggestions(): string[] {
    const suggestions: string[] = []
    
    if (this.metrics.fps < this.config.targetFPS * 0.8) {
      suggestions.push('Consider reducing render complexity or frequency')
    }
    
    if (this.metrics.memoryUsage.percentage > 70) {
      suggestions.push('Memory usage is high - consider clearing caches or reducing data retention')
    }
    
    if (this.metrics.apiLatency.average > 150) {
      suggestions.push('API latency is high - consider request batching or caching')
    }
    
    if (this.metrics.cacheHitRate < 0.3) {
      suggestions.push('Cache hit rate is low - consider adjusting cache strategy')
    }
    
    return suggestions
  }

  // Force garbage collection (if available)
  public forceGarbageCollection(): void {
    if ('gc' in window) {
      (window as any).gc()
      console.log('🗑️ Forced garbage collection')
    }
  }

  // Clear all caches
  public clearCaches(): void {
    this.requestCache.clear()
    console.log('🧹 Cleared all caches')
  }

  // Reset metrics
  public resetMetrics(): void {
    this.frameCount = 0
    this.apiLatencies = []
    this.renderTimes = []
    this.metrics.cacheHitRate = 0
    console.log('📊 Reset performance metrics')
  }

  // Export comprehensive performance report
  public exportReport(): string {
    const report = {
      timestamp: new Date().toISOString(),
      metrics: this.metrics,
      config: this.config,
      suggestions: this.getOptimizationSuggestions(),
      gpuMemory: gpuMemoryProfiler.getCurrentMetrics(),
      rendering: renderingOptimizer.getRenderMetrics(),
      requests: requestBatcher.getMetrics(),
      systemInfo: {
        userAgent: navigator.userAgent,
        platform: navigator.platform,
        hardwareConcurrency: navigator.hardwareConcurrency,
        deviceMemory: (navigator as any).deviceMemory,
        connection: (navigator as any).connection?.effectiveType
      }
    }
    
    return JSON.stringify(report, null, 2)
  }
}

import React from 'react'
import React from 'react'
// Import additional optimizers
import { gpuMemoryProfiler } from './gpuMemoryProfiler'
import { renderingOptimizer } from './renderingOptimizer'
import { requestBatcher } from './requestBatcher'

// Global performance optimizer instance
export const performanceOptimizer = new PerformanceOptimizer()

// Performance monitoring hook for React components
export function usePerformanceMonitoring() {
  const [metrics, setMetrics] = React.useState<PerformanceMetrics>(performanceOptimizer.getMetrics())
  
  React.useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(performanceOptimizer.getMetrics())
    }, 1000)
    
    return () => clearInterval(interval)
  }, [])
  
  return {
    metrics,
    trackRender: performanceOptimizer.trackRender.bind(performanceOptimizer),
    trackAPIRequest: performanceOptimizer.trackAPIRequest.bind(performanceOptimizer),
    getOptimizationSuggestions: performanceOptimizer.getOptimizationSuggestions.bind(performanceOptimizer),
    forceGarbageCollection: performanceOptimizer.forceGarbageCollection.bind(performanceOptimizer),
    clearCaches: performanceOptimizer.clearCaches.bind(performanceOptimizer),
    exportReport: performanceOptimizer.exportReport.bind(performanceOptimizer)
  }
}

// Throttle utility for performance optimization
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: NodeJS.Timeout | null = null
  let lastExecTime = 0
  
  return (...args: Parameters<T>) => {
    const currentTime = Date.now()
    
    if (currentTime - lastExecTime > delay) {
      func(...args)
      lastExecTime = currentTime
    } else {
      if (timeoutId) clearTimeout(timeoutId)
      timeoutId = setTimeout(() => {
        func(...args)
        lastExecTime = Date.now()
      }, delay - (currentTime - lastExecTime))
    }
  }
}

// Debounce utility for performance optimization
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: NodeJS.Timeout | null = null
  
  return (...args: Parameters<T>) => {
    if (timeoutId) clearTimeout(timeoutId)
    timeoutId = setTimeout(() => func(...args), delay)
  }
}

// Batch processing utility
export class BatchProcessor<T> {
  private queue: T[] = []
  private processing = false
  
  constructor(
    private processFn: (items: T[]) => Promise<void>,
    private batchSize: number = 10,
    private delay: number = 100
  ) {}
  
  add(item: T): void {
    this.queue.push(item)
    this.scheduleProcessing()
  }
  
  private scheduleProcessing(): void {
    if (this.processing) return
    
    setTimeout(() => this.processBatch(), this.delay)
  }
  
  private async processBatch(): Promise<void> {
    if (this.processing || this.queue.length === 0) return
    
    this.processing = true
    
    try {
      const batch = this.queue.splice(0, this.batchSize)
      await this.processFn(batch)
      
      // Process remaining items if any
      if (this.queue.length > 0) {
        setTimeout(() => this.processBatch(), this.delay)
      }
    } finally {
      this.processing = false
    }
  }
}

export default PerformanceOptimizer
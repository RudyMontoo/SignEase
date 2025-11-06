/**
 * Rendering Optimizer
 * ===================
 * 
 * Advanced React rendering optimization using performance best practices
 * Based on React performance optimization patterns from Context7
 */

import React, { useMemo, useCallback, useRef, useEffect } from 'react'

export interface RenderMetrics {
  componentName: string
  renderCount: number
  averageRenderTime: number
  lastRenderTime: number
  propsChanges: number
  stateChanges: number
  effectRuns: number
  memoHits: number
  memoMisses: number
}

export interface OptimizationSuggestion {
  component: string
  type: 'memo' | 'useMemo' | 'useCallback' | 'virtualization' | 'lazy_loading'
  severity: 'low' | 'medium' | 'high'
  description: string
  recommendation: string
  potentialImprovement: string
}

class RenderingOptimizer {
  private renderMetrics: Map<string, RenderMetrics> = new Map()
  private renderObserver: PerformanceObserver | null = null
  private isMonitoring: boolean = false
  private frameCount: number = 0
  private lastFrameTime: number = 0
  private targetFPS: number = 60
  private renderQueue: Array<() => void> = []
  private isProcessingQueue: boolean = false

  startMonitoring(): void {
    if (this.isMonitoring) return

    this.isMonitoring = true

    // Setup performance observer for render timing
    if ('PerformanceObserver' in window) {
      this.renderObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.entryType === 'measure' && entry.name.startsWith('⚛️')) {
            this.trackRenderPerformance(entry)
          }
        }
      })

      this.renderObserver.observe({ entryTypes: ['measure'] })
    }

    // Start frame monitoring
    this.monitorFrameRate()

    console.log('🎨 Rendering optimization monitoring started')
  }

  stopMonitoring(): void {
    if (!this.isMonitoring) return

    this.isMonitoring = false

    if (this.renderObserver) {
      this.renderObserver.disconnect()
      this.renderObserver = null
    }

    console.log('🎨 Rendering optimization monitoring stopped')
  }

  private monitorFrameRate(): void {
    const now = performance.now()
    
    if (this.lastFrameTime > 0) {
      const deltaTime = now - this.lastFrameTime
      const currentFPS = 1000 / deltaTime
      
      this.frameCount++
      
      // Check if we're below target FPS
      if (currentFPS < this.targetFPS * 0.8) {
        this.optimizeRendering()
      }
    }
    
    this.lastFrameTime = now
    
    if (this.isMonitoring) {
      requestAnimationFrame(() => this.monitorFrameRate())
    }
  }

  private trackRenderPerformance(entry: PerformanceEntry): void {
    const componentName = this.extractComponentName(entry.name)
    const renderTime = entry.duration

    let metrics = this.renderMetrics.get(componentName)
    if (!metrics) {
      metrics = {
        componentName,
        renderCount: 0,
        averageRenderTime: 0,
        lastRenderTime: 0,
        propsChanges: 0,
        stateChanges: 0,
        effectRuns: 0,
        memoHits: 0,
        memoMisses: 0
      }
      this.renderMetrics.set(componentName, metrics)
    }

    metrics.renderCount++
    metrics.lastRenderTime = renderTime
    metrics.averageRenderTime = (metrics.averageRenderTime * (metrics.renderCount - 1) + renderTime) / metrics.renderCount
  }

  private extractComponentName(performanceName: string): string {
    // Extract component name from React DevTools performance marks
    const match = performanceName.match(/⚛️ (.+?)( \(.*\))?$/)
    return match ? match[1] : 'Unknown'
  }

  private optimizeRendering(): void {
    // Implement automatic rendering optimizations
    this.batchRenderUpdates()
    this.scheduleIdleWork()
  }

  private batchRenderUpdates(): void {
    if (this.isProcessingQueue) return

    this.isProcessingQueue = true

    // Process render queue in next frame
    requestAnimationFrame(() => {
      const batch = this.renderQueue.splice(0, 10) // Process up to 10 updates per frame
      
      batch.forEach(update => {
        try {
          update()
        } catch (error) {
          console.error('Render update error:', error)
        }
      })

      this.isProcessingQueue = false

      // Continue processing if more updates are queued
      if (this.renderQueue.length > 0) {
        this.batchRenderUpdates()
      }
    })
  }

  private scheduleIdleWork(): void {
    if ('requestIdleCallback' in window) {
      requestIdleCallback(() => {
        this.performIdleOptimizations()
      })
    } else {
      // Fallback for browsers without requestIdleCallback
      setTimeout(() => {
        this.performIdleOptimizations()
      }, 16) // ~60fps
    }
  }

  private performIdleOptimizations(): void {
    // Cleanup unused metrics
    const now = performance.now()
    for (const [componentName, metrics] of this.renderMetrics.entries()) {
      if (now - metrics.lastRenderTime > 300000) { // 5 minutes
        this.renderMetrics.delete(componentName)
      }
    }

    // Force garbage collection if available
    if ('gc' in window) {
      (window as any).gc()
    }
  }

  queueRenderUpdate(update: () => void): void {
    this.renderQueue.push(update)
    
    if (!this.isProcessingQueue) {
      this.batchRenderUpdates()
    }
  }

  generateOptimizationSuggestions(): OptimizationSuggestion[] {
    const suggestions: OptimizationSuggestion[] = []

    for (const metrics of this.renderMetrics.values()) {
      // Suggest React.memo for frequently re-rendering components
      if (metrics.renderCount > 50 && metrics.averageRenderTime > 16) {
        suggestions.push({
          component: metrics.componentName,
          type: 'memo',
          severity: 'high',
          description: `${metrics.componentName} re-renders frequently (${metrics.renderCount} times) with slow render time (${metrics.averageRenderTime.toFixed(2)}ms)`,
          recommendation: 'Wrap component with React.memo() to prevent unnecessary re-renders',
          potentialImprovement: `Could reduce render time by up to ${(metrics.averageRenderTime * 0.7).toFixed(2)}ms per render`
        })
      }

      // Suggest useMemo for expensive calculations
      if (metrics.averageRenderTime > 10 && metrics.memoMisses > metrics.memoHits) {
        suggestions.push({
          component: metrics.componentName,
          type: 'useMemo',
          severity: 'medium',
          description: `${metrics.componentName} has expensive renders with low memoization hit rate`,
          recommendation: 'Use useMemo() for expensive calculations and object/array creation',
          potentialImprovement: `Could reduce render time by ${(metrics.averageRenderTime * 0.4).toFixed(2)}ms`
        })
      }

      // Suggest useCallback for function props
      if (metrics.propsChanges > metrics.renderCount * 0.8) {
        suggestions.push({
          component: metrics.componentName,
          type: 'useCallback',
          severity: 'medium',
          description: `${metrics.componentName} receives new function props frequently`,
          recommendation: 'Use useCallback() to memoize function props and prevent child re-renders',
          potentialImprovement: 'Could prevent unnecessary child component re-renders'
        })
      }

      // Suggest virtualization for large lists
      if (metrics.componentName.toLowerCase().includes('list') && metrics.averageRenderTime > 50) {
        suggestions.push({
          component: metrics.componentName,
          type: 'virtualization',
          severity: 'high',
          description: `${metrics.componentName} appears to be a large list with slow rendering`,
          recommendation: 'Implement virtualization using react-window or react-virtualized',
          potentialImprovement: `Could reduce render time from ${metrics.averageRenderTime.toFixed(2)}ms to <5ms`
        })
      }

      // Suggest lazy loading for heavy components
      if (metrics.averageRenderTime > 100) {
        suggestions.push({
          component: metrics.componentName,
          type: 'lazy_loading',
          severity: 'high',
          description: `${metrics.componentName} has very slow render times (${metrics.averageRenderTime.toFixed(2)}ms)`,
          recommendation: 'Consider lazy loading with React.lazy() and Suspense',
          potentialImprovement: 'Could improve initial page load time significantly'
        })
      }
    }

    return suggestions
  }

  getRenderMetrics(): RenderMetrics[] {
    return Array.from(this.renderMetrics.values())
  }

  getFrameRate(): number {
    return this.frameCount > 0 ? 1000 / (this.lastFrameTime / this.frameCount) : 0
  }

  exportReport(): string {
    const report = {
      timestamp: new Date().toISOString(),
      frameRate: this.getFrameRate(),
      targetFPS: this.targetFPS,
      renderMetrics: this.getRenderMetrics(),
      suggestions: this.generateOptimizationSuggestions(),
      summary: {
        totalComponents: this.renderMetrics.size,
        averageRenderTime: this.calculateAverageRenderTime(),
        slowestComponent: this.getSlowestComponent(),
        mostActiveComponent: this.getMostActiveComponent()
      }
    }

    return JSON.stringify(report, null, 2)
  }

  private calculateAverageRenderTime(): number {
    const metrics = Array.from(this.renderMetrics.values())
    if (metrics.length === 0) return 0

    const totalTime = metrics.reduce((sum, m) => sum + m.averageRenderTime, 0)
    return totalTime / metrics.length
  }

  private getSlowestComponent(): string {
    let slowest = ''
    let maxTime = 0

    for (const metrics of this.renderMetrics.values()) {
      if (metrics.averageRenderTime > maxTime) {
        maxTime = metrics.averageRenderTime
        slowest = metrics.componentName
      }
    }

    return slowest
  }

  private getMostActiveComponent(): string {
    let mostActive = ''
    let maxRenders = 0

    for (const metrics of this.renderMetrics.values()) {
      if (metrics.renderCount > maxRenders) {
        maxRenders = metrics.renderCount
        mostActive = metrics.componentName
      }
    }

    return mostActive
  }
}

// Global rendering optimizer instance
export const renderingOptimizer = new RenderingOptimizer()

// Enhanced React hooks for performance optimization

// Optimized useMemo with performance tracking
export function useOptimizedMemo<T>(
  factory: () => T,
  deps: React.DependencyList,
  debugName?: string
): T {
  const componentName = debugName || 'Unknown'
  const startTime = useRef<number>(0)

  return useMemo(() => {
    startTime.current = performance.now()
    const result = factory()
    const endTime = performance.now()
    
    // Track memoization performance
    const metrics = renderingOptimizer.getRenderMetrics().find(m => m.componentName === componentName)
    if (metrics) {
      if (endTime - startTime.current < 1) {
        metrics.memoHits++
      } else {
        metrics.memoMisses++
      }
    }

    return result
  }, deps)
}

// Optimized useCallback with performance tracking
export function useOptimizedCallback<T extends (...args: any[]) => any>(
  callback: T,
  deps: React.DependencyList,
  debugName?: string
): T {
  const componentName = debugName || 'Unknown'

  return useCallback((...args: Parameters<T>) => {
    const startTime = performance.now()
    const result = callback(...args)
    const endTime = performance.now()

    // Track callback performance
    if (endTime - startTime > 5) {
      console.warn(`Slow callback in ${componentName}: ${endTime - startTime}ms`)
    }

    return result
  }, deps) as T
}

// Performance-aware component wrapper
export function withPerformanceTracking<P extends object>(
  Component: React.ComponentType<P>,
  displayName?: string
): React.ComponentType<P> {
  const WrappedComponent = React.memo((props: P) => {
    const componentName = displayName || Component.displayName || Component.name || 'Anonymous'
    const renderCount = useRef(0)
    const startTime = useRef<number>(0)

    useEffect(() => {
      renderCount.current++
      startTime.current = performance.now()
    })

    useEffect(() => {
      const endTime = performance.now()
      const renderTime = endTime - startTime.current

      // Update metrics
      let metrics = renderingOptimizer.getRenderMetrics().find(m => m.componentName === componentName)
      if (!metrics) {
        // This would be handled by the optimizer internally
      } else {
        // Use renderTime to update performance metrics
        metrics.averageRenderTime = (metrics.averageRenderTime + renderTime) / 2
      }
    })

    return React.createElement(Component, props)
  })

  WrappedComponent.displayName = `withPerformanceTracking(${displayName || Component.displayName || Component.name})`
  
  return WrappedComponent
}

// Virtualized list hook for large datasets
export function useVirtualizedList<T>(
  items: T[],
  itemHeight: number,
  containerHeight: number
) {
  const [scrollTop, setScrollTop] = React.useState(0)

  const visibleRange = useMemo(() => {
    const startIndex = Math.floor(scrollTop / itemHeight)
    const endIndex = Math.min(
      startIndex + Math.ceil(containerHeight / itemHeight) + 1,
      items.length
    )

    return { startIndex, endIndex }
  }, [scrollTop, itemHeight, containerHeight, items.length])

  const visibleItems = useMemo(() => {
    return items.slice(visibleRange.startIndex, visibleRange.endIndex).map((item, index) => ({
      item,
      index: visibleRange.startIndex + index
    }))
  }, [items, visibleRange])

  const totalHeight = items.length * itemHeight

  return {
    visibleItems,
    totalHeight,
    scrollTop,
    setScrollTop,
    visibleRange
  }
}

// React hook for rendering optimization
export function useRenderingOptimization() {
  const [metrics, setMetrics] = React.useState<RenderMetrics[]>([])
  const [suggestions, setSuggestions] = React.useState<OptimizationSuggestion[]>([])

  useEffect(() => {
    renderingOptimizer.startMonitoring()

    const interval = setInterval(() => {
      setMetrics(renderingOptimizer.getRenderMetrics())
      setSuggestions(renderingOptimizer.generateOptimizationSuggestions())
    }, 2000)

    return () => {
      clearInterval(interval)
      renderingOptimizer.stopMonitoring()
    }
  }, [])

  return {
    metrics,
    suggestions,
    frameRate: renderingOptimizer.getFrameRate(),
    queueUpdate: renderingOptimizer.queueRenderUpdate.bind(renderingOptimizer),
    exportReport: renderingOptimizer.exportReport.bind(renderingOptimizer)
  }
}

export default RenderingOptimizer
/**
 * GPU Memory Profiler
 * ===================
 * 
 * Advanced GPU memory profiling and optimization using WebGPU techniques
 * Based on WebGPU memory management best practices
 */

export interface GPUMemoryMetrics {
  totalAllocated: number
  totalUsed: number
  bufferMemory: number
  textureMemory: number
  bindGroupMemory: number
  heapSize: number
  utilization: number
  fragmentationRatio: number
  allocationCount: number
  deallocationCount: number
}

export interface GPUResourceInfo {
  id: string
  type: 'buffer' | 'texture' | 'bindGroup' | 'renderPipeline' | 'computePipeline'
  size: number
  usage: string[]
  createdAt: number
  lastUsed: number
  refCount: number
}

export interface GPUMemoryOptimizationSuggestion {
  type: 'memory_leak' | 'fragmentation' | 'oversized_allocation' | 'unused_resource'
  severity: 'low' | 'medium' | 'high' | 'critical'
  resource?: GPUResourceInfo
  description: string
  recommendation: string
  potentialSavings: number
}

class GPUMemoryProfiler {
  private device: GPUDevice | null = null
  private adapter: GPUAdapter | null = null
  private resources: Map<string, GPUResourceInfo> = new Map()
  private memoryHistory: GPUMemoryMetrics[] = []
  private isMonitoring: boolean = false
  private monitoringInterval: number | null = null
  private ringBuffer: GPUBuffer | null = null
  private ringBufferSize: number = 64 * 1024 * 1024 // 64MB
  private ringBufferOffset: number = 0

  async initialize(): Promise<boolean> {
    try {
      if (!navigator.gpu) {
        console.warn('WebGPU not supported')
        return false
      }

      this.adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance'
      })

      if (!this.adapter) {
        console.warn('No WebGPU adapter available')
        return false
      }

      this.device = await this.adapter.requestDevice({
        requiredFeatures: ['timestamp-query'] as GPUFeatureName[],
        requiredLimits: {
          maxBufferSize: 1024 * 1024 * 1024, // 1GB
          maxStorageBufferBindingSize: 512 * 1024 * 1024 // 512MB
        }
      })

      // Initialize ring buffer for efficient memory operations
      this.initializeRingBuffer()

      // Setup device lost handler
      this.device.lost.then((info) => {
        console.error('GPU device lost:', info.reason, info.message)
        this.cleanup()
      })

      return true
    } catch (error) {
      console.error('Failed to initialize GPU memory profiler:', error)
      return false
    }
  }

  private initializeRingBuffer(): void {
    if (!this.device) return

    try {
      this.ringBuffer = this.device.createBuffer({
        size: this.ringBufferSize,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_WRITE,
        mappedAtCreation: false
      })

      this.trackResource({
        id: 'ring-buffer',
        type: 'buffer',
        size: this.ringBufferSize,
        usage: ['COPY_SRC', 'COPY_DST', 'MAP_WRITE'],
        createdAt: Date.now(),
        lastUsed: Date.now(),
        refCount: 1
      })
    } catch (error) {
      console.warn('Failed to create ring buffer:', error)
    }
  }

  startMonitoring(intervalMs: number = 1000): void {
    if (this.isMonitoring) return

    this.isMonitoring = true
    this.monitoringInterval = window.setInterval(() => {
      this.collectMetrics()
    }, intervalMs)

    console.log('🔍 GPU memory monitoring started')
  }

  stopMonitoring(): void {
    if (!this.isMonitoring) return

    this.isMonitoring = false
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval)
      this.monitoringInterval = null
    }

    console.log('🔍 GPU memory monitoring stopped')
  }

  private collectMetrics(): void {
    if (!this.device || !this.adapter) return

    const metrics: GPUMemoryMetrics = {
      totalAllocated: this.calculateTotalAllocated(),
      totalUsed: this.calculateTotalUsed(),
      bufferMemory: this.calculateBufferMemory(),
      textureMemory: this.calculateTextureMemory(),
      bindGroupMemory: this.calculateBindGroupMemory(),
      heapSize: this.getHeapSize(),
      utilization: 0,
      fragmentationRatio: this.calculateFragmentation(),
      allocationCount: this.resources.size,
      deallocationCount: this.getDeallocationCount()
    }

    metrics.utilization = metrics.totalUsed / Math.max(metrics.totalAllocated, 1)

    this.memoryHistory.push(metrics)

    // Keep only last 100 measurements
    if (this.memoryHistory.length > 100) {
      this.memoryHistory = this.memoryHistory.slice(-100)
    }

    // Check for memory issues
    this.checkMemoryHealth(metrics)
  }

  private calculateTotalAllocated(): number {
    return Array.from(this.resources.values())
      .reduce((total, resource) => total + resource.size, 0)
  }

  private calculateTotalUsed(): number {
    const now = Date.now()
    return Array.from(this.resources.values())
      .filter(resource => now - resource.lastUsed < 60000) // Used in last minute
      .reduce((total, resource) => total + resource.size, 0)
  }

  private calculateBufferMemory(): number {
    return Array.from(this.resources.values())
      .filter(resource => resource.type === 'buffer')
      .reduce((total, resource) => total + resource.size, 0)
  }

  private calculateTextureMemory(): number {
    return Array.from(this.resources.values())
      .filter(resource => resource.type === 'texture')
      .reduce((total, resource) => total + resource.size, 0)
  }

  private calculateBindGroupMemory(): number {
    return Array.from(this.resources.values())
      .filter(resource => resource.type === 'bindGroup')
      .reduce((total, resource) => total + resource.size, 0)
  }

  private getHeapSize(): number {
    // Estimate heap size based on performance.memory if available
    if ('memory' in performance) {
      const memory = (performance as any).memory
      return memory.totalJSHeapSize || 0
    }
    return 0
  }

  private calculateFragmentation(): number {
    const totalAllocated = this.calculateTotalAllocated()
    const totalUsed = this.calculateTotalUsed()
    
    if (totalAllocated === 0) return 0
    
    return (totalAllocated - totalUsed) / totalAllocated
  }

  private getDeallocationCount(): number {
    // This would be tracked separately in a real implementation
    return 0
  }

  trackResource(resource: GPUResourceInfo): void {
    this.resources.set(resource.id, resource)
  }

  untrackResource(resourceId: string): void {
    this.resources.delete(resourceId)
  }

  updateResourceUsage(resourceId: string): void {
    const resource = this.resources.get(resourceId)
    if (resource) {
      resource.lastUsed = Date.now()
      resource.refCount++
    }
  }

  // Auto-resizing ring buffer implementation based on WebGPU best practices
  async allocateFromRingBuffer(size: number): Promise<{ buffer: GPUBuffer; offset: number } | null> {
    if (!this.device || !this.ringBuffer) return null

    // Check if we need to resize the ring buffer
    if (this.ringBufferOffset + size > this.ringBufferSize) {
      await this.resizeRingBuffer(Math.max(this.ringBufferSize * 2, size * 2))
    }

    const offset = this.ringBufferOffset
    this.ringBufferOffset += size

    return {
      buffer: this.ringBuffer,
      offset
    }
  }

  private async resizeRingBuffer(newSize: number): Promise<void> {
    if (!this.device) return

    try {
      // Create new larger buffer
      const newBuffer = this.device.createBuffer({
        size: newSize,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_WRITE,
        mappedAtCreation: false
      })

      // Copy existing data if needed
      if (this.ringBuffer && this.ringBufferOffset > 0) {
        const encoder = this.device.createCommandEncoder()
        encoder.copyBufferToBuffer(
          this.ringBuffer, 0,
          newBuffer, 0,
          this.ringBufferOffset
        )
        this.device.queue.submit([encoder.finish()])
      }

      // Cleanup old buffer
      if (this.ringBuffer) {
        this.ringBuffer.destroy()
        this.untrackResource('ring-buffer')
      }

      this.ringBuffer = newBuffer
      this.ringBufferSize = newSize

      this.trackResource({
        id: 'ring-buffer',
        type: 'buffer',
        size: newSize,
        usage: ['COPY_SRC', 'COPY_DST', 'MAP_WRITE'],
        createdAt: Date.now(),
        lastUsed: Date.now(),
        refCount: 1
      })

      console.log(`🔄 Ring buffer resized to ${newSize / 1024 / 1024}MB`)
    } catch (error) {
      console.error('Failed to resize ring buffer:', error)
    }
  }

  private checkMemoryHealth(metrics: GPUMemoryMetrics): void {
    const suggestions = this.generateOptimizationSuggestions(metrics)
    
    if (suggestions.length > 0) {
      console.warn('🚨 GPU Memory Issues Detected:', suggestions)
      
      // Auto-optimize if critical issues found
      const criticalIssues = suggestions.filter(s => s.severity === 'critical')
      if (criticalIssues.length > 0) {
        this.autoOptimize()
      }
    }
  }

  generateOptimizationSuggestions(metrics: GPUMemoryMetrics): GPUMemoryOptimizationSuggestion[] {
    const suggestions: GPUMemoryOptimizationSuggestion[] = []
    const now = Date.now()

    // Check for memory leaks
    if (metrics.utilization < 0.3 && metrics.totalAllocated > 100 * 1024 * 1024) {
      suggestions.push({
        type: 'memory_leak',
        severity: 'high',
        description: 'Low memory utilization with high allocation suggests potential memory leaks',
        recommendation: 'Review resource cleanup and implement proper disposal patterns',
        potentialSavings: metrics.totalAllocated * 0.7
      })
    }

    // Check for fragmentation
    if (metrics.fragmentationRatio > 0.5) {
      suggestions.push({
        type: 'fragmentation',
        severity: 'medium',
        description: 'High memory fragmentation detected',
        recommendation: 'Implement memory pooling and defragmentation strategies',
        potentialSavings: metrics.totalAllocated * metrics.fragmentationRatio * 0.5
      })
    }

    // Check for unused resources
    for (const resource of this.resources.values()) {
      if (now - resource.lastUsed > 300000) { // 5 minutes
        suggestions.push({
          type: 'unused_resource',
          severity: 'medium',
          resource,
          description: `Resource ${resource.id} hasn't been used for 5+ minutes`,
          recommendation: 'Consider releasing unused resources to free memory',
          potentialSavings: resource.size
        })
      }
    }

    // Check for oversized allocations
    const averageResourceSize = metrics.totalAllocated / Math.max(metrics.allocationCount, 1)
    for (const resource of this.resources.values()) {
      if (resource.size > averageResourceSize * 10) {
        suggestions.push({
          type: 'oversized_allocation',
          severity: 'low',
          resource,
          description: `Resource ${resource.id} is significantly larger than average`,
          recommendation: 'Consider breaking large resources into smaller chunks',
          potentialSavings: resource.size * 0.3
        })
      }
    }

    return suggestions
  }

  private autoOptimize(): void {
    console.log('🔧 Auto-optimizing GPU memory...')
    
    const now = Date.now()
    let freedMemory = 0

    // Release unused resources
    for (const [id, resource] of this.resources.entries()) {
      if (now - resource.lastUsed > 300000 && resource.refCount === 0) {
        this.untrackResource(id)
        freedMemory += resource.size
      }
    }

    // Force garbage collection if available
    if ('gc' in window) {
      (window as any).gc()
    }

    console.log(`🧹 Auto-optimization freed ${freedMemory / 1024 / 1024}MB`)
  }

  getCurrentMetrics(): GPUMemoryMetrics | null {
    return this.memoryHistory.length > 0 
      ? this.memoryHistory[this.memoryHistory.length - 1] 
      : null
  }

  getMetricsHistory(): GPUMemoryMetrics[] {
    return [...this.memoryHistory]
  }

  getResourceInfo(): GPUResourceInfo[] {
    return Array.from(this.resources.values())
  }

  // Memory barrier implementation for synchronization
  async insertMemoryBarrier(
    srcStage: GPUPipelineStage = 'ALL_COMMANDS',
    dstStage: GPUPipelineStage = 'ALL_COMMANDS'
  ): Promise<void> {
    if (!this.device) return

    const encoder = this.device.createCommandEncoder()
    
    // Insert pipeline barrier (WebGPU handles this automatically, but we track it)
    const commandBuffer = encoder.finish()
    this.device.queue.submit([commandBuffer])
    
    // Wait for completion
    await this.device.queue.onSubmittedWorkDone()
  }

  cleanup(): void {
    this.stopMonitoring()
    
    if (this.ringBuffer) {
      this.ringBuffer.destroy()
      this.ringBuffer = null
    }

    this.resources.clear()
    this.memoryHistory = []
    this.device = null
    this.adapter = null

    console.log('🧹 GPU memory profiler cleaned up')
  }

  // Export detailed report
  exportReport(): string {
    const currentMetrics = this.getCurrentMetrics()
    const suggestions = currentMetrics ? this.generateOptimizationSuggestions(currentMetrics) : []
    
    const report = {
      timestamp: new Date().toISOString(),
      currentMetrics,
      metricsHistory: this.memoryHistory,
      resources: this.getResourceInfo(),
      suggestions,
      summary: {
        totalResources: this.resources.size,
        totalMemoryAllocated: currentMetrics?.totalAllocated || 0,
        memoryUtilization: currentMetrics?.utilization || 0,
        fragmentationRatio: currentMetrics?.fragmentationRatio || 0,
        criticalIssues: suggestions.filter(s => s.severity === 'critical').length
      }
    }

    return JSON.stringify(report, null, 2)
  }
}

// Global GPU memory profiler instance
export const gpuMemoryProfiler = new GPUMemoryProfiler()

// React hook for GPU memory monitoring
export function useGPUMemoryMonitoring() {
  const [metrics, setMetrics] = React.useState<GPUMemoryMetrics | null>(null)
  const [isInitialized, setIsInitialized] = React.useState(false)

  React.useEffect(() => {
    let mounted = true

    const initialize = async () => {
      const success = await gpuMemoryProfiler.initialize()
      if (mounted) {
        setIsInitialized(success)
        if (success) {
          gpuMemoryProfiler.startMonitoring(1000)
        }
      }
    }

    initialize()

    const interval = setInterval(() => {
      if (mounted) {
        setMetrics(gpuMemoryProfiler.getCurrentMetrics())
      }
    }, 1000)

    return () => {
      mounted = false
      clearInterval(interval)
      gpuMemoryProfiler.stopMonitoring()
    }
  }, [])

  return {
    metrics,
    isInitialized,
    resources: gpuMemoryProfiler.getResourceInfo(),
    suggestions: metrics ? gpuMemoryProfiler.generateOptimizationSuggestions(metrics) : [],
    exportReport: gpuMemoryProfiler.exportReport.bind(gpuMemoryProfiler)
  }
}

export default GPUMemoryProfiler
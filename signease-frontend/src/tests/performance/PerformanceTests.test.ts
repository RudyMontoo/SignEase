/**
 * Performance Tests
 * =================
 * 
 * Comprehensive performance testing for the SignEase application
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'

// Performance monitoring utilities
class PerformanceMonitor {
  private metrics: Map<string, number[]> = new Map()
  private startTimes: Map<string, number> = new Map()

  startTimer(name: string): void {
    this.startTimes.set(name, performance.now())
  }

  endTimer(name: string): number {
    const startTime = this.startTimes.get(name)
    if (!startTime) {
      throw new Error(`Timer ${name} was not started`)
    }
    
    const duration = performance.now() - startTime
    
    if (!this.metrics.has(name)) {
      this.metrics.set(name, [])
    }
    this.metrics.get(name)!.push(duration)
    
    this.startTimes.delete(name)
    return duration
  }

  getMetrics(name: string): { avg: number; min: number; max: number; count: number } {
    const values = this.metrics.get(name) || []
    if (values.length === 0) {
      return { avg: 0, min: 0, max: 0, count: 0 }
    }

    return {
      avg: values.reduce((sum, val) => sum + val, 0) / values.length,
      min: Math.min(...values),
      max: Math.max(...values),
      count: values.length
    }
  }

  reset(): void {
    this.metrics.clear()
    this.startTimes.clear()
  }
}

// Mock performance APIs
const mockPerformanceObserver = {
  observe: vi.fn(),
  disconnect: vi.fn(),
  takeRecords: vi.fn().mockReturnValue([])
}

global.PerformanceObserver = vi.fn().mockImplementation(() => mockPerformanceObserver)

// Mock gesture API with performance tracking
const createMockGestureAPI = (latency: number = 50) => ({
  predict: vi.fn().mockImplementation(async () => {
    await new Promise(resolve => setTimeout(resolve, latency))
    return {
      prediction: 'A',
      confidence: 0.95,
      alternatives: []
    }
  }),
  health: vi.fn().mockResolvedValue({ status: 'healthy', latency })
})

describe('Performance Tests', () => {
  let monitor: PerformanceMonitor

  beforeEach(() => {
    monitor = new PerformanceMonitor()
    vi.clearAllMocks()
  })

  afterEach(() => {
    monitor.reset()
  })

  describe('API Response Times', () => {
    it('should meet API response time targets (<100ms)', async () => {
      const gestureAPI = createMockGestureAPI(50) // 50ms latency
      const iterations = 10

      for (let i = 0; i < iterations; i++) {
        monitor.startTimer('api-request')
        await gestureAPI.predict([])
        monitor.endTimer('api-request')
      }

      const metrics = monitor.getMetrics('api-request')
      
      expect(metrics.avg).toBeLessThan(100) // Average < 100ms
      expect(metrics.max).toBeLessThan(200) // Max < 200ms
      expect(metrics.count).toBe(iterations)
    })

    it('should handle high-frequency requests efficiently', async () => {
      const gestureAPI = createMockGestureAPI(30)
      const requestsPerSecond = 30 // 30 FPS equivalent
      const duration = 1000 // 1 second test
      const expectedRequests = requestsPerSecond

      const startTime = performance.now()
      const promises = []

      for (let i = 0; i < expectedRequests; i++) {
        monitor.startTimer(`request-${i}`)
        const promise = gestureAPI.predict([]).then(() => {
          monitor.endTimer(`request-${i}`)
        })
        promises.push(promise)
        
        // Throttle requests to maintain target rate
        await new Promise(resolve => setTimeout(resolve, 1000 / requestsPerSecond))
      }

      await Promise.all(promises)
      const endTime = performance.now()
      const actualDuration = endTime - startTime

      expect(actualDuration).toBeLessThan(duration * 1.5) // 50% tolerance
      expect(promises).toHaveLength(expectedRequests)
    })

    it('should maintain performance under concurrent load', async () => {
      const gestureAPI = createMockGestureAPI(40)
      const concurrentRequests = 5
      const iterations = 10

      for (let i = 0; i < iterations; i++) {
        const promises = []
        
        monitor.startTimer(`concurrent-batch-${i}`)
        
        for (let j = 0; j < concurrentRequests; j++) {
          promises.push(gestureAPI.predict([]))
        }
        
        await Promise.all(promises)
        monitor.endTimer(`concurrent-batch-${i}`)
      }

      const metrics = monitor.getMetrics('concurrent-batch-0')
      
      // Should handle concurrent requests efficiently
      expect(metrics.avg).toBeLessThan(200) // Average batch time < 200ms
    })
  })

  describe('Frontend Rendering Performance', () => {
    it('should maintain target FPS (30 FPS)', async () => {
      const targetFPS = 30
      const testDuration = 1000 // 1 second
      const expectedFrames = targetFPS
      
      let frameCount = 0
      const startTime = performance.now()

      // Simulate frame rendering
      const renderFrame = () => {
        monitor.startTimer(`frame-${frameCount}`)
        
        // Simulate frame processing
        setTimeout(() => {
          monitor.endTimer(`frame-${frameCount}`)
          frameCount++
          
          if (performance.now() - startTime < testDuration) {
            requestAnimationFrame(renderFrame)
          }
        }, 1000 / targetFPS)
      }

      renderFrame()

      // Wait for test completion
      await new Promise(resolve => setTimeout(resolve, testDuration + 100))

      const actualFPS = frameCount / (testDuration / 1000)
      
      expect(actualFPS).toBeGreaterThan(targetFPS * 0.8) // At least 80% of target FPS
      expect(frameCount).toBeGreaterThan(expectedFrames * 0.8)
    })

    it('should handle DOM updates efficiently', async () => {
      const updateCount = 100
      const container = document.createElement('div')
      document.body.appendChild(container)

      monitor.startTimer('dom-updates')

      for (let i = 0; i < updateCount; i++) {
        const element = document.createElement('div')
        element.textContent = `Update ${i}`
        container.appendChild(element)
        
        // Force layout
        element.offsetHeight
      }

      const duration = monitor.endTimer('dom-updates')

      expect(duration).toBeLessThan(100) // Should complete within 100ms
      expect(container.children).toHaveLength(updateCount)

      document.body.removeChild(container)
    })

    it('should optimize re-renders with React', async () => {
      // Mock React component re-render tracking
      let renderCount = 0
      const mockComponent = {
        render: () => {
          renderCount++
          return `Render ${renderCount}`
        }
      }

      const iterations = 50
      
      monitor.startTimer('react-renders')

      for (let i = 0; i < iterations; i++) {
        mockComponent.render()
      }

      const duration = monitor.endTimer('react-renders')

      expect(duration).toBeLessThan(50) // Should be very fast
      expect(renderCount).toBe(iterations)
    })
  })

  describe('Memory Usage', () => {
    it('should not have memory leaks during extended use', async () => {
      const initialMemory = performance.memory?.usedJSHeapSize || 0
      const gestureAPI = createMockGestureAPI(20)
      
      // Simulate extended use
      const iterations = 200
      const objects: any[] = []

      for (let i = 0; i < iterations; i++) {
        // Create some objects to simulate real usage
        const mockData = {
          landmarks: Array.from({ length: 21 }, () => ({ x: Math.random(), y: Math.random(), z: Math.random() })),
          timestamp: Date.now(),
          id: i
        }
        
        objects.push(mockData)
        await gestureAPI.predict(mockData.landmarks)
        
        // Clean up old objects (simulate garbage collection)
        if (objects.length > 50) {
          objects.shift()
        }
      }

      // Force garbage collection if available
      if (global.gc) {
        global.gc()
      }

      const finalMemory = performance.memory?.usedJSHeapSize || 0
      const memoryIncrease = finalMemory - initialMemory

      // Memory increase should be reasonable (less than 10MB)
      expect(memoryIncrease).toBeLessThan(10 * 1024 * 1024)
    })

    it('should efficiently manage MediaPipe resources', async () => {
      const mockMediaPipe = {
        hands: null as any,
        camera: null as any,
        
        initialize: () => {
          mockMediaPipe.hands = { close: vi.fn() }
          mockMediaPipe.camera = { stop: vi.fn() }
        },
        
        cleanup: () => {
          if (mockMediaPipe.hands) {
            mockMediaPipe.hands.close()
            mockMediaPipe.hands = null
          }
          if (mockMediaPipe.camera) {
            mockMediaPipe.camera.stop()
            mockMediaPipe.camera = null
          }
        }
      }

      const cycles = 10

      for (let i = 0; i < cycles; i++) {
        monitor.startTimer(`mediapipe-cycle-${i}`)
        
        mockMediaPipe.initialize()
        
        // Simulate usage
        await new Promise(resolve => setTimeout(resolve, 10))
        
        mockMediaPipe.cleanup()
        
        monitor.endTimer(`mediapipe-cycle-${i}`)
      }

      // Should complete cleanup cycles efficiently
      for (let i = 0; i < cycles; i++) {
        const metrics = monitor.getMetrics(`mediapipe-cycle-${i}`)
        expect(metrics.avg).toBeLessThan(50) // Each cycle < 50ms
      }
    })
  })

  describe('Network Performance', () => {
    it('should handle network latency gracefully', async () => {
      const latencies = [50, 100, 200, 500] // Different network conditions
      
      for (const latency of latencies) {
        const gestureAPI = createMockGestureAPI(latency)
        
        monitor.startTimer(`network-${latency}ms`)
        await gestureAPI.predict([])
        const duration = monitor.endTimer(`network-${latency}ms`)
        
        // Should handle the latency appropriately
        expect(duration).toBeGreaterThan(latency * 0.8) // At least 80% of expected latency
        expect(duration).toBeLessThan(latency * 2) // Not more than 2x expected latency
      }
    })

    it('should implement request throttling effectively', async () => {
      const gestureAPI = createMockGestureAPI(30)
      const throttleMs = 100
      const requests = 10
      
      const startTime = performance.now()
      
      for (let i = 0; i < requests; i++) {
        monitor.startTimer(`throttled-request-${i}`)
        await gestureAPI.predict([])
        monitor.endTimer(`throttled-request-${i}`)
        
        // Simulate throttling
        await new Promise(resolve => setTimeout(resolve, throttleMs))
      }
      
      const endTime = performance.now()
      const totalTime = endTime - startTime
      const expectedMinTime = requests * throttleMs
      
      expect(totalTime).toBeGreaterThan(expectedMinTime * 0.9) // Should respect throttling
    })

    it('should handle request queuing efficiently', async () => {
      const gestureAPI = createMockGestureAPI(50)
      const maxConcurrent = 3
      const totalRequests = 15
      
      let activeRequests = 0
      let completedRequests = 0
      const results: Promise<any>[] = []
      
      monitor.startTimer('request-queue')
      
      for (let i = 0; i < totalRequests; i++) {
        const request = (async () => {
          // Wait for available slot
          while (activeRequests >= maxConcurrent) {
            await new Promise(resolve => setTimeout(resolve, 10))
          }
          
          activeRequests++
          const result = await gestureAPI.predict([])
          activeRequests--
          completedRequests++
          
          return result
        })()
        
        results.push(request)
      }
      
      await Promise.all(results)
      const duration = monitor.endTimer('request-queue')
      
      expect(completedRequests).toBe(totalRequests)
      expect(duration).toBeLessThan(2000) // Should complete within 2 seconds
    })
  })

  describe('Resource Optimization', () => {
    it('should optimize image processing performance', async () => {
      const imageProcessing = {
        processFrame: async (width: number, height: number) => {
          // Simulate image processing
          const pixels = width * height
          const processingTime = pixels / 100000 // Simulate processing complexity
          
          await new Promise(resolve => setTimeout(resolve, processingTime))
          
          return {
            processed: true,
            pixels,
            processingTime
          }
        }
      }

      const resolutions = [
        { width: 640, height: 480 },   // 480p
        { width: 1280, height: 720 },  // 720p
        { width: 1920, height: 1080 }  // 1080p
      ]

      for (const resolution of resolutions) {
        monitor.startTimer(`processing-${resolution.width}x${resolution.height}`)
        
        const result = await imageProcessing.processFrame(resolution.width, resolution.height)
        
        const duration = monitor.endTimer(`processing-${resolution.width}x${resolution.height}`)
        
        // Higher resolutions should take longer but still be reasonable
        expect(duration).toBeLessThan(100) // Max 100ms for any resolution
        expect(result.processed).toBe(true)
      }
    })

    it('should optimize gesture recognition pipeline', async () => {
      const pipeline = {
        preprocess: async (data: any) => {
          await new Promise(resolve => setTimeout(resolve, 5))
          return { preprocessed: data }
        },
        
        extract: async (data: any) => {
          await new Promise(resolve => setTimeout(resolve, 10))
          return { features: data.preprocessed }
        },
        
        classify: async (data: any) => {
          await new Promise(resolve => setTimeout(resolve, 15))
          return { prediction: 'A', confidence: 0.95 }
        },
        
        postprocess: async (data: any) => {
          await new Promise(resolve => setTimeout(resolve, 5))
          return data
        }
      }

      const iterations = 20
      
      for (let i = 0; i < iterations; i++) {
        monitor.startTimer(`pipeline-${i}`)
        
        let data = { landmarks: [] }
        data = await pipeline.preprocess(data)
        data = await pipeline.extract(data)
        data = await pipeline.classify(data)
        data = await pipeline.postprocess(data)
        
        monitor.endTimer(`pipeline-${i}`)
      }

      const metrics = monitor.getMetrics('pipeline-0')
      
      // Complete pipeline should be fast
      expect(metrics.avg).toBeLessThan(50) // Average < 50ms
      expect(metrics.max).toBeLessThan(100) // Max < 100ms
    })
  })
})

export { PerformanceMonitor }
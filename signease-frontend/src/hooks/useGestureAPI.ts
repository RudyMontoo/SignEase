/**
 * Gesture API Hook
 * ================
 * 
 * React hook for gesture prediction with caching, error handling, and performance monitoring
 */

import { useState, useCallback, useRef, useEffect } from 'react'

export interface GesturePrediction {
  gesture: string
  confidence: number
  alternatives: Array<{
    gesture: string
    confidence: number
  }>
  inference_time_ms: number
  request_id?: number
  timestamp?: string
  cached?: boolean
}

export interface APIError {
  error: string
  message: string
  status?: number
}

export interface GestureAPIState {
  currentGesture: string | null
  confidence: number
  alternatives: Array<{ gesture: string; confidence: number }>
  isLoading: boolean
  error: string | null
  isConnected: boolean
  lastPredictionTime: number
  totalPredictions: number
  averageResponseTime: number
}

export interface GestureAPIActions {
  predictGesture: (landmarks: number[], handedness?: string) => Promise<void>
  clearError: () => void
  resetStats: () => void
  testConnection: () => Promise<boolean>
}

export interface UseGestureAPIOptions {
  enableCaching?: boolean
  maxCacheSize?: number
  cacheTTL?: number
  throttleMs?: number
  autoConnect?: boolean
}

interface CacheEntry {
  result: GesturePrediction
  timestamp: number
}

export const useGestureAPI = (options: UseGestureAPIOptions = {}): GestureAPIState & GestureAPIActions => {
  const {
    enableCaching = true,
    maxCacheSize = 100,
    cacheTTL = 5000, // 5 seconds
    throttleMs = 100, // 100ms throttle
    autoConnect = true
  } = options

  // State
  const [state, setState] = useState<GestureAPIState>({
    currentGesture: null,
    confidence: 0,
    alternatives: [],
    isLoading: false,
    error: null,
    isConnected: false,
    lastPredictionTime: 0,
    totalPredictions: 0,
    averageResponseTime: 0
  })

  // Refs for performance tracking
  const responseTimes = useRef<number[]>([])
  const cache = useRef<Map<string, CacheEntry>>(new Map())
  const lastRequestTime = useRef<number>(0)
  const requestQueue = useRef<Promise<void> | null>(null)

  // Generate cache key from landmarks
  const generateCacheKey = useCallback((landmarks: number[]): string => {
    // Round landmarks to 3 decimal places to improve cache hits
    const rounded = landmarks.map(l => Math.round(l * 1000) / 1000)
    return JSON.stringify(rounded)
  }, [])

  // Clean expired cache entries
  const cleanCache = useCallback(() => {
    const now = Date.now()
    const entries = Array.from(cache.current.entries())
    
    entries.forEach(([key, entry]) => {
      if (now - entry.timestamp > cacheTTL) {
        cache.current.delete(key)
      }
    })

    // Limit cache size
    if (cache.current.size > maxCacheSize) {
      const sortedEntries = entries
        .sort((a, b) => a[1].timestamp - b[1].timestamp)
        .slice(0, cache.current.size - maxCacheSize)
      
      sortedEntries.forEach(([key]) => {
        cache.current.delete(key)
      })
    }
  }, [cacheTTL, maxCacheSize])

  // Update performance stats
  const updateStats = useCallback((responseTime: number) => {
    responseTimes.current.push(responseTime)
    if (responseTimes.current.length > 50) {
      responseTimes.current.shift()
    }

    const avgResponseTime = responseTimes.current.reduce((a, b) => a + b, 0) / responseTimes.current.length

    setState(prev => ({
      ...prev,
      totalPredictions: prev.totalPredictions + 1,
      averageResponseTime: avgResponseTime,
      lastPredictionTime: Date.now()
    }))
  }, [])

  // Test connection to backend
  const testConnection = useCallback(async (): Promise<boolean> => {
    try {
      const response = await fetch('http://localhost:5000/health')
      const isConnected = response.ok
      setState(prev => ({ ...prev, isConnected }))
      return isConnected
    } catch {
      setState(prev => ({ ...prev, isConnected: false }))
      return false
    }
  }, [])

  // Main prediction function
  const predictGesture = useCallback(async (
    landmarks: number[],
    handedness: string = 'Right'
  ): Promise<void> => {
    // Throttling
    const now = Date.now()
    if (now - lastRequestTime.current < throttleMs) {
      return
    }
    lastRequestTime.current = now

    // Wait for any pending request to complete
    if (requestQueue.current) {
      await requestQueue.current
    }

    // Create new request
    const request = async () => {
      try {
        setState(prev => ({ ...prev, isLoading: true, error: null }))

        // Check cache first
        if (enableCaching) {
          const cacheKey = generateCacheKey(landmarks)
          const cached = cache.current.get(cacheKey)
          
          if (cached && (now - cached.timestamp) < cacheTTL) {
            setState(prev => ({
              ...prev,
              currentGesture: cached.result.gesture,
              confidence: cached.result.confidence,
              alternatives: cached.result.alternatives,
              isLoading: false,
              isConnected: true
            }))
            return
          }
        }

        // Make API request
        const startTime = Date.now()
        const response = await fetch('http://localhost:5000/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            landmarks,
            handedness,
            alternatives: 3
          })
        })

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }

        const result: GesturePrediction = await response.json()
        const responseTime = Date.now() - startTime

        // Cache result
        if (enableCaching) {
          const cacheKey = generateCacheKey(landmarks)
          cache.current.set(cacheKey, {
            result,
            timestamp: now
          })
          cleanCache()
        }

        // Update state
        setState(prev => ({
          ...prev,
          currentGesture: result.gesture,
          confidence: result.confidence,
          alternatives: result.alternatives,
          isLoading: false,
          error: null,
          isConnected: true
        }))

        updateStats(responseTime)

      } catch (error) {
        console.error('Gesture prediction error:', error)
        
        let errorMessage = 'Unknown error occurred'
        let isConnected = true

        if (error instanceof Error) {
          errorMessage = error.message
          if (error.message.includes('fetch') || error.message.includes('NetworkError')) {
            isConnected = false
            errorMessage = 'Network error - backend server may be down'
          }
        }

        setState(prev => ({
          ...prev,
          isLoading: false,
          error: errorMessage,
          isConnected
        }))
      }
    }

    requestQueue.current = request()
    await requestQueue.current
    requestQueue.current = null
  }, [enableCaching, generateCacheKey, cacheTTL, cleanCache, updateStats, throttleMs])

  // Clear error
  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null }))
  }, [])

  // Reset stats
  const resetStats = useCallback(() => {
    responseTimes.current = []
    cache.current.clear()
    setState(prev => ({
      ...prev,
      totalPredictions: 0,
      averageResponseTime: 0,
      lastPredictionTime: 0
    }))
  }, [])

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect) {
      testConnection()
    }
  }, [autoConnect, testConnection])

  // Periodic connection check
  useEffect(() => {
    if (!autoConnect) return

    const interval = setInterval(() => {
      if (!state.isConnected) {
        testConnection()
      }
    }, 10000) // Check every 10 seconds

    return () => clearInterval(interval)
  }, [autoConnect, state.isConnected, testConnection])

  return {
    ...state,
    predictGesture,
    clearError,
    resetStats,
    testConnection
  }
}
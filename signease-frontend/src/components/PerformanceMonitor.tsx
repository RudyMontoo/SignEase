/**
 * Performance Monitor Component
 * ============================
 * 
 * Real-time performance monitoring and optimization controls
 */

import React, { useState, useEffect } from 'react'
import { useTheme } from '../hooks/useTheme'
import { usePerformanceMonitoring, type PerformanceMetrics } from '../utils/performanceOptimizer'
import { useGPUMemoryMonitoring } from '../utils/gpuMemoryProfiler'
import { useRenderingOptimization } from '../utils/renderingOptimizer'
import { useRequestBatcher } from '../utils/requestBatcher'
import Card, { CardHeader, CardTitle, CardContent } from './ui/Card'
import Button from './ui/Button'
import Badge from './ui/Badge'

export interface PerformanceMonitorProps {
  className?: string
  compact?: boolean
  showControls?: boolean
}

export const PerformanceMonitor: React.FC<PerformanceMonitorProps> = ({
  className = '',
  compact = false,
  showControls = true
}) => {
  const { isDark } = useTheme()
  const {
    metrics,
    getOptimizationSuggestions,
    forceGarbageCollection,
    clearCaches,
    exportReport
  } = usePerformanceMonitoring()
  
  const gpuMonitoring = useGPUMemoryMonitoring()
  const renderingOptimization = useRenderingOptimization()
  const requestBatching = useRequestBatcher()
  
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [isExpanded, setIsExpanded] = useState(!compact)

  useEffect(() => {
    const interval = setInterval(() => {
      setSuggestions(getOptimizationSuggestions())
    }, 5000)
    
    return () => clearInterval(interval)
  }, [getOptimizationSuggestions])

  const getStatusColor = (value: number, thresholds: { good: number; warning: number }) => {
    if (value <= thresholds.good) return 'success'
    if (value <= thresholds.warning) return 'warning'
    return 'error'
  }

  const getFPSStatus = () => getStatusColor(30 - metrics.fps, { good: 5, warning: 10 })
  const getMemoryStatus = () => getStatusColor(metrics.memoryUsage.percentage, { good: 50, warning: 75 })
  const getLatencyStatus = () => getStatusColor(metrics.apiLatency.average, { good: 100, warning: 200 })

  const handleExportReport = () => {
    const report = exportReport()
    const blob = new Blob([report], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `performance-report-${new Date().toISOString().slice(0, 19)}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  if (compact && !isExpanded) {
    return (
      <div className={`flex items-center space-x-2 ${className}`}>
        <Badge variant={getFPSStatus()} size="sm">
          {metrics.fps} FPS
        </Badge>
        <Badge variant={getMemoryStatus()} size="sm">
          {metrics.memoryUsage.percentage}% RAM
        </Badge>
        <Badge variant={getLatencyStatus()} size="sm">
          {Math.round(metrics.apiLatency.average)}ms API
        </Badge>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setIsExpanded(true)}
          className="text-xs"
        >
          📊
        </Button>
      </div>
    )
  }

  return (
    <Card className={`${className}`} padding="sm">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">Performance Monitor</CardTitle>
          <div className="flex items-center space-x-2">
            {compact && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsExpanded(false)}
                className="text-xs"
              >
                ✕
              </Button>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Core Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className={`p-3 rounded-lg ${
            isDark ? 'bg-gray-700' : 'bg-gray-100'
          }`}>
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium">FPS</span>
              <Badge variant={getFPSStatus()} size="sm">
                {metrics.fps}
              </Badge>
            </div>
            <div className="mt-1">
              <div className={`text-xs ${
                isDark ? 'text-gray-400' : 'text-gray-600'
              }`}>
                Target: 30
              </div>
            </div>
          </div>

          <div className={`p-3 rounded-lg ${
            isDark ? 'bg-gray-700' : 'bg-gray-100'
          }`}>
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium">Memory</span>
              <Badge variant={getMemoryStatus()} size="sm">
                {metrics.memoryUsage.percentage}%
              </Badge>
            </div>
            <div className="mt-1">
              <div className={`text-xs ${
                isDark ? 'text-gray-400' : 'text-gray-600'
              }`}>
                {metrics.memoryUsage.used}MB / {metrics.memoryUsage.total}MB
              </div>
            </div>
          </div>

          <div className={`p-3 rounded-lg ${
            isDark ? 'bg-gray-700' : 'bg-gray-100'
          }`}>
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium">API Latency</span>
              <Badge variant={getLatencyStatus()} size="sm">
                {Math.round(metrics.apiLatency.average)}ms
              </Badge>
            </div>
            <div className="mt-1">
              <div className={`text-xs ${
                isDark ? 'text-gray-400' : 'text-gray-600'
              }`}>
                Min: {Math.round(metrics.apiLatency.min)}ms | Max: {Math.round(metrics.apiLatency.max)}ms
              </div>
            </div>
          </div>

          <div className={`p-3 rounded-lg ${
            isDark ? 'bg-gray-700' : 'bg-gray-100'
          }`}>
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium">Cache Hit</span>
              <Badge variant={metrics.cacheHitRate > 0.5 ? 'success' : 'warning'} size="sm">
                {Math.round(metrics.cacheHitRate * 100)}%
              </Badge>
            </div>
            <div className="mt-1">
              <div className={`text-xs ${
                isDark ? 'text-gray-400' : 'text-gray-600'
              }`}>
                Connections: {metrics.activeConnections}
              </div>
            </div>
          </div>
        </div>

        {/* Detailed Metrics */}
        {!compact && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Render Performance */}
            <div className={`p-3 rounded-lg border ${
              isDark ? 'border-gray-600 bg-gray-800' : 'border-gray-200 bg-white'
            }`}>
              <h4 className="text-sm font-medium mb-2">Render Performance</h4>
              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span>Average Render Time:</span>
                  <span>{metrics.renderTime.average.toFixed(1)}ms</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span>Recent Renders:</span>
                  <span>{metrics.renderTime.recent.length}</span>
                </div>
                {metrics.renderTime.recent.length > 0 && (
                  <div className="w-full h-8 flex items-end space-x-1">
                    {metrics.renderTime.recent.slice(-10).map((time, i) => (
                      <div
                        key={i}
                        className={`flex-1 rounded-t ${
                          time > 16 ? 'bg-red-400' : time > 8 ? 'bg-yellow-400' : 'bg-green-400'
                        }`}
                        style={{ height: `${Math.min((time / 32) * 100, 100)}%` }}
                        title={`${time.toFixed(1)}ms`}
                      />
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* API Performance */}
            <div className={`p-3 rounded-lg border ${
              isDark ? 'border-gray-600 bg-gray-800' : 'border-gray-200 bg-white'
            }`}>
              <h4 className="text-sm font-medium mb-2">API Performance</h4>
              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span>Recent Requests:</span>
                  <span>{metrics.apiLatency.recent.length}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span>Active Connections:</span>
                  <span>{metrics.activeConnections}</span>
                </div>
                {metrics.apiLatency.recent.length > 0 && (
                  <div className="w-full h-8 flex items-end space-x-1">
                    {metrics.apiLatency.recent.slice(-10).map((latency, i) => (
                      <div
                        key={i}
                        className={`flex-1 rounded-t ${
                          latency > 200 ? 'bg-red-400' : latency > 100 ? 'bg-yellow-400' : 'bg-green-400'
                        }`}
                        style={{ height: `${Math.min((latency / 400) * 100, 100)}%` }}
                        title={`${latency.toFixed(1)}ms`}
                      />
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Optimization Suggestions */}
        {suggestions.length > 0 && (
          <div className={`p-3 rounded-lg border-l-4 border-yellow-400 ${
            isDark ? 'bg-yellow-900/20' : 'bg-yellow-50'
          }`}>
            <h4 className="text-sm font-medium mb-2 flex items-center">
              <span className="mr-2">💡</span>
              Optimization Suggestions
            </h4>
            <ul className="space-y-1">
              {suggestions.map((suggestion, i) => (
                <li key={i} className="text-xs flex items-start">
                  <span className="mr-2">•</span>
                  <span>{suggestion}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Controls */}
        {showControls && (
          <div className="flex flex-wrap gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={forceGarbageCollection}
              className="text-xs"
            >
              🗑️ Force GC
            </Button>
            
            <Button
              variant="outline"
              size="sm"
              onClick={clearCaches}
              className="text-xs"
            >
              🧹 Clear Cache
            </Button>
            
            <Button
              variant="outline"
              size="sm"
              onClick={handleExportReport}
              className="text-xs"
            >
              📊 Export Report
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default PerformanceMonitor
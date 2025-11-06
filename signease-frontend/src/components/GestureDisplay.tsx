/**
 * Gesture Display Component
 * =========================
 * 
 * Enhanced display for current gesture with confidence indicators and visual feedback
 */

import React, { useState, useEffect } from 'react'

export interface GestureDisplayProps {
  currentGesture: string | null
  confidence: number
  alternatives?: Array<{ gesture: string; confidence: number }>
  isLoading?: boolean
  isConnected?: boolean
  className?: string
}

export const GestureDisplay: React.FC<GestureDisplayProps> = ({
  currentGesture,
  confidence,
  alternatives = [],
  isLoading = false,
  isConnected = true,
  className = ''
}) => {
  const [displayGesture, setDisplayGesture] = useState<string>('-')
  const [animationKey, setAnimationKey] = useState<number>(0)
  const [confidenceLevel, setConfidenceLevel] = useState<'low' | 'medium' | 'high'>('low')

  // Update display gesture with animation
  useEffect(() => {
    if (currentGesture && currentGesture !== displayGesture) {
      setDisplayGesture(currentGesture)
      setAnimationKey(prev => prev + 1)
    } else if (!currentGesture) {
      setDisplayGesture('-')
    }
  }, [currentGesture, displayGesture])

  // Update confidence level
  useEffect(() => {
    if (confidence >= 0.8) {
      setConfidenceLevel('high')
    } else if (confidence >= 0.6) {
      setConfidenceLevel('medium')
    } else {
      setConfidenceLevel('low')
    }
  }, [confidence])

  // Get confidence color
  const getConfidenceColor = () => {
    switch (confidenceLevel) {
      case 'high':
        return 'text-green-600 bg-green-100'
      case 'medium':
        return 'text-yellow-600 bg-yellow-100'
      case 'low':
        return 'text-red-600 bg-red-100'
      default:
        return 'text-gray-600 bg-gray-100'
    }
  }

  // Get gesture display color
  const getGestureColor = () => {
    if (!isConnected) return 'text-gray-400'
    if (isLoading) return 'text-blue-600'
    
    switch (confidenceLevel) {
      case 'high':
        return 'text-green-600'
      case 'medium':
        return 'text-yellow-600'
      case 'low':
        return 'text-red-600'
      default:
        return 'text-gray-600'
    }
  }

  // Format confidence percentage
  const formatConfidence = (conf: number) => {
    return Math.round(conf * 100)
  }

  // Get status message
  const getStatusMessage = () => {
    if (!isConnected) return 'Disconnected'
    if (isLoading) return 'Processing...'
    if (!currentGesture) return 'Show a gesture'
    if (confidenceLevel === 'low') return 'Low confidence'
    return 'Detected'
  }

  return (
    <div className={`bg-white rounded-xl shadow-lg p-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800">Current Gesture</h3>
        <div className="flex items-center space-x-2">
          {/* Connection indicator */}
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-xs text-gray-500">
            {isConnected ? 'Connected' : 'Offline'}
          </span>
        </div>
      </div>

      {/* Main gesture display */}
      <div className="text-center mb-6">
        <div className="relative">
          {/* Loading spinner overlay */}
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-75 rounded-lg">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            </div>
          )}
          
          {/* Gesture letter */}
          <div 
            key={animationKey}
            className={`text-8xl font-bold mb-2 transition-all duration-300 transform ${getGestureColor()} ${
              animationKey > 0 ? 'animate-pulse scale-110' : ''
            }`}
            style={{
              fontFamily: 'monospace',
              textShadow: confidenceLevel === 'high' ? '0 0 10px rgba(34, 197, 94, 0.3)' : 'none'
            }}
          >
            {displayGesture}
          </div>
          
          {/* Status message */}
          <p className="text-sm text-gray-500 mb-3">
            {getStatusMessage()}
          </p>
        </div>

        {/* Confidence indicator */}
        {currentGesture && (
          <div className="space-y-2">
            {/* Confidence bar */}
            <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
              <div 
                className={`h-full transition-all duration-500 ease-out ${
                  confidenceLevel === 'high' ? 'bg-green-500' :
                  confidenceLevel === 'medium' ? 'bg-yellow-500' : 'bg-red-500'
                }`}
                style={{ width: `${Math.max(confidence * 100, 5)}%` }}
              />
            </div>
            
            {/* Confidence percentage */}
            <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getConfidenceColor()}`}>
              <span className="mr-1">Confidence:</span>
              <span className="font-bold">{formatConfidence(confidence)}%</span>
            </div>
          </div>
        )}
      </div>

      {/* Alternative predictions */}
      {alternatives.length > 0 && currentGesture && (
        <div className="border-t pt-4">
          <h4 className="text-sm font-medium text-gray-700 mb-3">Alternative Predictions</h4>
          <div className="space-y-2">
            {alternatives.slice(0, 3).map((alt, index) => (
              <div key={index} className="flex items-center justify-between text-sm">
                <div className="flex items-center space-x-2">
                  <span className="w-6 h-6 bg-gray-100 rounded-full flex items-center justify-center text-xs font-medium text-gray-600">
                    {index + 2}
                  </span>
                  <span className="font-medium text-gray-700">{alt.gesture}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-16 bg-gray-200 rounded-full h-1.5">
                    <div 
                      className="bg-gray-400 h-1.5 rounded-full transition-all duration-300"
                      style={{ width: `${alt.confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-gray-500 text-xs w-8 text-right">
                    {formatConfidence(alt.confidence)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Gesture guide hint */}
      {!currentGesture && isConnected && !isLoading && (
        <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
          <p className="text-sm text-blue-700 text-center">
            💡 <strong>Tip:</strong> Hold your hand steady in front of the camera to detect gestures
          </p>
        </div>
      )}

      {/* Error state */}
      {!isConnected && (
        <div className="mt-4 p-3 bg-red-50 rounded-lg border border-red-200">
          <p className="text-sm text-red-700 text-center">
            ⚠️ <strong>Connection Lost:</strong> Check if the backend server is running
          </p>
        </div>
      )}
    </div>
  )
}

export default GestureDisplay
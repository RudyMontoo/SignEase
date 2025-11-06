/**
 * Animated Gesture Display Component
 * ==================================
 * 
 * Enhanced gesture display with smooth animations and micro-interactions
 */

import React, { useEffect, useState, useRef } from 'react'
import Card, { CardHeader, CardTitle, CardContent } from './ui/Card'
import Badge from './ui/Badge'
import '../styles/animations.css'

interface AnimatedGestureDisplayProps {
  currentGesture: string | null
  confidence: number
  alternatives: Array<{ gesture: string; confidence: number }>
  isLoading: boolean
  isConnected: boolean
}

const AnimatedGestureDisplay: React.FC<AnimatedGestureDisplayProps> = ({
  currentGesture,
  confidence,
  alternatives,
  isLoading,
  isConnected
}) => {
  const [previousGesture, setPreviousGesture] = useState<string | null>(null)
  const [animationClass, setAnimationClass] = useState('')
  const [confidenceAnimation, setConfidenceAnimation] = useState('')
  const gestureRef = useRef<HTMLDivElement>(null)
  const confidenceRef = useRef<HTMLDivElement>(null)

  // Animate gesture changes
  useEffect(() => {
    if (currentGesture && currentGesture !== previousGesture) {
      setAnimationClass('gesture-detected scale-in')
      
      // Trigger success animation for high confidence
      if (confidence > 0.8) {
        setTimeout(() => {
          setAnimationClass('success-animation')
        }, 300)
      }
      
      // Clear animation after completion
      setTimeout(() => {
        setAnimationClass('')
      }, 600)
      
      setPreviousGesture(currentGesture)
    }
  }, [currentGesture, previousGesture, confidence])

  // Animate confidence changes
  useEffect(() => {
    if (confidence > 0) {
      setConfidenceAnimation('confidence-bar')
      setTimeout(() => {
        setConfidenceAnimation('')
      }, 300)
    }
  }, [confidence])

  // Error animation for low confidence
  useEffect(() => {
    if (currentGesture && confidence < 0.5) {
      setAnimationClass('error-animation')
      setTimeout(() => {
        setAnimationClass('')
      }, 500)
    }
  }, [currentGesture, confidence])

  const getConfidenceColor = (conf: number) => {
    if (conf >= 0.8) return 'bg-green-500'
    if (conf >= 0.6) return 'bg-yellow-500'
    return 'bg-red-500'
  }

  const getConfidenceTextColor = (conf: number) => {
    if (conf >= 0.8) return 'text-green-600'
    if (conf >= 0.6) return 'text-yellow-600'
    return 'text-red-600'
  }

  return (
    <Card padding="lg" shadow="lg" className="card-hover transition-smooth">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <span>Gesture Recognition</span>
          {isConnected ? (
            <div className="relative">
              <div className="w-3 h-3 bg-green-500 rounded-full connection-indicator" />
            </div>
          ) : (
            <div className="w-3 h-3 bg-red-500 rounded-full status-pulse" />
          )}
        </CardTitle>
      </CardHeader>
      
      <CardContent>
        <div className="space-y-6">
          {/* Current Gesture Display */}
          <div className="text-center">
            {isLoading ? (
              <div className="flex items-center justify-center space-x-2">
                <div className="loading-spinner w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full" />
                <span className="text-gray-500">Recognizing...</span>
              </div>
            ) : currentGesture ? (
              <div className="space-y-4">
                <div 
                  ref={gestureRef}
                  className={`text-8xl font-bold text-blue-600 ${animationClass}`}
                >
                  {currentGesture}
                </div>
                
                {/* Confidence Display */}
                <div className="space-y-2">
                  <div className="flex items-center justify-center space-x-2">
                    <span className="text-sm text-gray-500">Confidence:</span>
                    <span className={`font-bold ${getConfidenceTextColor(confidence)}`}>
                      {Math.round(confidence * 100)}%
                    </span>
                  </div>
                  
                  {/* Animated Confidence Bar */}
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 overflow-hidden">
                    <div 
                      ref={confidenceRef}
                      className={`h-full transition-all duration-300 ${getConfidenceColor(confidence)} ${confidenceAnimation}`}
                      style={{ 
                        width: `${confidence * 100}%`,
                        '--confidence-width': `${confidence * 100}%`
                      } as React.CSSProperties}
                    />
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-gray-400 text-center py-8">
                <div className="text-4xl mb-2">👋</div>
                <p>Show your hand to start recognition</p>
              </div>
            )}
          </div>

          {/* Alternative Predictions */}
          {alternatives.length > 0 && (
            <div className="space-y-2 fade-in">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Alternative Predictions:
              </h4>
              <div className="flex flex-wrap gap-2">
                {alternatives.map((alt, index) => (
                  <div 
                    key={index}
                    className="slide-in-right"
                    style={{ animationDelay: `${index * 0.1}s` }}
                  >
                    <Badge 
                      variant="secondary"
                      className="transition-smooth hover:scale-105"
                    >
                      {alt.gesture} ({Math.round(alt.confidence * 100)}%)
                    </Badge>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Status Messages */}
          <div className="text-center">
            {!isConnected && (
              <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-3 error-animation">
                <p className="text-red-600 dark:text-red-400 text-sm">
                  ⚠️ Backend connection lost. Trying to reconnect...
                </p>
              </div>
            )}
            
            {isConnected && !currentGesture && !isLoading && (
              <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-3 fade-in">
                <p className="text-blue-600 dark:text-blue-400 text-sm">
                  💡 Position your hand clearly in the camera view
                </p>
              </div>
            )}
            
            {currentGesture && confidence > 0.9 && (
              <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-3 success-animation">
                <p className="text-green-600 dark:text-green-400 text-sm">
                  ✅ Excellent recognition! Keep it up!
                </p>
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export default AnimatedGestureDisplay
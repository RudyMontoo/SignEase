/**
 * AR Overlay Component
 * ====================
 * 
 * Augmented reality text overlay that displays gesture predictions floating near the user's hand
 */

import React, { useRef } from 'react'
import { useAROverlay, type AROverlayOptions } from '../hooks/useAROverlay'
import { useTheme } from '../hooks/useTheme'

export interface AROverlayProps {
  landmarks: number[] | null
  currentGesture: string | null
  confidence: number
  containerWidth: number
  containerHeight: number
  enabled?: boolean
  options?: Partial<AROverlayOptions>
  className?: string
}

export const AROverlay: React.FC<AROverlayProps> = ({
  landmarks,
  currentGesture,
  confidence,
  containerWidth,
  containerHeight,
  enabled = true,
  options = {},
  className = ''
}) => {
  const { isDark } = useTheme()
  const containerRef = useRef<HTMLDivElement>(null)
  
  const { overlayState, isVisible } = useAROverlay({
    landmarks: enabled ? landmarks : null,
    currentGesture: enabled ? currentGesture : null,
    confidence,
    containerWidth,
    containerHeight,
    options
  })

  // Don't render if not enabled or not visible
  if (!enabled || !isVisible) {
    return null
  }

  const { position, text, confidence: overlayConfidence, scale, opacity } = overlayState

  // Calculate confidence color
  const getConfidenceColor = () => {
    if (overlayConfidence >= 0.9) return isDark ? '#10b981' : '#059669' // green
    if (overlayConfidence >= 0.7) return isDark ? '#f59e0b' : '#d97706' // yellow
    return isDark ? '#ef4444' : '#dc2626' // red
  }

  // Calculate confidence glow intensity
  const getGlowIntensity = () => {
    return Math.max(0.3, overlayConfidence) * opacity
  }

  return (
    <div
      ref={containerRef}
      className={`absolute inset-0 pointer-events-none ${className}`}
      style={{
        width: containerWidth,
        height: containerHeight
      }}
    >
      {/* Main AR Text */}
      <div
        className="absolute transform -translate-x-1/2 -translate-y-1/2 transition-all duration-200 ease-out"
        style={{
          left: position.x + 60, // Center the text
          top: position.y + 20,
          transform: `translate(-50%, -50%) scale(${scale})`,
          opacity: opacity
        }}
      >
        {/* Glow Effect Background */}
        <div
          className="absolute inset-0 rounded-lg blur-md"
          style={{
            backgroundColor: getConfidenceColor(),
            opacity: getGlowIntensity() * 0.4,
            transform: 'scale(1.2)'
          }}
        />
        
        {/* Main Text Container */}
        <div
          className={`relative px-4 py-2 rounded-lg border-2 backdrop-blur-sm ${
            isDark 
              ? 'bg-gray-900/80 border-gray-600 text-white' 
              : 'bg-white/90 border-gray-300 text-gray-900'
          }`}
          style={{
            borderColor: getConfidenceColor(),
            boxShadow: `0 0 20px ${getConfidenceColor()}${Math.round(getGlowIntensity() * 255).toString(16).padStart(2, '0')}`
          }}
        >
          {/* Gesture Text */}
          <div className="text-2xl font-bold text-center min-w-[60px]">
            {text}
          </div>
          
          {/* Confidence Indicator */}
          <div className="flex items-center justify-center mt-1">
            <div className="w-8 h-1 bg-gray-300 dark:bg-gray-600 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-300"
                style={{
                  width: `${overlayConfidence * 100}%`,
                  backgroundColor: getConfidenceColor()
                }}
              />
            </div>
            <span className="ml-2 text-xs font-medium opacity-75">
              {Math.round(overlayConfidence * 100)}%
            </span>
          </div>
        </div>

        {/* Animated Particles */}
        <div className="absolute inset-0 pointer-events-none">
          {[...Array(3)].map((_, i) => (
            <div
              key={i}
              className="absolute w-1 h-1 rounded-full animate-pulse"
              style={{
                backgroundColor: getConfidenceColor(),
                left: `${20 + i * 30}%`,
                top: `${-10 - i * 5}px`,
                animationDelay: `${i * 200}ms`,
                animationDuration: '2s',
                opacity: opacity * 0.6
              }}
            />
          ))}
        </div>
      </div>

      {/* Connection Line (optional visual effect) */}
      {landmarks && landmarks.length >= 3 && (
        <svg
          className="absolute inset-0 pointer-events-none"
          width={containerWidth}
          height={containerHeight}
          style={{ opacity: opacity * 0.3 }}
        >
          <line
            x1={landmarks[0] * containerWidth} // Wrist X
            y1={landmarks[1] * containerHeight} // Wrist Y
            x2={position.x + 60}
            y2={position.y + 20}
            stroke={getConfidenceColor()}
            strokeWidth="2"
            strokeDasharray="5,5"
            className="animate-pulse"
          />
        </svg>
      )}
    </div>
  )
}

// AR Overlay Settings Component
export interface AROverlaySettingsProps {
  enabled: boolean
  onEnabledChange: (enabled: boolean) => void
  options: Partial<AROverlayOptions>
  onOptionsChange: (options: Partial<AROverlayOptions>) => void
  className?: string
}

export const AROverlaySettings: React.FC<AROverlaySettingsProps> = ({
  enabled,
  onEnabledChange,
  options,
  onOptionsChange,
  className = ''
}) => {
  const { isDark } = useTheme()

  const handleDisplayModeChange = (mode: AROverlayOptions['displayMode']) => {
    onOptionsChange({ ...options, displayMode: mode })
  }

  const handleConfidenceChange = (confidence: number) => {
    onOptionsChange({ ...options, minConfidence: confidence })
  }

  const handleSmoothingChange = (smoothing: boolean) => {
    onOptionsChange({ ...options, smoothingEnabled: smoothing })
  }

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Enable/Disable Toggle */}
      <div className="flex items-center justify-between">
        <span className={`font-medium ${isDark ? 'text-gray-200' : 'text-gray-800'}`}>
          AR Text Overlay
        </span>
        <button
          onClick={() => onEnabledChange(!enabled)}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
            enabled ? 'bg-blue-600' : isDark ? 'bg-gray-600' : 'bg-gray-200'
          }`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
              enabled ? 'translate-x-6' : 'translate-x-1'
            }`}
          />
        </button>
      </div>

      {enabled && (
        <>
          {/* Display Mode */}
          <div>
            <label className={`block text-sm font-medium mb-2 ${
              isDark ? 'text-gray-300' : 'text-gray-700'
            }`}>
              Display Mode
            </label>
            <div className="grid grid-cols-3 gap-2">
              {(['floating', 'fixed', 'following'] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => handleDisplayModeChange(mode)}
                  className={`px-3 py-2 text-sm rounded-lg transition-colors ${
                    options.displayMode === mode
                      ? 'bg-blue-600 text-white'
                      : isDark
                        ? 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {mode.charAt(0).toUpperCase() + mode.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {/* Confidence Threshold */}
          <div>
            <label className={`block text-sm font-medium mb-2 ${
              isDark ? 'text-gray-300' : 'text-gray-700'
            }`}>
              Min Confidence: {Math.round((options.minConfidence || 0.7) * 100)}%
            </label>
            <input
              type="range"
              min="0.5"
              max="0.95"
              step="0.05"
              value={options.minConfidence || 0.7}
              onChange={(e) => handleConfidenceChange(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
            />
          </div>

          {/* Smoothing */}
          <div className="flex items-center justify-between">
            <span className={`text-sm ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
              Position Smoothing
            </span>
            <button
              onClick={() => handleSmoothingChange(!(options.smoothingEnabled ?? true))}
              className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                (options.smoothingEnabled ?? true) ? 'bg-blue-600' : isDark ? 'bg-gray-600' : 'bg-gray-200'
              }`}
            >
              <span
                className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${
                  (options.smoothingEnabled ?? true) ? 'translate-x-5' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
        </>
      )}
    </div>
  )
}

export default AROverlay
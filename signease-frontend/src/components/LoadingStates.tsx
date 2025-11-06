/**
 * Loading States Component
 * ========================
 * 
 * Various loading animations and states for different contexts
 */

import React from 'react'
import '../styles/animations.css'

// Typing Indicator
export const TypingIndicator: React.FC<{ className?: string }> = ({ className = '' }) => (
  <div className={`flex items-center space-x-1 ${className}`}>
    <span className="text-gray-500 text-sm">Processing</span>
    <div className="typing-indicator flex space-x-1">
      <div className="dot w-1 h-1 bg-gray-400 rounded-full" />
      <div className="dot w-1 h-1 bg-gray-400 rounded-full" />
      <div className="dot w-1 h-1 bg-gray-400 rounded-full" />
    </div>
  </div>
)

// Spinner Loader
export const SpinnerLoader: React.FC<{ 
  size?: 'sm' | 'md' | 'lg'
  color?: string
  className?: string 
}> = ({ 
  size = 'md', 
  color = 'border-blue-500', 
  className = '' 
}) => {
  const sizeClasses = {
    sm: 'w-4 h-4 border-2',
    md: 'w-6 h-6 border-2',
    lg: 'w-8 h-8 border-3'
  }

  return (
    <div className={`
      loading-spinner 
      ${sizeClasses[size]} 
      ${color} 
      border-t-transparent 
      rounded-full 
      ${className}
    `} />
  )
}

// Progress Bar
export const ProgressBar: React.FC<{
  progress?: number
  indeterminate?: boolean
  className?: string
  color?: string
}> = ({ 
  progress = 0, 
  indeterminate = false, 
  className = '',
  color = 'bg-blue-500'
}) => (
  <div className={`w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 overflow-hidden ${className}`}>
    <div 
      className={`
        h-full 
        transition-all 
        duration-300 
        ${color} 
        ${indeterminate ? 'progress-indeterminate' : ''}
      `}
      style={indeterminate ? {} : { width: `${Math.min(100, Math.max(0, progress))}%` }}
    />
  </div>
)

// Skeleton Loader
export const SkeletonLoader: React.FC<{
  width?: string
  height?: string
  className?: string
}> = ({ 
  width = 'w-full', 
  height = 'h-4', 
  className = '' 
}) => (
  <div className={`
    ${width} 
    ${height} 
    bg-gray-200 
    dark:bg-gray-700 
    rounded 
    animate-pulse 
    ${className}
  `} />
)

// Pulse Loader
export const PulseLoader: React.FC<{
  count?: number
  size?: string
  color?: string
  className?: string
}> = ({ 
  count = 3, 
  size = 'w-2 h-2', 
  color = 'bg-blue-500', 
  className = '' 
}) => (
  <div className={`flex space-x-1 ${className}`}>
    {Array.from({ length: count }).map((_, index) => (
      <div 
        key={index}
        className={`
          ${size} 
          ${color} 
          rounded-full 
          animate-pulse
        `}
        style={{ animationDelay: `${index * 0.2}s` }}
      />
    ))}
  </div>
)

// Connection Status
export const ConnectionStatus: React.FC<{
  isConnected: boolean
  className?: string
}> = ({ isConnected, className = '' }) => (
  <div className={`flex items-center space-x-2 ${className}`}>
    <div className="relative">
      <div className={`
        w-3 h-3 
        rounded-full 
        ${isConnected ? 'bg-green-500 connection-indicator' : 'bg-red-500 status-pulse'}
      `} />
    </div>
    <span className={`text-sm ${
      isConnected ? 'text-green-600' : 'text-red-600'
    }`}>
      {isConnected ? 'Connected' : 'Disconnected'}
    </span>
  </div>
)

// Speech Wave Animation
export const SpeechWave: React.FC<{
  isActive: boolean
  className?: string
}> = ({ isActive, className = '' }) => (
  <div className={`flex items-center space-x-1 ${isActive ? 'speech-active' : ''} ${className}`}>
    <div className="wave-bar w-1 h-4 bg-current rounded-full" />
    <div className="wave-bar w-1 h-6 bg-current rounded-full" />
    <div className="wave-bar w-1 h-3 bg-current rounded-full" />
    <div className="wave-bar w-1 h-5 bg-current rounded-full" />
    <div className="wave-bar w-1 h-2 bg-current rounded-full" />
  </div>
)

// Loading Card
export const LoadingCard: React.FC<{
  lines?: number
  className?: string
}> = ({ 
  lines = 3, 
  className = '' 
}) => (
  <div className={`bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg ${className}`}>
    <div className="space-y-4">
      <SkeletonLoader width="w-1/2" height="h-6" />
      <div className="space-y-2">
        {Array.from({ length: lines }).map((_, index) => (
          <SkeletonLoader 
            key={index}
            width={index === lines - 1 ? 'w-3/4' : 'w-full'}
            height="h-4"
          />
        ))}
      </div>
    </div>
  </div>
)

// Gesture Loading State
export const GestureLoadingState: React.FC<{
  className?: string
}> = ({ className = '' }) => (
  <div className={`text-center py-8 ${className}`}>
    <div className="relative inline-block">
      <div className="text-6xl text-gray-300 dark:text-gray-600">👋</div>
      <div className="absolute inset-0 flex items-center justify-center">
        <SpinnerLoader size="lg" color="border-blue-500" />
      </div>
    </div>
    <div className="mt-4">
      <TypingIndicator />
    </div>
  </div>
)
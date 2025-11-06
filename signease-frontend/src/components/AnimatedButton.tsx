/**
 * Animated Button Component
 * =========================
 * 
 * Enhanced button with hover effects and loading animations
 */

import React, { useState } from 'react'
import Button from './ui/Button'
import '../styles/animations.css'

interface AnimatedButtonProps {
  onClick?: () => void | Promise<void>
  children: React.ReactNode
  variant?: 'primary' | 'secondary' | 'success' | 'error'
  size?: 'sm' | 'md' | 'lg'
  disabled?: boolean
  loading?: boolean
  className?: string
  fullWidth?: boolean
  icon?: React.ReactNode
  loadingText?: string
}

const AnimatedButton: React.FC<AnimatedButtonProps> = ({
  onClick,
  children,
  variant = 'primary',
  size = 'md',
  disabled = false,
  loading: externalLoading = false,
  className = '',
  fullWidth = false,
  icon,
  loadingText = 'Loading...'
}) => {
  const [internalLoading, setInternalLoading] = useState(false)
  const [animationClass, setAnimationClass] = useState('')

  const isLoading = externalLoading || internalLoading

  const handleClick = async () => {
    if (!onClick || isLoading || disabled) return

    try {
      setInternalLoading(true)
      setAnimationClass('scale-in')
      
      const result = onClick()
      
      if (result instanceof Promise) {
        await result
        // Success animation
        setAnimationClass('success-animation')
      } else {
        // Immediate success animation
        setAnimationClass('success-animation')
      }
    } catch (error) {
      // Error animation
      setAnimationClass('error-animation')
      console.error('Button action failed:', error)
    } finally {
      setInternalLoading(false)
      
      // Clear animation after completion
      setTimeout(() => {
        setAnimationClass('')
      }, 600)
    }
  }

  const LoadingSpinner = () => (
    <div className="loading-spinner w-4 h-4 border-2 border-current border-t-transparent rounded-full" />
  )

  const buttonContent = isLoading ? (
    <div className="flex items-center space-x-2">
      <LoadingSpinner />
      <span>{loadingText}</span>
    </div>
  ) : (
    <div className="flex items-center space-x-2">
      {icon && <span className="transition-transform group-hover:scale-110">{icon}</span>}
      <span>{children}</span>
    </div>
  )

  return (
    <Button
      onClick={handleClick}
      variant={variant}
      size={size}
      disabled={disabled || isLoading}
      className={`
        btn-hover-lift 
        transition-smooth 
        group 
        ${animationClass} 
        ${className}
        ${isLoading ? 'cursor-wait' : ''}
      `}
      fullWidth={fullWidth}
    >
      {buttonContent}
    </Button>
  )
}

export default AnimatedButton
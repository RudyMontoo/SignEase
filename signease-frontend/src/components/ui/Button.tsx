/**
 * Button Component
 * ================
 * 
 * Reusable button component with consistent styling and variants
 */

import React, { forwardRef } from 'react'
import { useTheme } from '../../hooks/useTheme'

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'success' | 'warning' | 'error' | 'ghost' | 'outline'
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl'
  fullWidth?: boolean
  loading?: boolean
  leftIcon?: React.ReactNode
  rightIcon?: React.ReactNode
  children: React.ReactNode
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(({
  variant = 'primary',
  size = 'md',
  fullWidth = false,
  loading = false,
  leftIcon,
  rightIcon,
  disabled,
  className = '',
  children,
  ...props
}, ref) => {
  const { isDark } = useTheme()

  // Size configurations
  const sizeClasses = {
    xs: 'px-2 py-1 text-xs',
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-base',
    lg: 'px-6 py-3 text-lg',
    xl: 'px-8 py-4 text-xl'
  }

  // Variant configurations
  const getVariantClasses = () => {
    const baseClasses = 'font-medium rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2'
    
    switch (variant) {
      case 'primary':
        return `${baseClasses} bg-blue-600 hover:bg-blue-700 text-white focus:ring-blue-500 disabled:bg-blue-300`
      
      case 'secondary':
        return `${baseClasses} ${
          isDark 
            ? 'bg-gray-700 hover:bg-gray-600 text-gray-100 focus:ring-gray-500' 
            : 'bg-gray-200 hover:bg-gray-300 text-gray-800 focus:ring-gray-500'
        } disabled:bg-gray-100 disabled:text-gray-400`
      
      case 'success':
        return `${baseClasses} bg-green-600 hover:bg-green-700 text-white focus:ring-green-500 disabled:bg-green-300`
      
      case 'warning':
        return `${baseClasses} bg-yellow-600 hover:bg-yellow-700 text-white focus:ring-yellow-500 disabled:bg-yellow-300`
      
      case 'error':
        return `${baseClasses} bg-red-600 hover:bg-red-700 text-white focus:ring-red-500 disabled:bg-red-300`
      
      case 'ghost':
        return `${baseClasses} ${
          isDark
            ? 'text-gray-300 hover:bg-gray-800 focus:ring-gray-500'
            : 'text-gray-600 hover:bg-gray-100 focus:ring-gray-500'
        } disabled:text-gray-400`
      
      case 'outline':
        return `${baseClasses} border-2 ${
          isDark
            ? 'border-gray-600 text-gray-300 hover:bg-gray-800 focus:ring-gray-500'
            : 'border-gray-300 text-gray-700 hover:bg-gray-50 focus:ring-gray-500'
        } disabled:border-gray-200 disabled:text-gray-400`
      
      default:
        return baseClasses
    }
  }

  const buttonClasses = [
    getVariantClasses(),
    sizeClasses[size],
    fullWidth ? 'w-full' : '',
    'inline-flex items-center justify-center',
    'disabled:cursor-not-allowed disabled:opacity-50',
    loading ? 'cursor-wait' : '',
    className
  ].filter(Boolean).join(' ')

  return (
    <button
      ref={ref}
      className={buttonClasses}
      disabled={disabled || loading}
      {...props}
    >
      {loading && (
        <svg 
          className="animate-spin -ml-1 mr-2 h-4 w-4" 
          fill="none" 
          viewBox="0 0 24 24"
        >
          <circle 
            className="opacity-25" 
            cx="12" 
            cy="12" 
            r="10" 
            stroke="currentColor" 
            strokeWidth="4"
          />
          <path 
            className="opacity-75" 
            fill="currentColor" 
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          />
        </svg>
      )}
      
      {!loading && leftIcon && (
        <span className="mr-2 flex-shrink-0">
          {leftIcon}
        </span>
      )}
      
      <span className="flex-1">
        {children}
      </span>
      
      {!loading && rightIcon && (
        <span className="ml-2 flex-shrink-0">
          {rightIcon}
        </span>
      )}
    </button>
  )
})

Button.displayName = 'Button'

export default Button
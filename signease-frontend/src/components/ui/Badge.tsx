/**
 * Badge Component
 * ===============
 * 
 * Small status indicators and labels
 */

import React, { forwardRef } from 'react'
import { useTheme } from '../../hooks/useTheme'

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'default' | 'primary' | 'secondary' | 'success' | 'warning' | 'error'
  size?: 'sm' | 'md' | 'lg'
  rounded?: boolean
  children: React.ReactNode
}

export const Badge = forwardRef<HTMLSpanElement, BadgeProps>(({
  variant = 'default',
  size = 'md',
  rounded = false,
  className = '',
  children,
  ...props
}, ref) => {
  const { isDark } = useTheme()

  // Size configurations
  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-2.5 py-1 text-sm',
    lg: 'px-3 py-1.5 text-base'
  }

  // Variant configurations
  const getVariantClasses = () => {
    const baseClasses = 'inline-flex items-center font-medium'
    
    switch (variant) {
      case 'primary':
        return `${baseClasses} bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200`
      
      case 'secondary':
        return `${baseClasses} ${
          isDark 
            ? 'bg-gray-700 text-gray-200' 
            : 'bg-gray-100 text-gray-800'
        }`
      
      case 'success':
        return `${baseClasses} bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200`
      
      case 'warning':
        return `${baseClasses} bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200`
      
      case 'error':
        return `${baseClasses} bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200`
      
      case 'default':
      default:
        return `${baseClasses} ${
          isDark
            ? 'bg-gray-700 text-gray-200'
            : 'bg-gray-100 text-gray-700'
        }`
    }
  }

  const badgeClasses = [
    getVariantClasses(),
    sizeClasses[size],
    rounded ? 'rounded-full' : 'rounded-md',
    className
  ].filter(Boolean).join(' ')

  return (
    <span
      ref={ref}
      className={badgeClasses}
      {...props}
    >
      {children}
    </span>
  )
})

Badge.displayName = 'Badge'

export default Badge
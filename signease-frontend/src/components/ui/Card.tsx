/**
 * Card Component
 * ==============
 * 
 * Reusable card component with consistent styling and variants
 */

import React, { forwardRef } from 'react'
import { useTheme } from '../../hooks/useTheme'

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'elevated' | 'outlined' | 'filled'
  padding?: 'none' | 'sm' | 'md' | 'lg' | 'xl'
  rounded?: 'sm' | 'md' | 'lg' | 'xl' | 'full'
  shadow?: 'none' | 'sm' | 'md' | 'lg' | 'xl'
  hover?: boolean
  interactive?: boolean
  children: React.ReactNode
}

export const Card = forwardRef<HTMLDivElement, CardProps>(({
  variant = 'default',
  padding = 'md',
  rounded = 'lg',
  shadow = 'md',
  hover = false,
  interactive = false,
  className = '',
  children,
  ...props
}, ref) => {
  const { isDark } = useTheme()

  // Padding configurations
  const paddingClasses = {
    none: '',
    sm: 'p-3',
    md: 'p-4',
    lg: 'p-6',
    xl: 'p-8'
  }

  // Rounded configurations
  const roundedClasses = {
    sm: 'rounded-sm',
    md: 'rounded-md',
    lg: 'rounded-lg',
    xl: 'rounded-xl',
    full: 'rounded-full'
  }

  // Shadow configurations
  const shadowClasses = {
    none: '',
    sm: 'shadow-sm',
    md: 'shadow-md',
    lg: 'shadow-lg',
    xl: 'shadow-xl'
  }

  // Variant configurations
  const getVariantClasses = () => {
    switch (variant) {
      case 'elevated':
        return isDark 
          ? 'bg-gray-800 border border-gray-700' 
          : 'bg-white border border-gray-100'
      
      case 'outlined':
        return isDark
          ? 'bg-transparent border-2 border-gray-600'
          : 'bg-transparent border-2 border-gray-200'
      
      case 'filled':
        return isDark
          ? 'bg-gray-700 border border-gray-600'
          : 'bg-gray-50 border border-gray-200'
      
      case 'default':
      default:
        return isDark
          ? 'bg-gray-800 border border-gray-700'
          : 'bg-white border border-gray-200'
    }
  }

  // Interactive states
  const interactiveClasses = interactive || hover ? [
    'transition-all duration-200',
    hover ? 'hover:shadow-lg hover:-translate-y-0.5' : '',
    interactive ? 'cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2' : ''
  ].filter(Boolean).join(' ') : ''

  const cardClasses = [
    getVariantClasses(),
    paddingClasses[padding],
    roundedClasses[rounded],
    shadowClasses[shadow],
    interactiveClasses,
    className
  ].filter(Boolean).join(' ')

  if (interactive) {
    return (
      <button
        ref={ref as React.ForwardedRef<HTMLButtonElement>}
        className={cardClasses}
        {...(props as React.ButtonHTMLAttributes<HTMLButtonElement>)}
      >
        {children}
      </button>
    )
  }

  return (
    <div
      ref={ref}
      className={cardClasses}
      {...props}
    >
      {children}
    </div>
  )
})

Card.displayName = 'Card'

// Card sub-components for better composition
export const CardHeader = forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(({
  className = '',
  children,
  ...props
}, ref) => {
  return (
    <div
      ref={ref}
      className={`border-b border-gray-200 dark:border-gray-700 pb-4 mb-4 ${className}`}
      {...props}
    >
      {children}
    </div>
  )
})

CardHeader.displayName = 'CardHeader'

export const CardTitle = forwardRef<HTMLHeadingElement, React.HTMLAttributes<HTMLHeadingElement>>(({
  className = '',
  children,
  ...props
}, ref) => {
  return (
    <h3
      ref={ref}
      className={`text-lg font-semibold text-gray-900 dark:text-gray-100 ${className}`}
      {...props}
    >
      {children}
    </h3>
  )
})

CardTitle.displayName = 'CardTitle'

export const CardDescription = forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLParagraphElement>>(({
  className = '',
  children,
  ...props
}, ref) => {
  return (
    <p
      ref={ref}
      className={`text-sm text-gray-600 dark:text-gray-400 ${className}`}
      {...props}
    >
      {children}
    </p>
  )
})

CardDescription.displayName = 'CardDescription'

export const CardContent = forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(({
  className = '',
  children,
  ...props
}, ref) => {
  return (
    <div
      ref={ref}
      className={className}
      {...props}
    >
      {children}
    </div>
  )
})

CardContent.displayName = 'CardContent'

export const CardFooter = forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(({
  className = '',
  children,
  ...props
}, ref) => {
  return (
    <div
      ref={ref}
      className={`border-t border-gray-200 dark:border-gray-700 pt-4 mt-4 ${className}`}
      {...props}
    >
      {children}
    </div>
  )
})

CardFooter.displayName = 'CardFooter'

export default Card
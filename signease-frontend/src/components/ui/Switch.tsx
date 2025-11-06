/**
 * Switch Component
 * ================
 * 
 * Toggle switch for boolean settings
 */

import React, { forwardRef } from 'react'
import { useTheme } from '../../hooks/useTheme'

export interface SwitchProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'type' | 'size'> {
  label?: string
  description?: string
  size?: 'sm' | 'md' | 'lg'
}

export const Switch = forwardRef<HTMLInputElement, SwitchProps>(({
  label,
  description,
  size = 'md',
  className = '',
  disabled = false,
  checked = false,
  onChange,
  ...props
}, ref) => {
  const { isDark } = useTheme()

  // Size configurations
  const sizeConfig = {
    sm: {
      switch: 'h-5 w-9',
      thumb: 'h-4 w-4',
      translate: 'translate-x-4'
    },
    md: {
      switch: 'h-6 w-11',
      thumb: 'h-5 w-5',
      translate: 'translate-x-5'
    },
    lg: {
      switch: 'h-7 w-14',
      thumb: 'h-6 w-6',
      translate: 'translate-x-7'
    }
  }

  const config = sizeConfig[size]

  const switchClasses = [
    'relative inline-flex items-center rounded-full transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2',
    config.switch,
    checked 
      ? 'bg-blue-600' 
      : isDark 
        ? 'bg-gray-600' 
        : 'bg-gray-200',
    disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer',
    className
  ].filter(Boolean).join(' ')

  const thumbClasses = [
    'inline-block rounded-full bg-white shadow transform transition-transform duration-200 ease-in-out',
    config.thumb,
    checked ? config.translate : 'translate-x-0.5'
  ].join(' ')

  return (
    <div className="flex items-start space-x-3">
      <label className="relative">
        <input
          ref={ref}
          type="checkbox"
          className="sr-only"
          checked={checked}
          onChange={onChange}
          disabled={disabled}
          {...props}
        />
        <div className={switchClasses}>
          <span className={thumbClasses} />
        </div>
      </label>
      
      {(label || description) && (
        <div className="flex-1">
          {label && (
            <div className={`text-sm font-medium ${
              isDark ? 'text-gray-200' : 'text-gray-900'
            } ${disabled ? 'opacity-50' : ''}`}>
              {label}
            </div>
          )}
          {description && (
            <div className={`text-sm ${
              isDark ? 'text-gray-400' : 'text-gray-500'
            } ${disabled ? 'opacity-50' : ''}`}>
              {description}
            </div>
          )}
        </div>
      )}
    </div>
  )
})

Switch.displayName = 'Switch'

export default Switch
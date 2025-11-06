/**
 * Theme Hook
 * ==========
 * 
 * React hook for managing theme state and dark mode toggle
 */

import React, { useState, useEffect, useCallback, createContext, useContext } from 'react'
import { type Theme, type ThemeMode, getTheme, generateCSSVariables } from '../styles/theme'

interface ThemeContextType {
  theme: Theme
  mode: ThemeMode
  toggleMode: () => void
  setMode: (mode: ThemeMode) => void
  isDark: boolean
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

export const useTheme = (): ThemeContextType => {
  const context = useContext(ThemeContext)
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider')
  }
  return context
}

export const useThemeState = () => {
  // Get initial theme from localStorage or system preference
  const getInitialTheme = (): ThemeMode => {
    if (typeof window === 'undefined') return 'light'
    
    const stored = localStorage.getItem('signease-theme')
    if (stored === 'light' || stored === 'dark') {
      return stored
    }
    
    // Check system preference
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark'
    }
    
    return 'light'
  }

  const [mode, setModeState] = useState<ThemeMode>(getInitialTheme)
  const [theme, setTheme] = useState<Theme>(() => getTheme(mode))

  // Update theme when mode changes
  useEffect(() => {
    const newTheme = getTheme(mode)
    setTheme(newTheme)
    
    // Update CSS custom properties
    const cssVariables = generateCSSVariables(newTheme)
    
    // Apply CSS variables to root
    const styleElement = document.getElementById('theme-variables') || document.createElement('style')
    styleElement.id = 'theme-variables'
    styleElement.textContent = `:root {\n${cssVariables}\n}`
    
    if (!document.getElementById('theme-variables')) {
      document.head.appendChild(styleElement)
    }
    
    // Update document class for dark mode
    if (mode === 'dark') {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
    
    // Store in localStorage
    localStorage.setItem('signease-theme', mode)
  }, [mode])

  // Listen for system theme changes
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    
    const handleChange = (e: MediaQueryListEvent) => {
      // Only auto-switch if user hasn't manually set a preference
      const stored = localStorage.getItem('signease-theme')
      if (!stored) {
        setModeState(e.matches ? 'dark' : 'light')
      }
    }

    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [])

  const setMode = useCallback((newMode: ThemeMode) => {
    setModeState(newMode)
  }, [])

  const toggleMode = useCallback(() => {
    setModeState(prev => prev === 'light' ? 'dark' : 'light')
  }, [])

  const isDark = mode === 'dark'

  return {
    theme,
    mode,
    setMode,
    toggleMode,
    isDark
  }
}

// Theme Provider Component
export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const themeState = useThemeState()

  return (
    <ThemeContext.Provider value={themeState}>
      {children}
    </ThemeContext.Provider>
  )
}

// Hook for accessing theme values directly
export const useThemeValue = () => {
  const { theme } = useTheme()
  
  const getColor = useCallback((colorPath: string): string => {
    const keys = colorPath.split('.')
    let value: any = theme.colors
    
    for (const key of keys) {
      value = value[key]
      if (value === undefined) {
        console.warn(`Color path "${colorPath}" not found in theme`)
        return theme.colors.gray[500]
      }
    }
    
    return value
  }, [theme])

  const getSpacing = useCallback((spacing: keyof typeof theme.spacing): string => {
    return theme.spacing[spacing] || theme.spacing.md
  }, [theme])

  const getFontSize = useCallback((size: keyof typeof theme.typography.fontSize): string => {
    return theme.typography.fontSize[size] || theme.typography.fontSize.base
  }, [theme])

  const getShadow = useCallback((shadow: keyof typeof theme.shadows): string => {
    return theme.shadows[shadow] || theme.shadows.md
  }, [theme])

  const getBorderRadius = useCallback((radius: keyof typeof theme.borderRadius): string => {
    return theme.borderRadius[radius] || theme.borderRadius.md
  }, [theme])

  return {
    theme,
    getColor,
    getSpacing,
    getFontSize,
    getShadow,
    getBorderRadius
  }
}

// Utility hook for responsive design
export const useBreakpoint = () => {
  const { theme } = useTheme()
  const [breakpoint, setBreakpoint] = useState<keyof typeof theme.breakpoints>('sm')

  useEffect(() => {
    const updateBreakpoint = () => {
      const width = window.innerWidth
      
      if (width >= parseInt(theme.breakpoints['2xl'])) {
        setBreakpoint('2xl')
      } else if (width >= parseInt(theme.breakpoints.xl)) {
        setBreakpoint('xl')
      } else if (width >= parseInt(theme.breakpoints.lg)) {
        setBreakpoint('lg')
      } else if (width >= parseInt(theme.breakpoints.md)) {
        setBreakpoint('md')
      } else {
        setBreakpoint('sm')
      }
    }

    updateBreakpoint()
    window.addEventListener('resize', updateBreakpoint)
    return () => window.removeEventListener('resize', updateBreakpoint)
  }, [theme.breakpoints])

  const isMobile = breakpoint === 'sm'
  const isTablet = breakpoint === 'md'
  const isDesktop = breakpoint === 'lg' || breakpoint === 'xl' || breakpoint === '2xl'

  return {
    breakpoint,
    isMobile,
    isTablet,
    isDesktop,
    width: window.innerWidth
  }
}
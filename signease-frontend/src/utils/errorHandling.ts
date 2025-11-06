/**
 * Error Handling Utilities
 * ========================
 * 
 * Centralized error handling and user-friendly error messages
 */

export interface ErrorInfo {
  title: string
  message: string
  type: 'network' | 'validation' | 'server' | 'unknown'
  recoverable: boolean
  suggestions: string[]
}

/**
 * Convert various error types to user-friendly error information
 */
export const parseError = (error: unknown): ErrorInfo => {
  // Network/Fetch Error
  if (error instanceof Error) {
    if (error.message.includes('Failed to fetch') || error.message.includes('Network error')) {
      return {
        title: 'Connection Error',
        message: 'Unable to connect to the backend server',
        type: 'network',
        recoverable: true,
        suggestions: [
          'Check if the backend server is running on localhost:5000',
          'Verify your internet connection',
          'Try refreshing the page'
        ]
      }
    }

    if (error.name === 'AbortError' || error.message.includes('timeout')) {
      return {
        title: 'Request Timeout',
        message: 'The request took too long to complete',
        type: 'network',
        recoverable: true,
        suggestions: [
          'Check your internet connection',
          'The server might be overloaded, try again',
          'Reduce the frequency of requests'
        ]
      }
    }

    return {
      title: 'Unexpected Error',
      message: error.message,
      type: 'unknown',
      recoverable: false,
      suggestions: ['Try refreshing the page', 'Contact support if the problem persists']
    }
  }

  // String error
  if (typeof error === 'string') {
    if (error.includes('HTTP')) {
      const statusMatch = error.match(/HTTP (\d+)/)
      const status = statusMatch ? parseInt(statusMatch[1]) : 0
      
      switch (status) {
        case 404:
          return {
            title: 'Service Not Found',
            message: 'The requested service is not available',
            type: 'server',
            recoverable: false,
            suggestions: [
              'Check if you\'re using the correct API version',
              'Verify the backend server is properly configured'
            ]
          }
        case 500:
          return {
            title: 'Server Error',
            message: 'An error occurred on the server',
            type: 'server',
            recoverable: true,
            suggestions: [
              'Try again in a moment',
              'Check the server logs for more details',
              'Restart the backend server if the problem persists'
            ]
          }
        default:
          return {
            title: 'API Error',
            message: error,
            type: 'server',
            recoverable: true,
            suggestions: [
              'Try again in a moment',
              'Check the server status',
              'Contact support if the problem persists'
            ]
          }
      }
    }

    return {
      title: 'Error',
      message: error,
      type: 'unknown',
      recoverable: false,
      suggestions: ['Try refreshing the page', 'Contact support if the problem persists']
    }
  }

  // Unknown error
  return {
    title: 'Unknown Error',
    message: 'An unexpected error occurred',
    type: 'unknown',
    recoverable: false,
    suggestions: ['Try refreshing the page', 'Contact support if the problem persists']
  }
}

/**
 * Get appropriate retry delay based on error type
 */
export const getRetryDelay = (error: ErrorInfo, attemptNumber: number): number => {
  const baseDelay = 1000 // 1 second
  const maxDelay = 30000 // 30 seconds

  switch (error.type) {
    case 'network':
      // Exponential backoff for network errors
      return Math.min(baseDelay * Math.pow(2, attemptNumber), maxDelay)
    
    case 'validation':
      // Short delay for validation errors
      return 500
    
    case 'server':
      // Medium delay for server errors
      return Math.min(baseDelay * attemptNumber, 10000)
    
    default:
      return baseDelay
  }
}

/**
 * Check if an error is recoverable and should be retried
 */
export const shouldRetry = (error: ErrorInfo, attemptNumber: number): boolean => {
  if (!error.recoverable || attemptNumber >= 3) {
    return false
  }

  // Always retry network errors
  if (error.type === 'network') {
    return true
  }

  // Retry server errors with limit
  if (error.type === 'server' && attemptNumber < 2) {
    return true
  }

  return false
}

/**
 * Format error for display in UI
 */
export const formatErrorForDisplay = (error: ErrorInfo): string => {
  return `${error.title}: ${error.message}`
}

/**
 * Log error with context
 */
export const logError = (error: unknown, context: string): void => {
  const errorInfo = parseError(error)
  console.error(`[${context}] ${errorInfo.title}:`, {
    message: errorInfo.message,
    type: errorInfo.type,
    originalError: error
  })
}
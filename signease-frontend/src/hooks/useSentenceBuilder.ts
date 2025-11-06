/**
 * Sentence Builder Hook
 * ====================
 * 
 * Manages sentence building logic, gesture history, and text-to-speech integration
 */

import { useState, useCallback, useRef, useEffect } from 'react'

export interface GestureDetection {
  gesture: string
  confidence: number
  timestamp: number
  id: string
}

export interface SentenceBuilderState {
  sentence: string
  gestureHistory: GestureDetection[]
  isBuilding: boolean
  lastGesture: string | null
  lastConfidence: number
  wordCount: number
  characterCount: number
}

export interface SentenceBuilderActions {
  addGesture: (gesture: string, confidence: number) => void
  addSpace: () => void
  deleteLast: () => void
  clearSentence: () => void
  clearHistory: () => void
  undoLast: () => void
}

export interface UseSentenceBuilderOptions {
  confidenceThreshold?: number
  stabilityDelay?: number
  maxHistorySize?: number
  autoAddDelay?: number
  enableAutoSpace?: boolean
}

export const useSentenceBuilder = (options: UseSentenceBuilderOptions = {}): SentenceBuilderState & SentenceBuilderActions => {
  const {
    confidenceThreshold = 0.6,
    stabilityDelay = 1000, // 1 second
    maxHistorySize = 20,
    autoAddDelay = 1500, // 1.5 seconds for auto-add
    enableAutoSpace = true
  } = options

  // State
  const [state, setState] = useState<SentenceBuilderState>({
    sentence: '',
    gestureHistory: [],
    isBuilding: false,
    lastGesture: null,
    lastConfidence: 0,
    wordCount: 0,
    characterCount: 0
  })

  // Refs for timers and stability tracking
  const stabilityTimer = useRef<number | null>(null)
  const autoAddTimer = useRef<number | null>(null)
  const lastStableGesture = useRef<string | null>(null)
  const stableGestureCount = useRef<number>(0)
  const lastAddedGesture = useRef<string | null>(null)

  // Generate unique ID for gesture detection
  const generateId = useCallback(() => {
    return `gesture_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }, [])

  // Update word and character counts
  const updateCounts = useCallback((sentence: string) => {
    const words = sentence.trim().split(/\s+/).filter(word => word.length > 0)
    return {
      wordCount: words.length,
      characterCount: sentence.length
    }
  }, [])

  // Add gesture to history
  const addToHistory = useCallback((gesture: string, confidence: number) => {
    const detection: GestureDetection = {
      gesture,
      confidence,
      timestamp: Date.now(),
      id: generateId()
    }

    setState(prev => {
      const newHistory = [detection, ...prev.gestureHistory].slice(0, maxHistorySize)
      return {
        ...prev,
        gestureHistory: newHistory,
        lastGesture: gesture,
        lastConfidence: confidence
      }
    })
  }, [generateId, maxHistorySize])

  // Handle special gestures
  const handleSpecialGesture = useCallback((gesture: string) => {
    switch (gesture.toLowerCase()) {
      case 'space':
      case ' ':
        return 'SPACE'
      case 'delete':
      case 'del':
      case 'backspace':
        return 'DELETE'
      case 'nothing':
      case 'none':
      case '':
        return 'NOTHING'
      default:
        return null
    }
  }, [])

  // Add gesture with stability checking
  const addGesture = useCallback((gesture: string, confidence: number) => {
    // Skip if confidence too low
    if (confidence < confidenceThreshold) {
      return
    }

    // Add to history regardless
    addToHistory(gesture, confidence)

    // Handle special gestures immediately
    const specialGesture = handleSpecialGesture(gesture)
    if (specialGesture === 'SPACE') {
      addSpace()
      return
    }
    if (specialGesture === 'DELETE') {
      deleteLast()
      return
    }
    if (specialGesture === 'NOTHING') {
      // Reset stability tracking for 'nothing' gesture
      lastStableGesture.current = null
      stableGestureCount.current = 0
      if (stabilityTimer.current) {
        clearTimeout(stabilityTimer.current)
        stabilityTimer.current = null
      }
      return
    }

    // Check for gesture stability
    if (lastStableGesture.current === gesture) {
      stableGestureCount.current += 1
    } else {
      lastStableGesture.current = gesture
      stableGestureCount.current = 1
    }

    // Clear existing timers
    if (stabilityTimer.current) {
      window.clearTimeout(stabilityTimer.current)
    }
    if (autoAddTimer.current) {
      window.clearTimeout(autoAddTimer.current)
    }

    // Set stability timer
    stabilityTimer.current = window.setTimeout(() => {
      // Check if gesture is still stable and hasn't been added recently
      if (lastStableGesture.current === gesture && 
          stableGestureCount.current >= 3 && 
          lastAddedGesture.current !== gesture) {
        
        // Auto-add the stable gesture
        setState(prev => {
          const newSentence = prev.sentence + gesture.toUpperCase()
          const counts = updateCounts(newSentence)
          
          return {
            ...prev,
            sentence: newSentence,
            isBuilding: true,
            ...counts
          }
        })

        lastAddedGesture.current = gesture
        
        // Set timer to reset the last added gesture
        autoAddTimer.current = window.setTimeout(() => {
          lastAddedGesture.current = null
        }, autoAddDelay)
      }
    }, stabilityDelay)

    setState(prev => ({
      ...prev,
      isBuilding: true
    }))
  }, [confidenceThreshold, addToHistory, handleSpecialGesture, stabilityDelay, autoAddDelay, updateCounts])

  // Add space manually
  const addSpace = useCallback(() => {
    setState(prev => {
      // Don't add space if sentence is empty or already ends with space
      if (!prev.sentence.trim() || prev.sentence.endsWith(' ')) {
        return prev
      }

      const newSentence = prev.sentence + ' '
      const counts = updateCounts(newSentence)
      
      return {
        ...prev,
        sentence: newSentence,
        ...counts
      }
    })

    // Reset stability tracking
    lastStableGesture.current = null
    stableGestureCount.current = 0
    lastAddedGesture.current = null
  }, [updateCounts])

  // Delete last character
  const deleteLast = useCallback(() => {
    setState(prev => {
      if (!prev.sentence) return prev

      const newSentence = prev.sentence.slice(0, -1)
      const counts = updateCounts(newSentence)
      
      return {
        ...prev,
        sentence: newSentence,
        ...counts
      }
    })

    // Reset stability tracking
    lastStableGesture.current = null
    stableGestureCount.current = 0
    lastAddedGesture.current = null
  }, [updateCounts])

  // Clear entire sentence
  const clearSentence = useCallback(() => {
    setState(prev => ({
      ...prev,
      sentence: '',
      isBuilding: false,
      wordCount: 0,
      characterCount: 0
    }))

    // Clear timers and reset tracking
    if (stabilityTimer.current) {
      window.clearTimeout(stabilityTimer.current)
      stabilityTimer.current = null
    }
    if (autoAddTimer.current) {
      window.clearTimeout(autoAddTimer.current)
      autoAddTimer.current = null
    }
    
    lastStableGesture.current = null
    stableGestureCount.current = 0
    lastAddedGesture.current = null
  }, [])

  // Clear gesture history
  const clearHistory = useCallback(() => {
    setState(prev => ({
      ...prev,
      gestureHistory: [],
      lastGesture: null,
      lastConfidence: 0
    }))
  }, [])

  // Undo last action (remove last word or character)
  const undoLast = useCallback(() => {
    setState(prev => {
      if (!prev.sentence) return prev

      // If sentence ends with space, remove the last word
      if (prev.sentence.endsWith(' ')) {
        const words = prev.sentence.trim().split(' ')
        words.pop()
        const newSentence = words.length > 0 ? words.join(' ') + ' ' : ''
        const counts = updateCounts(newSentence)
        
        return {
          ...prev,
          sentence: newSentence,
          ...counts
        }
      } else {
        // Remove last character
        const newSentence = prev.sentence.slice(0, -1)
        const counts = updateCounts(newSentence)
        
        return {
          ...prev,
          sentence: newSentence,
          ...counts
        }
      }
    })
  }, [updateCounts])

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      if (stabilityTimer.current) {
        window.clearTimeout(stabilityTimer.current)
      }
      if (autoAddTimer.current) {
        window.clearTimeout(autoAddTimer.current)
      }
    }
  }, [])

  // Auto-space detection (add space after complete words)
  useEffect(() => {
    if (!enableAutoSpace || !state.sentence) return

    const words = state.sentence.trim().split(' ')
    const lastWord = words[words.length - 1]
    
    // Auto-add space after common complete words
    const completeWords = ['THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BUT', 'HAVE', 'THIS', 'WILL', 'EACH', 'MAKE', 'LIKE', 'INTO', 'TIME', 'VERY', 'WHEN', 'COME', 'HERE', 'JUST', 'KNOW', 'TAKE', 'THAN', 'THEM', 'WELL', 'WERE']
    
    if (completeWords.includes(lastWord) && !state.sentence.endsWith(' ')) {
      const timer = window.setTimeout(() => {
        addSpace()
      }, 2000) // Wait 2 seconds before auto-adding space

      return () => window.clearTimeout(timer)
    }
  }, [state.sentence, enableAutoSpace, addSpace])

  return {
    ...state,
    addGesture,
    addSpace,
    deleteLast,
    clearSentence,
    clearHistory,
    undoLast
  }
}
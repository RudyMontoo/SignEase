/**
 * AR Overlay Hook
 * ===============
 * 
 * React hook for managing AR text overlay positioning and animations
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import {
  type Point2D,
  type Point3D,
  getHandCenter,
  getHandBoundingBox,
  calculateTextPosition,
  getDistance,
  PositionSmoother
} from '../utils/coordinateUtils'

export interface AROverlayState {
  isVisible: boolean
  position: Point2D
  text: string
  confidence: number
  animationPhase: 'fadeIn' | 'visible' | 'fadeOut' | 'hidden'
  scale: number
  opacity: number
}

export interface AROverlayOptions {
  displayMode: 'floating' | 'fixed' | 'following'
  animationDuration: number
  fadeInDelay: number
  fadeOutDelay: number
  minConfidence: number
  smoothingEnabled: boolean
  smoothingFactor: number
  maxDistance: number
  textOffset: number
}

export interface UseAROverlayProps {
  landmarks: number[] | null
  currentGesture: string | null
  confidence: number
  containerWidth: number
  containerHeight: number
  options?: Partial<AROverlayOptions>
}

const defaultOptions: AROverlayOptions = {
  displayMode: 'floating',
  animationDuration: 300,
  fadeInDelay: 500,
  fadeOutDelay: 1500,
  minConfidence: 0.7,
  smoothingEnabled: true,
  smoothingFactor: 0.3,
  maxDistance: 50,
  textOffset: 30
}

export const useAROverlay = ({
  landmarks,
  currentGesture,
  confidence,
  containerWidth,
  containerHeight,
  options = {}
}: UseAROverlayProps) => {
  const opts = { ...defaultOptions, ...options }
  
  const [overlayState, setOverlayState] = useState<AROverlayState>({
    isVisible: false,
    position: { x: 0, y: 0 },
    text: '',
    confidence: 0,
    animationPhase: 'hidden',
    scale: 0,
    opacity: 0
  })

  // Refs for animation and smoothing
  const animationRef = useRef<number | undefined>(undefined)
  const fadeInTimeoutRef = useRef<number | undefined>(undefined)
  const fadeOutTimeoutRef = useRef<number | undefined>(undefined)
  const positionSmootherRef = useRef(new PositionSmoother(5, opts.smoothingFactor))
  const lastPositionRef = useRef<Point2D>({ x: 0, y: 0 })
  const lastUpdateTimeRef = useRef<number>(Date.now())

  // Convert landmarks array to Point3D array
  const convertLandmarks = useCallback((landmarksArray: number[]): Point3D[] => {
    const points: Point3D[] = []
    for (let i = 0; i < landmarksArray.length; i += 3) {
      points.push({
        x: landmarksArray[i],
        y: landmarksArray[i + 1],
        z: landmarksArray[i + 2]
      })
    }
    return points
  }, [])

  // Calculate optimal position for AR text
  const calculateARPosition = useCallback((handLandmarks: Point3D[]): Point2D => {
    const handCenter = getHandCenter(handLandmarks)
    const handBoundingBox = getHandBoundingBox(handLandmarks)
    
    const position = calculateTextPosition(
      handCenter,
      handBoundingBox,
      containerWidth,
      containerHeight,
      120, // estimated text width
      40,  // estimated text height
      opts.textOffset
    )

    // Apply smoothing if enabled
    if (opts.smoothingEnabled) {
      return positionSmootherRef.current.smooth(position)
    }

    return position
  }, [containerWidth, containerHeight, opts.textOffset, opts.smoothingEnabled])

  // Animation functions
  const startFadeIn = useCallback(() => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
    }

    const startTime = Date.now()
    const animate = () => {
      const elapsed = Date.now() - startTime
      const progress = Math.min(elapsed / opts.animationDuration, 1)
      
      // Easing function for smooth animation
      const easeOut = 1 - Math.pow(1 - progress, 3)
      
      setOverlayState(prev => ({
        ...prev,
        animationPhase: progress < 1 ? 'fadeIn' : 'visible',
        scale: 0.5 + (0.5 * easeOut),
        opacity: easeOut
      }))

      if (progress < 1) {
        animationRef.current = requestAnimationFrame(animate)
      }
    }

    animate()
  }, [opts.animationDuration])

  const startFadeOut = useCallback(() => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
    }

    const startTime = Date.now()
    const animate = () => {
      const elapsed = Date.now() - startTime
      const progress = Math.min(elapsed / opts.animationDuration, 1)
      
      // Easing function for smooth animation
      const easeIn = Math.pow(progress, 2)
      
      setOverlayState(prev => ({
        ...prev,
        animationPhase: progress < 1 ? 'fadeOut' : 'hidden',
        scale: 1 - (0.3 * easeIn),
        opacity: 1 - easeIn
      }))

      if (progress >= 1) {
        setOverlayState(prev => ({
          ...prev,
          isVisible: false
        }))
      } else {
        animationRef.current = requestAnimationFrame(animate)
      }
    }

    animate()
  }, [opts.animationDuration])

  // Main effect for handling gesture changes and positioning
  useEffect(() => {
    const now = Date.now()
    
    // Clear existing timeouts
    if (fadeInTimeoutRef.current) {
      window.clearTimeout(fadeInTimeoutRef.current)
    }
    if (fadeOutTimeoutRef.current) {
      window.clearTimeout(fadeOutTimeoutRef.current)
    }

    // Check if we should show the overlay
    const shouldShow = (
      landmarks &&
      landmarks.length >= 63 && // 21 landmarks * 3 coordinates
      currentGesture &&
      currentGesture !== 'nothing' &&
      confidence >= opts.minConfidence
    )

    if (shouldShow) {
      const handLandmarks = convertLandmarks(landmarks!)
      const newPosition = calculateARPosition(handLandmarks)
      
      // Check if position changed significantly (to avoid unnecessary updates)
      const distance = getDistance(newPosition, lastPositionRef.current)
      const shouldUpdatePosition = distance > 5 || now - lastUpdateTimeRef.current > 100

      if (shouldUpdatePosition) {
        lastPositionRef.current = newPosition
        lastUpdateTimeRef.current = now
      }

      // Update overlay state
      setOverlayState(prev => ({
        ...prev,
        position: shouldUpdatePosition ? newPosition : prev.position,
        text: currentGesture!,
        confidence
      }))

      // Handle visibility and animations
      if (!overlayState.isVisible) {
        // Start fade in after delay
        fadeInTimeoutRef.current = window.setTimeout(() => {
          setOverlayState(prev => ({
            ...prev,
            isVisible: true
          }))
          startFadeIn()
        }, opts.fadeInDelay)
      } else if (overlayState.animationPhase === 'hidden') {
        // Restart fade in if hidden
        setOverlayState(prev => ({
          ...prev,
          isVisible: true
        }))
        startFadeIn()
      }
    } else {
      // Hide overlay
      if (overlayState.isVisible && overlayState.animationPhase !== 'fadeOut') {
        fadeOutTimeoutRef.current = window.setTimeout(() => {
          startFadeOut()
        }, opts.fadeOutDelay)
      }
    }

    // Cleanup function
    return () => {
      if (fadeInTimeoutRef.current) {
        window.clearTimeout(fadeInTimeoutRef.current)
      }
      if (fadeOutTimeoutRef.current) {
        window.clearTimeout(fadeOutTimeoutRef.current)
      }
    }
  }, [
    landmarks,
    currentGesture,
    confidence,
    overlayState.isVisible,
    overlayState.animationPhase,
    convertLandmarks,
    calculateARPosition,
    startFadeIn,
    startFadeOut,
    opts.minConfidence,
    opts.fadeInDelay,
    opts.fadeOutDelay
  ])

  // Reset smoother when landmarks are lost
  useEffect(() => {
    if (!landmarks) {
      positionSmootherRef.current.reset()
    }
  }, [landmarks])

  // Cleanup animations on unmount
  useEffect(() => {
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
      if (fadeInTimeoutRef.current) {
        window.clearTimeout(fadeInTimeoutRef.current)
      }
      if (fadeOutTimeoutRef.current) {
        window.clearTimeout(fadeOutTimeoutRef.current)
      }
    }
  }, [])

  // Update options
  const updateOptions = useCallback((newOptions: Partial<AROverlayOptions>) => {
    Object.assign(opts, newOptions)
    
    // Update smoother if smoothing factor changed
    if (newOptions.smoothingFactor !== undefined) {
      positionSmootherRef.current = new PositionSmoother(5, newOptions.smoothingFactor)
    }
  }, [opts])

  return {
    overlayState,
    updateOptions,
    isVisible: overlayState.isVisible && overlayState.opacity > 0
  }
}
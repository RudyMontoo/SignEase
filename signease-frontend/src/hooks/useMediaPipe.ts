import { useEffect, useRef, useState, useCallback } from 'react'
import { Hands, type Results } from '@mediapipe/hands'
import { Camera } from '@mediapipe/camera_utils'

interface MediaPipeHook {
  landmarks: number[] | null
  isProcessing: boolean
  error: string | null
  startProcessing: () => void
  stopProcessing: () => void
}

export const useMediaPipe = (videoElement: HTMLVideoElement | null): MediaPipeHook => {
  const [landmarks, setLandmarks] = useState<number[] | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  const handsRef = useRef<Hands | null>(null)
  const cameraRef = useRef<Camera | null>(null)

  const onResults = useCallback((results: Results) => {
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      // Extract landmarks from the first detected hand
      const handLandmarks = results.multiHandLandmarks[0]
      const landmarkArray: number[] = []
      
      // Convert landmarks to flat array [x1, y1, z1, x2, y2, z2, ...]
      handLandmarks.forEach(landmark => {
        landmarkArray.push(landmark.x, landmark.y, landmark.z)
      })
      
      console.log('MediaPipe detected hand:', landmarkArray.length, 'landmarks')
      setLandmarks(landmarkArray)
    } else {
      setLandmarks(null)
    }
  }, [])

  const initializeMediaPipe = useCallback(async () => {
    try {
      console.log('Initializing MediaPipe...')
      
      // Initialize MediaPipe Hands
      const hands = new Hands({
        locateFile: (file) => {
          const url = `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
          console.log('Loading MediaPipe file:', url)
          return url
        }
      })

      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,  // Lower threshold like working HTML
        minTrackingConfidence: 0.5
      })

      hands.onResults(onResults)
      handsRef.current = hands

      console.log('MediaPipe initialized successfully')
      setError(null)
    } catch (err) {
      console.error('Failed to initialize MediaPipe:', err)
      setError(`Failed to initialize hand tracking: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }, [onResults])

  const startProcessing = useCallback(async () => {
    console.log('Starting MediaPipe processing...', { 
      videoElement: !!videoElement, 
      hands: !!handsRef.current,
      videoReady: videoElement?.readyState,
      videoSize: videoElement ? `${videoElement.videoWidth}x${videoElement.videoHeight}` : 'N/A'
    })
    
    if (!videoElement || videoElement.readyState < 2) {
      setError('Video element not ready. Please wait for camera to initialize.')
      console.log('Video not ready:', { 
        exists: !!videoElement, 
        readyState: videoElement?.readyState,
        videoWidth: videoElement?.videoWidth 
      })
      return
    }

    if (!handsRef.current) {
      await initializeMediaPipe()
      if (!handsRef.current) {
        setError('Failed to initialize MediaPipe hands model')
        return
      }
    }

    try {
      console.log('Creating MediaPipe camera...', { 
        videoReady: videoElement.readyState,
        videoWidth: videoElement.videoWidth,
        videoHeight: videoElement.videoHeight
      })

      // Use a simpler approach that matches the working HTML test
      const processFrame = async () => {
        if (handsRef.current && videoElement && videoElement.readyState >= 2 && setIsProcessing) {
          try {
            await handsRef.current.send({ image: videoElement })
          } catch (frameError) {
            console.warn('Frame processing error:', frameError)
          }
        }
        
        // Continue processing if still active
        if (cameraRef.current) {
          requestAnimationFrame(processFrame)
        }
      }

      // Start the processing loop
      cameraRef.current = { stop: () => { cameraRef.current = null } } // Simple reference
      setIsProcessing(true)
      setError(null)
      
      // Start processing frames
      processFrame()
      
      console.log('MediaPipe processing started successfully')
    } catch (err) {
      console.error('Failed to start camera processing:', err)
      setError(`Failed to start hand tracking: ${err instanceof Error ? err.message : 'Unknown error'}`)
      setIsProcessing(false)
    }
  }, [videoElement, initializeMediaPipe])

  const stopProcessing = useCallback(() => {
    if (cameraRef.current) {
      cameraRef.current.stop()
      cameraRef.current = null
    }
    setIsProcessing(false)
    setLandmarks(null)
  }, [])

  useEffect(() => {
    initializeMediaPipe()
    
    return () => {
      stopProcessing()
      if (handsRef.current) {
        handsRef.current.close()
      }
    }
  }, [initializeMediaPipe, stopProcessing])

  return {
    landmarks,
    isProcessing,
    error,
    startProcessing,
    stopProcessing
  }
}
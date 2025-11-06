/**
 * Camera utility functions for SignEase MVP
 */

export interface CameraConstraints {
  width?: number
  height?: number
  facingMode?: 'user' | 'environment'
  frameRate?: number
}

export interface CameraCapabilities {
  hasCamera: boolean
  hasPermission: boolean
  supportedConstraints: MediaTrackSupportedConstraints
}

/**
 * Check if camera is available and get permissions
 */
export const checkCameraCapabilities = async (): Promise<CameraCapabilities> => {
  const result: CameraCapabilities = {
    hasCamera: false,
    hasPermission: false,
    supportedConstraints: {}
  }

  try {
    // Check if getUserMedia is supported
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return result
    }

    // Get supported constraints
    result.supportedConstraints = navigator.mediaDevices.getSupportedConstraints()

    // Check for camera devices
    const devices = await navigator.mediaDevices.enumerateDevices()
    result.hasCamera = devices.some(device => device.kind === 'videoinput')

    if (result.hasCamera) {
      // Test camera permission
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true })
        result.hasPermission = true
        // Stop the test stream
        stream.getTracks().forEach(track => track.stop())
      } catch (permissionError) {
        result.hasPermission = false
      }
    }

    return result
  } catch (error) {
    console.error('Error checking camera capabilities:', error)
    return result
  }
}

/**
 * Get optimal camera constraints based on device capabilities
 */
export const getOptimalConstraints = (
  preferredWidth: number = 640,
  preferredHeight: number = 480
): MediaStreamConstraints => {
  return {
    video: {
      width: { ideal: preferredWidth, min: 320, max: 1920 },
      height: { ideal: preferredHeight, min: 240, max: 1080 },
      facingMode: 'user',
      frameRate: { ideal: 30, min: 15, max: 60 }
    },
    audio: false
  }
}

/**
 * Start camera stream with error handling
 */
export const startCameraStream = async (
  constraints?: MediaStreamConstraints
): Promise<MediaStream> => {
  const defaultConstraints = getOptimalConstraints()
  const finalConstraints = constraints || defaultConstraints

  try {
    const stream = await navigator.mediaDevices.getUserMedia(finalConstraints)
    return stream
  } catch (error) {
    console.error('Error starting camera stream:', error)
    
    // Provide user-friendly error messages
    if (error instanceof Error) {
      switch (error.name) {
        case 'NotAllowedError':
          throw new Error('Camera permission denied. Please allow camera access and try again.')
        case 'NotFoundError':
          throw new Error('No camera found. Please connect a camera and try again.')
        case 'NotReadableError':
          throw new Error('Camera is already in use by another application.')
        case 'OverconstrainedError':
          throw new Error('Camera does not support the requested settings.')
        case 'SecurityError':
          throw new Error('Camera access blocked due to security restrictions.')
        default:
          throw new Error(`Camera error: ${error.message}`)
      }
    }
    
    throw new Error('Unknown camera error occurred.')
  }
}

/**
 * Stop camera stream and release resources
 */
export const stopCameraStream = (stream: MediaStream): void => {
  try {
    stream.getTracks().forEach(track => {
      track.stop()
    })
  } catch (error) {
    console.error('Error stopping camera stream:', error)
  }
}

/**
 * Get camera stream info
 */
export const getCameraStreamInfo = (stream: MediaStream): {
  width: number
  height: number
  frameRate: number
  deviceId: string
} => {
  const videoTrack = stream.getVideoTracks()[0]
  const settings = videoTrack.getSettings()
  
  return {
    width: settings.width || 0,
    height: settings.height || 0,
    frameRate: settings.frameRate || 0,
    deviceId: settings.deviceId || ''
  }
}

/**
 * Capture frame from video element
 */
export const captureFrame = (
  videoElement: HTMLVideoElement,
  canvas?: HTMLCanvasElement
): ImageData | null => {
  try {
    const canvasElement = canvas || document.createElement('canvas')
    const ctx = canvasElement.getContext('2d')
    
    if (!ctx) {
      throw new Error('Could not get canvas context')
    }
    
    canvasElement.width = videoElement.videoWidth
    canvasElement.height = videoElement.videoHeight
    
    ctx.drawImage(videoElement, 0, 0)
    
    return ctx.getImageData(0, 0, canvasElement.width, canvasElement.height)
  } catch (error) {
    console.error('Error capturing frame:', error)
    return null
  }
}

/**
 * Convert ImageData to base64 string
 */
export const imageDataToBase64 = (imageData: ImageData): string => {
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  
  if (!ctx) {
    throw new Error('Could not get canvas context')
  }
  
  canvas.width = imageData.width
  canvas.height = imageData.height
  ctx.putImageData(imageData, 0, 0)
  
  return canvas.toDataURL('image/jpeg', 0.8)
}

/**
 * Resize image data
 */
export const resizeImageData = (
  imageData: ImageData,
  newWidth: number,
  newHeight: number
): ImageData => {
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  
  if (!ctx) {
    throw new Error('Could not get canvas context')
  }
  
  // Create temporary canvas with original image
  const tempCanvas = document.createElement('canvas')
  const tempCtx = tempCanvas.getContext('2d')!
  tempCanvas.width = imageData.width
  tempCanvas.height = imageData.height
  tempCtx.putImageData(imageData, 0, 0)
  
  // Resize to new dimensions
  canvas.width = newWidth
  canvas.height = newHeight
  ctx.drawImage(tempCanvas, 0, 0, newWidth, newHeight)
  
  return ctx.getImageData(0, 0, newWidth, newHeight)
}
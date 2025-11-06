import { useRef, useEffect, useState, forwardRef, useImperativeHandle } from 'react'

interface WebcamCaptureProps {
  onFrame?: (imageData: ImageData) => void
  width?: number
  height?: number
  className?: string
}

export interface WebcamCaptureRef {
  getVideoElement: () => HTMLVideoElement | null
}

const WebcamCapture = forwardRef<WebcamCaptureRef, WebcamCaptureProps>(({
  onFrame,
  width = 640,
  height = 480,
  className = ''
}, ref) => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  useImperativeHandle(ref, () => ({
    getVideoElement: () => videoRef.current
  }))

  useEffect(() => {
    startWebcam()
    return () => {
      stopWebcam()
    }
  }, [])

  const startWebcam = async () => {
    try {
      // Check if getUserMedia is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera not supported in this browser')
      }

      setError(null)
      
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: width },
          height: { ideal: height },
          facingMode: 'user'
        }
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.onloadedmetadata = () => {
          console.log('Video metadata loaded:', {
            readyState: videoRef.current?.readyState,
            videoWidth: videoRef.current?.videoWidth,
            videoHeight: videoRef.current?.videoHeight
          })
          setIsStreaming(true)
          setError(null)
          if (onFrame) {
            startFrameCapture()
          }
        }
        
        // Also handle when video data is loaded
        videoRef.current.onloadeddata = () => {
          console.log('Video data loaded, ready for MediaPipe')
        }
        
        // Handle video errors
        videoRef.current.onerror = (e) => {
          console.error('Video element error:', e)
          setError('Video playback error')
          setIsStreaming(false)
        }
      }
    } catch (err) {
      console.error('Error accessing webcam:', err)
      
      let errorMessage = 'Unable to access webcam'
      
      if (err instanceof Error) {
        if (err.name === 'NotAllowedError') {
          errorMessage = 'Camera permission denied. Please allow camera access and refresh.'
        } else if (err.name === 'NotFoundError') {
          errorMessage = 'No camera found. Please connect a camera and try again.'
        } else if (err.name === 'NotReadableError') {
          errorMessage = 'Camera is already in use by another application.'
        } else if (err.name === 'OverconstrainedError') {
          errorMessage = 'Camera does not support the requested settings.'
        } else {
          errorMessage = `Camera error: ${err.message}`
        }
      }
      
      setError(errorMessage)
      setIsStreaming(false)
    }
  }

  const stopWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach(track => track.stop())
      videoRef.current.srcObject = null
    }
    setIsStreaming(false)
  }

  const startFrameCapture = () => {
    const captureFrame = () => {
      if (videoRef.current && canvasRef.current && isStreaming && onFrame) {
        const canvas = canvasRef.current
        const video = videoRef.current
        const ctx = canvas.getContext('2d')

        if (ctx) {
          canvas.width = video.videoWidth
          canvas.height = video.videoHeight
          ctx.drawImage(video, 0, 0)
          
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
          onFrame(imageData)
        }
      }
      
      if (isStreaming) {
        requestAnimationFrame(captureFrame)
      }
    }
    
    requestAnimationFrame(captureFrame)
  }

  if (error) {
    return (
      <div className={`flex items-center justify-center bg-red-50 border border-red-200 rounded-lg ${className}`}>
        <div className="text-center p-8">
          <div className="text-red-600 mb-2">
            <svg className="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
          </div>
          <p className="text-red-800 font-medium">{error}</p>
          <button 
            onClick={startWebcam}
            className="mt-4 bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className={`relative ${className}`}>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full h-full object-cover rounded-lg"
        style={{ transform: 'scaleX(-1)' }} // Mirror the video
      />
      <canvas
        ref={canvasRef}
        className="hidden"
      />
      
      {!isStreaming && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-200 rounded-lg">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Starting webcam...</p>
          </div>
        </div>
      )}
      
      <div className="absolute top-2 right-2">
        <div className={`status-indicator ${isStreaming ? 'status-connected' : 'status-processing'}`}></div>
      </div>
    </div>
  )
})

WebcamCapture.displayName = 'WebcamCapture'

export default WebcamCapture
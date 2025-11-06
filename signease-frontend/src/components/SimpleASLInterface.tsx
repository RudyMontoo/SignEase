import React, { useState, useRef, useEffect } from 'react'
import { useMediaPipe } from '../hooks/useMediaPipe'
import { useGestureAPI } from '../hooks/useGestureAPI'

interface WebcamRef {
  getVideoElement: () => HTMLVideoElement | null
}

const SimpleASLInterface: React.FC = () => {
  const [videoElement, setVideoElement] = useState<HTMLVideoElement | null>(null)
  const [cameraActive, setCameraActive] = useState(false)
  const [currentLandmarks, setCurrentLandmarks] = useState<number[] | null>(null)
  const [predictionCount, setPredictionCount] = useState(0)
  const [lastPrediction, setLastPrediction] = useState<string | null>(null)
  const [lastConfidence, setLastConfidence] = useState<number>(0)
  const [status, setStatus] = useState('🔄 Initializing...')
  const [statusType, setStatusType] = useState<'success' | 'error' | 'warning'>('warning')
  
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  
  // MediaPipe hook
  const { landmarks, isProcessing, error: mediaPipeError, startProcessing, stopProcessing } = useMediaPipe(videoElement)
  
  // Gesture API hook
  const { predictGesture, isConnected } = useGestureAPI()
  
  // Update landmarks when MediaPipe detects them
  useEffect(() => {
    setCurrentLandmarks(landmarks)
    if (landmarks) {
      updateStatus(`✅ Hand detected (${landmarks.length} values)`, 'success')
    } else if (isProcessing) {
      updateStatus('👋 Show your hand to the camera', 'warning')
    } else if (cameraActive) {
      updateStatus('⚠️ MediaPipe not active - click Start Camera again', 'warning')
    }
  }, [landmarks, cameraActive, isProcessing])
  
  // Track MediaPipe processing status
  useEffect(() => {
    if (isProcessing) {
      updateStatus('✅ MediaPipe active - show your hand', 'success')
    }
  }, [isProcessing])
  
  // Handle MediaPipe errors
  useEffect(() => {
    if (mediaPipeError) {
      updateStatus(`❌ MediaPipe error: ${mediaPipeError}`, 'error')
    }
  }, [mediaPipeError])
  
  // Update status helper
  const updateStatus = (message: string, type: 'success' | 'error' | 'warning') => {
    setStatus(message)
    setStatusType(type)
  }
  
  // Start camera
  const startCamera = async () => {
    try {
      updateStatus('📹 Starting camera...', 'warning')
      
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' }
      })
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        streamRef.current = stream
        
        videoRef.current.onloadedmetadata = () => {
          console.log('Video metadata loaded, setting video element')
          setVideoElement(videoRef.current)
          setCameraActive(true)
          updateStatus('✅ Camera active, starting MediaPipe...', 'warning')
        }
        
        videoRef.current.onloadeddata = () => {
          console.log('Video data loaded, ready for MediaPipe')
          // Start MediaPipe processing after video is fully loaded
          setTimeout(() => {
            console.log('Starting MediaPipe processing...')
            startProcessing()
            updateStatus('✅ MediaPipe starting...', 'success')
          }, 2000) // Give more time for video to be ready
        }
      }
      
    } catch (error: any) {
      updateStatus(`❌ Camera error: ${error.message}`, 'error')
      console.error('Camera error:', error)
    }
  }
  
  // Test current gesture
  const testCurrentGesture = async () => {
    console.log('testCurrentGesture called', { 
      currentLandmarks: currentLandmarks ? currentLandmarks.length : 'null',
      isProcessing,
      cameraActive 
    })
    
    if (!currentLandmarks) {
      updateStatus('❌ No hand detected - show your hand to camera', 'error')
      console.log('No landmarks available')
      return
    }
    
    if (currentLandmarks.length !== 63) {
      updateStatus(`❌ Invalid landmarks count: ${currentLandmarks.length} (expected 63)`, 'error')
      console.log('Invalid landmarks:', currentLandmarks)
      return
    }
    
    try {
      updateStatus('🔄 Testing gesture...', 'warning')
      console.log('Sending landmarks to API:', currentLandmarks.slice(0, 6), '...')
      
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ landmarks: currentLandmarks })
      })
      
      console.log('API response status:', response.status)
      
      if (response.ok) {
        const data = await response.json()
        console.log('API response data:', data)
        
        setPredictionCount(prev => prev + 1)
        setLastPrediction(data.prediction)
        setLastConfidence(data.confidence)
        
        updateStatus(`✅ Gesture: ${data.prediction} (${(data.confidence * 100).toFixed(1)}%)`, 'success')
      } else {
        const errorText = await response.text()
        console.error('API error response:', errorText)
        throw new Error(`HTTP ${response.status}: ${errorText}`)
      }
      
    } catch (error: any) {
      console.error('API error:', error)
      updateStatus(`❌ API error: ${error.message}`, 'error')
    }
  }
  
  // Test API direct
  const testAPIDirect = async () => {
    try {
      updateStatus('🧪 Testing API with sample data...', 'warning')
      
      const testLandmarks = [0.5, 0.5, 0.0]
      for (let i = 0; i < 60; i++) {
        testLandmarks.push(0.4 + i * 0.01)
      }
      
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ landmarks: testLandmarks })
      })
      
      if (response.ok) {
        const data = await response.json()
        setLastPrediction(data.prediction)
        setLastConfidence(data.confidence)
        updateStatus('✅ API test successful', 'success')
      } else {
        throw new Error(`HTTP ${response.status}`)
      }
      
    } catch (error: any) {
      updateStatus(`❌ API test failed: ${error.message}`, 'error')
      console.error('API test error:', error)
    }
  }
  
  // Get status color
  const getStatusColor = () => {
    switch (statusType) {
      case 'success': return 'bg-green-100 text-green-800 border-green-200'
      case 'error': return 'bg-red-100 text-red-800 border-red-200'
      case 'warning': return 'bg-yellow-100 text-yellow-800 border-yellow-200'
      default: return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }
  
  return (
    <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      {/* Header */}
      <div className="flex items-center mb-6">
        <div className="text-3xl mr-3">🎯</div>
        <h1 className="text-3xl font-bold text-gray-900">Simple ASL Recognition Test</h1>
      </div>
      
      {/* Status */}
      <div className={`p-3 rounded-lg border mb-6 ${getStatusColor()}`}>
        {status}
      </div>
      
      {/* Camera Area */}
      <div className="text-center mb-6">
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className={`w-full max-w-2xl border-2 border-gray-300 rounded-lg ${cameraActive ? 'block' : 'hidden'}`}
          style={{ transform: 'scaleX(-1)' }}
        />
        
        {!cameraActive && (
          <div className="w-full max-w-2xl h-96 bg-gray-200 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center mx-auto">
            <div className="text-center text-gray-600">
              <div className="text-4xl mb-2">📹</div>
              <div>Camera will appear here</div>
            </div>
          </div>
        )}
      </div>
      
      {/* Control Buttons */}
      <div className="text-center mb-6 space-x-2">
        <button
          onClick={startCamera}
          disabled={cameraActive}
          className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg font-medium"
        >
          Start Camera
        </button>
        
        <button
          onClick={() => {
            console.log('Manual MediaPipe start')
            startProcessing()
            updateStatus('🔄 Starting MediaPipe manually...', 'warning')
          }}
          disabled={!cameraActive || isProcessing}
          className="bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg font-medium"
        >
          Start MediaPipe
        </button>
        
        <button
          onClick={testCurrentGesture}
          disabled={!currentLandmarks}
          className="bg-gray-600 hover:bg-gray-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg font-medium"
        >
          Test Current Gesture
        </button>
        
        <button
          onClick={testAPIDirect}
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-medium"
        >
          Test API Direct
        </button>
        
        <button
          onClick={() => {
            console.log('Debug info:', {
              currentLandmarks: currentLandmarks ? `${currentLandmarks.length} values` : 'null',
              landmarks: landmarks ? `${landmarks.length} values` : 'null',
              isProcessing,
              cameraActive,
              videoElement: !!videoElement
            })
            if (currentLandmarks) {
              console.log('Sample landmarks:', currentLandmarks.slice(0, 9))
            }
            updateStatus(`Debug: Landmarks=${currentLandmarks ? currentLandmarks.length : 0}, Processing=${isProcessing}`, 'warning')
          }}
          className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg font-medium"
        >
          Debug Info
        </button>
      </div>
      
      {/* Result Display */}
      <div className="text-center p-6 border-2 border-gray-300 rounded-lg mb-6">
        {lastPrediction ? (
          <div>
            <div className="text-6xl font-bold text-green-600 mb-2">{lastPrediction}</div>
            <div className="text-lg text-gray-600">
              Confidence: {(lastConfidence * 100).toFixed(1)}%
            </div>
          </div>
        ) : (
          <div className="text-xl text-gray-500">
            Make a gesture and click "Test Current Gesture"
          </div>
        )}
      </div>
      
      {/* Debug Info */}
      <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
        <h3 className="font-bold text-blue-800 mb-2">🔧 Debug Info:</h3>
        <div className="text-sm text-blue-700 space-y-1">
          <div>MediaPipe: {isProcessing ? 'Active' : 'Inactive'}</div>
          <div>Camera: {cameraActive ? 'Active' : 'Inactive'}</div>
          <div>Landmarks: {currentLandmarks ? 'Detected' : 'None'}</div>
          <div>Predictions: {predictionCount}</div>
          <div>Backend: {isConnected ? 'Connected' : 'Disconnected'}</div>
          {mediaPipeError && <div>Error: {mediaPipeError}</div>}
        </div>
      </div>
    </div>
  )
}

export default SimpleASLInterface
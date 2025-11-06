import React, { useRef, useEffect, useState } from 'react'

declare global {
  interface Window {
    Hands: any
    Camera: any
  }
}

const WorkingASLInterface: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [hands, setHands] = useState<any>(null)
  const [camera, setCamera] = useState<any>(null)
  const [currentLandmarks, setCurrentLandmarks] = useState<number[] | null>(null)
  const [status, setStatus] = useState('🔄 Initializing...')
  const [statusType, setStatusType] = useState<'success' | 'error' | 'warning'>('warning')
  const [lastPrediction, setLastPrediction] = useState<string | null>(null)
  const [lastConfidence, setLastConfidence] = useState<number>(0)
  const [predictionCount, setPredictionCount] = useState(0)

  const updateStatus = (message: string, type: 'success' | 'error' | 'warning') => {
    setStatus(message)
    setStatusType(type)
  }

  const getStatusColor = () => {
    switch (statusType) {
      case 'success': return 'bg-green-100 text-green-800 border-green-200'
      case 'error': return 'bg-red-100 text-red-800 border-red-200'
      case 'warning': return 'bg-yellow-100 text-yellow-800 border-yellow-200'
      default: return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  // Load MediaPipe scripts
  useEffect(() => {
    const loadScript = (src: string) => {
      return new Promise((resolve, reject) => {
        const script = document.createElement('script')
        script.src = src
        script.onload = resolve
        script.onerror = reject
        document.head.appendChild(script)
      })
    }

    const loadMediaPipe = async () => {
      try {
        updateStatus('📦 Loading MediaPipe...', 'warning')
        
        await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js')
        await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js')
        
        updateStatus('✅ MediaPipe loaded', 'success')
        initializeMediaPipe()
      } catch (error) {
        updateStatus('❌ Failed to load MediaPipe', 'error')
        console.error('MediaPipe loading error:', error)
      }
    }

    loadMediaPipe()
  }, [])

  const onResults = (results: any) => {
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const landmarks: number[] = []
      results.multiHandLandmarks[0].forEach((landmark: any) => {
        landmarks.push(landmark.x, landmark.y, landmark.z)
      })
      setCurrentLandmarks(landmarks)
      updateStatus(`✅ Hand detected (${landmarks.length} values)`, 'success')
    } else {
      setCurrentLandmarks(null)
      updateStatus('👋 Show your hand to the camera', 'warning')
    }
  }

  const initializeMediaPipe = () => {
    try {
      updateStatus('🤖 Initializing MediaPipe...', 'warning')
      
      const handsInstance = new window.Hands({
        locateFile: (file: string) => {
          return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        }
      })
      
      handsInstance.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      })
      
      handsInstance.onResults(onResults)
      setHands(handsInstance)
      
      updateStatus('✅ MediaPipe initialized', 'success')
    } catch (error) {
      updateStatus('❌ MediaPipe initialization failed', 'error')
      console.error('MediaPipe error:', error)
    }
  }

  const startCamera = async () => {
    try {
      updateStatus('📹 Starting camera...', 'warning')
      
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' }
      })
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        
        videoRef.current.onloadedmetadata = () => {
          updateStatus('✅ Camera active', 'success')
          
          if (hands) {
            const cameraInstance = new window.Camera(videoRef.current, {
              onFrame: async () => {
                if (hands && videoRef.current) {
                  await hands.send({ image: videoRef.current })
                }
              },
              width: 640,
              height: 480
            })
            
            cameraInstance.start()
            setCamera(cameraInstance)
            updateStatus('✅ Camera and MediaPipe active', 'success')
          }
        }
      }
      
    } catch (error: any) {
      updateStatus(`❌ Camera error: ${error.message}`, 'error')
      console.error('Camera error:', error)
    }
  }

  const testCurrentGesture = async () => {
    if (!currentLandmarks) {
      updateStatus('❌ No hand detected', 'error')
      return
    }
    
    try {
      updateStatus('🔄 Testing gesture...', 'warning')
      
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ landmarks: currentLandmarks })
      })
      
      if (response.ok) {
        const data = await response.json()
        setPredictionCount(prev => prev + 1)
        setLastPrediction(data.prediction)
        setLastConfidence(data.confidence)
        
        updateStatus('✅ Gesture recognized!', 'success')
      } else {
        throw new Error(`HTTP ${response.status}`)
      }
      
    } catch (error: any) {
      updateStatus(`❌ API error: ${error.message}`, 'error')
      console.error('API error:', error)
    }
  }

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
          className="w-full max-w-2xl border-2 border-gray-300 rounded-lg mx-auto"
          style={{ transform: 'scaleX(-1)', display: videoRef.current?.srcObject ? 'block' : 'none' }}
        />
        
        {!videoRef.current?.srcObject && (
          <div className="w-full max-w-2xl h-96 bg-gray-200 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center mx-auto">
            <div className="text-center text-gray-600">
              <div className="text-4xl mb-2">📹</div>
              <div>Camera will appear here</div>
            </div>
          </div>
        )}
      </div>
      
      {/* Control Buttons */}
      <div className="text-center mb-6 space-x-4">
        <button
          onClick={startCamera}
          disabled={!hands}
          className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-6 py-3 rounded-lg font-medium"
        >
          Start Camera
        </button>
        
        <button
          onClick={testCurrentGesture}
          disabled={!currentLandmarks}
          className="bg-gray-600 hover:bg-gray-700 disabled:bg-gray-400 text-white px-6 py-3 rounded-lg font-medium"
        >
          Test Current Gesture
        </button>
        
        <button
          onClick={testAPIDirect}
          className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-medium"
        >
          Test API Direct
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
          <div>MediaPipe: {hands ? 'Initialized' : 'Loading...'}</div>
          <div>Camera: {camera ? 'Active' : 'Inactive'}</div>
          <div>Landmarks: {currentLandmarks ? 'Detected' : 'None'}</div>
          <div>Predictions: {predictionCount}</div>
        </div>
      </div>
    </div>
  )
}

export default WorkingASLInterface
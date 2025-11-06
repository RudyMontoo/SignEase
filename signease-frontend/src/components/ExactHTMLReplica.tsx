import React, { useRef, useEffect, useState } from 'react'

// Declare MediaPipe globals
declare global {
  interface Window {
    Hands: any
    Camera: any
  }
}

const ExactHTMLReplica: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const videoPlaceholderRef = useRef<HTMLDivElement>(null)
  const predictionListRef = useRef<HTMLDivElement>(null)
  
  const [hands, setHands] = useState<any>(null)
  const [camera, setCamera] = useState<any>(null)
  const [currentLandmarks, setCurrentLandmarks] = useState<number[] | null>(null)
  const [predictionCount, setPredictionCount] = useState(0)
  const [status, setStatus] = useState('🔄 Initializing...')
  const [statusClass, setStatusClass] = useState('warning')
  const [result, setResult] = useState('Make a gesture and click "Test Current Gesture"')
  const [debugInfo, setDebugInfo] = useState('Loading...')
  const [predictionsVisible, setPredictionsVisible] = useState(false)
  const [testGestureDisabled, setTestGestureDisabled] = useState(true)

  // Exact replica of updateStatus function
  const updateStatus = (message: string, type: string = 'warning') => {
    setStatus(message)
    setStatusClass(type)
  }

  // Exact replica of updateDebugInfo function
  const updateDebugInfo = () => {
    const info = []
    info.push(`MediaPipe: ${hands ? 'Initialized' : 'Not initialized'}`)
    info.push(`Camera: ${camera ? 'Active' : 'Inactive'}`)
    info.push(`Landmarks: ${currentLandmarks ? 'Detected' : 'None'}`)
    info.push(`Predictions: ${predictionCount}`)
    info.push(`Browser: ${navigator.userAgent.split(' ').pop()}`)
    setDebugInfo(info.join('<br>'))
  }

  // Exact replica of onResults function
  const onResults = (results: any) => {
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const landmarks: number[] = []
      results.multiHandLandmarks[0].forEach((landmark: any) => {
        landmarks.push(landmark.x, landmark.y, landmark.z)
      })
      setCurrentLandmarks(landmarks)
      updateStatus(`✅ Hand detected (${landmarks.length} values)`, 'success')
      setTestGestureDisabled(false)
    } else {
      setCurrentLandmarks(null)
      updateStatus('👋 Show your hand to the camera', 'warning')
      setTestGestureDisabled(true)
    }
    updateDebugInfo()
  }

  // Exact replica of initializeMediaPipe function
  const initializeMediaPipe = async () => {
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
      updateDebugInfo()
      
    } catch (error: any) {
      updateStatus(`❌ MediaPipe error: ${error.message}`, 'error')
      console.error('MediaPipe error:', error)
    }
  }

  // Exact replica of startCamera function
  const startCamera = async () => {
    try {
      updateStatus('📹 Starting camera...', 'warning')
      
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' }
      })
      
      if (videoRef.current && videoPlaceholderRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.style.display = 'block'
        videoPlaceholderRef.current.style.display = 'none'
        
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
          await cameraInstance.start()
          setCamera(cameraInstance)
        }
        
        updateStatus('✅ Camera active', 'success')
        updateDebugInfo()
      }
      
    } catch (error: any) {
      updateStatus(`❌ Camera error: ${error.message}`, 'error')
      console.error('Camera error:', error)
    }
  }

  // Exact replica of testCurrentGesture function
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
        
        setResult(`
          <div style="font-size: 48px; color: green;">${data.prediction}</div>
          <div>Confidence: ${(data.confidence * 100).toFixed(1)}%</div>
          <div>Time: ${data.inference_time_ms.toFixed(1)}ms</div>
        `)
        
        // Add to predictions list
        if (predictionListRef.current) {
          const predItem = document.createElement('div')
          predItem.className = 'prediction-item'
          predItem.innerHTML = `
            <strong>${data.prediction}</strong> - ${(data.confidence * 100).toFixed(1)}% 
            (${data.inference_time_ms.toFixed(1)}ms)
          `
          predictionListRef.current.insertBefore(predItem, predictionListRef.current.firstChild)
          setPredictionsVisible(true)
        }
        
        updateStatus('✅ Gesture recognized!', 'success')
        
      } else {
        throw new Error(`HTTP ${response.status}`)
      }
      
    } catch (error: any) {
      updateStatus(`❌ API error: ${error.message}`, 'error')
      console.error('API error:', error)
    }
    
    updateDebugInfo()
  }

  // Exact replica of testAPIDirect function
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
        setResult(`
          <div style="color: blue;">API Test Result:</div>
          <div style="font-size: 48px; color: green;">${data.prediction}</div>
          <div>Confidence: ${(data.confidence * 100).toFixed(1)}%</div>
        `)
        updateStatus('✅ API test successful', 'success')
      } else {
        throw new Error(`HTTP ${response.status}`)
      }
      
    } catch (error: any) {
      updateStatus(`❌ API test failed: ${error.message}`, 'error')
      console.error('API test error:', error)
    }
  }

  // Load MediaPipe scripts and initialize
  useEffect(() => {
    const loadScript = (src: string): Promise<void> => {
      return new Promise((resolve, reject) => {
        if (document.querySelector(`script[src="${src}"]`)) {
          resolve()
          return
        }
        
        const script = document.createElement('script')
        script.src = src
        script.onload = () => resolve()
        script.onerror = reject
        document.head.appendChild(script)
      })
    }

    const loadMediaPipe = async () => {
      try {
        await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js')
        await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js')
        
        // Wait a bit for scripts to be ready
        setTimeout(() => {
          initializeMediaPipe()
        }, 500)
      } catch (error) {
        console.error('Failed to load MediaPipe scripts:', error)
        updateStatus('❌ Failed to load MediaPipe', 'error')
      }
    }

    loadMediaPipe()
    updateDebugInfo()
  }, [])

  // Update debug info when state changes
  useEffect(() => {
    updateDebugInfo()
  }, [hands, camera, currentLandmarks, predictionCount])

  const getStatusClass = () => {
    switch (statusClass) {
      case 'success': return 'bg-green-100 text-green-800 border border-green-200'
      case 'error': return 'bg-red-100 text-red-800 border border-red-200'
      case 'warning': return 'bg-yellow-100 text-yellow-800 border border-yellow-200'
      default: return 'bg-gray-100 text-gray-800 border border-gray-200'
    }
  }

  return (
    <div style={{ 
      fontFamily: 'Arial, sans-serif', 
      maxWidth: '800px', 
      margin: '0 auto', 
      padding: '20px' 
    }}>
      <div style={{ 
        background: 'white', 
        padding: '20px', 
        borderRadius: '10px', 
        boxShadow: '0 2px 10px rgba(0,0,0,0.1)' 
      }}>
        <h1>🎯 Simple ASL Recognition Test</h1>
        
        <div className={`p-2 my-2 rounded ${getStatusClass()}`}>
          {status}
        </div>
        
        <div style={{ textAlign: 'center', margin: '20px 0' }}>
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            style={{ 
              display: 'none',
              width: '100%', 
              maxWidth: '640px', 
              borderRadius: '10px',
              transform: 'scaleX(-1)'
            }}
          />
          <div
            ref={videoPlaceholderRef}
            style={{ 
              padding: '100px', 
              background: '#f0f0f0', 
              borderRadius: '10px' 
            }}
          >
            📹 Camera will appear here
          </div>
        </div>
        
        <div style={{ textAlign: 'center', margin: '20px 0' }}>
          <button
            onClick={startCamera}
            style={{
              background: '#007bff',
              color: 'white',
              border: 'none',
              padding: '10px 20px',
              borderRadius: '5px',
              cursor: 'pointer',
              margin: '5px'
            }}
          >
            Start Camera
          </button>
          <button
            onClick={testCurrentGesture}
            disabled={testGestureDisabled}
            style={{
              background: testGestureDisabled ? '#6c757d' : '#007bff',
              color: 'white',
              border: 'none',
              padding: '10px 20px',
              borderRadius: '5px',
              cursor: testGestureDisabled ? 'not-allowed' : 'pointer',
              margin: '5px'
            }}
          >
            Test Current Gesture
          </button>
          <button
            onClick={testAPIDirect}
            style={{
              background: '#007bff',
              color: 'white',
              border: 'none',
              padding: '10px 20px',
              borderRadius: '5px',
              cursor: 'pointer',
              margin: '5px'
            }}
          >
            Test API Direct
          </button>
        </div>
        
        <div
          style={{
            fontSize: '24px',
            fontWeight: 'bold',
            textAlign: 'center',
            padding: '20px',
            margin: '20px 0',
            border: '2px solid #ddd',
            borderRadius: '10px'
          }}
          dangerouslySetInnerHTML={{ __html: result }}
        />
        
        {predictionsVisible && (
          <div style={{ margin: '20px 0' }}>
            <h3>Recent Predictions:</h3>
            <div ref={predictionListRef}></div>
          </div>
        )}
        
        <div style={{ 
          marginTop: '30px', 
          padding: '20px', 
          background: '#e3f2fd', 
          borderRadius: '10px' 
        }}>
          <h3>🔧 Debug Info:</h3>
          <div dangerouslySetInnerHTML={{ __html: debugInfo }} />
        </div>
      </div>
    </div>
  )
}

export default ExactHTMLReplica
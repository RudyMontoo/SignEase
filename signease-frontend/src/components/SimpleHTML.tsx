import React, { useRef, useEffect, useState } from 'react'

const SimpleHTML: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [hands, setHands] = useState<any>(null)
  const [camera, setCamera] = useState<any>(null)
  const [currentLandmarks, setCurrentLandmarks] = useState<number[] | null>(null)
  const [status, setStatus] = useState('🔄 Initializing...')
  const [statusClass, setStatusClass] = useState('warning')
  const [result, setResult] = useState('Make a gesture and click "Test Current Gesture"')
  const [debugInfo, setDebugInfo] = useState('Loading...')
  const [testGestureDisabled, setTestGestureDisabled] = useState(true)
  const [predictionCount, setPredictionCount] = useState(0)

  const updateStatus = (message: string, type: string = 'warning') => {
    setStatus(message)
    setStatusClass(type)
  }

  const updateDebugInfo = () => {
    const info = []
    info.push(`MediaPipe: ${hands ? 'Initialized' : 'Not initialized'}`)
    info.push(`Camera: ${camera ? 'Active' : 'Inactive'}`)
    info.push(`Landmarks: ${currentLandmarks ? 'Detected' : 'None'}`)
    info.push(`Predictions: ${predictionCount}`)
    info.push(`Browser: ${navigator.userAgent.split(' ').pop()}`)
    setDebugInfo(info.join('<br>'))
  }

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

  const initializeMediaPipe = async () => {
    try {
      updateStatus('🤖 Initializing MediaPipe...', 'warning')
      console.log('Attempting to initialize MediaPipe...')
      
      // Check if MediaPipe is available
      if (!(window as any).Hands) {
        throw new Error('MediaPipe Hands not available on window object')
      }
      
      console.log('Creating Hands instance...')
      const handsInstance = new (window as any).Hands({
        locateFile: (file: string) => {
          const url = `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
          console.log('MediaPipe requesting file:', url)
          return url
        }
      })
      
      console.log('Setting MediaPipe options...')
      handsInstance.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      })
      
      console.log('Setting onResults callback...')
      handsInstance.onResults(onResults)
      setHands(handsInstance)
      
      updateStatus('✅ MediaPipe initialized', 'success')
      console.log('MediaPipe initialization complete!')
      updateDebugInfo()
      
    } catch (error: any) {
      console.error('MediaPipe initialization failed:', error)
      updateStatus(`❌ MediaPipe error: ${error.message}`, 'error')
      
      // Try alternative initialization
      setTimeout(() => {
        console.log('Retrying MediaPipe initialization...')
        if ((window as any).Hands) {
          initializeMediaPipe()
        } else {
          updateStatus('❌ MediaPipe scripts not loaded properly', 'error')
        }
      }, 3000)
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
          if (hands) {
            const cameraInstance = new (window as any).Camera(videoRef.current, {
              onFrame: async () => {
                if (hands && videoRef.current) {
                  await hands.send({ image: videoRef.current })
                }
              },
              width: 640,
              height: 480
            })
            cameraInstance.start().then(() => {
              setCamera(cameraInstance)
              updateStatus('✅ Camera and MediaPipe active', 'success')
              updateDebugInfo()
            })
          } else {
            setCamera(true) // Set camera as active even without MediaPipe
            updateStatus('✅ Camera active', 'success')
            updateDebugInfo()
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
        
        setResult(`
          <div style="font-size: 48px; color: green;">${data.prediction}</div>
          <div>Confidence: ${(data.confidence * 100).toFixed(1)}%</div>
          <div>Time: ${data.inference_time_ms.toFixed(1)}ms</div>
        `)
        
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

  useEffect(() => {
    const loadScript = (src: string): Promise<void> => {
      return new Promise((resolve, reject) => {
        // Check if script already exists
        const existingScript = document.querySelector(`script[src="${src}"]`)
        if (existingScript) {
          console.log('Script already loaded:', src)
          resolve()
          return
        }
        
        console.log('Loading script:', src)
        const script = document.createElement('script')
        script.src = src
        script.onload = () => {
          console.log('Script loaded successfully:', src)
          resolve()
        }
        script.onerror = (error) => {
          console.error('Script failed to load:', src, error)
          reject(error)
        }
        document.head.appendChild(script)
      })
    }

    const loadMediaPipe = async () => {
      try {
        updateStatus('📦 Loading MediaPipe scripts...', 'warning')
        
        // Load scripts sequentially
        await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js')
        console.log('Camera utils loaded')
        
        await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js')
        console.log('Hands model loaded')
        
        // Wait longer for scripts to be ready
        setTimeout(() => {
          console.log('Checking if MediaPipe is available:', {
            Hands: !!(window as any).Hands,
            Camera: !!(window as any).Camera
          })
          
          if ((window as any).Hands && (window as any).Camera) {
            initializeMediaPipe()
          } else {
            updateStatus('❌ MediaPipe not available after loading', 'error')
            console.error('MediaPipe objects not found on window')
          }
        }, 2000) // Wait 2 seconds
        
      } catch (error) {
        console.error('Failed to load MediaPipe scripts:', error)
        updateStatus('❌ Failed to load MediaPipe scripts', 'error')
      }
    }

    loadMediaPipe()
    updateDebugInfo()
  }, [])

  useEffect(() => {
    updateDebugInfo()
  }, [hands, camera, currentLandmarks, predictionCount])

  const getStatusClass = () => {
    switch (statusClass) {
      case 'success': return 'bg-green-100 text-green-800 border border-green-200 p-3 rounded-lg'
      case 'error': return 'bg-red-100 text-red-800 border border-red-200 p-3 rounded-lg'
      case 'warning': return 'bg-yellow-100 text-yellow-800 border border-yellow-200 p-3 rounded-lg'
      default: return 'bg-gray-100 text-gray-800 border border-gray-200 p-3 rounded-lg'
    }
  }

  return (
    <div style={{ fontFamily: 'Arial, sans-serif', maxWidth: '800px', margin: '0 auto', padding: '20px' }}>
      <div style={{ background: 'white', padding: '20px', borderRadius: '10px', boxShadow: '0 2px 10px rgba(0,0,0,0.1)' }}>
        <h1>🤟 SignEase</h1>
        
        <div className={getStatusClass()} style={{ margin: '10px 0' }}>
          {status}
        </div>
        
        <div style={{ textAlign: 'center', margin: '20px 0' }}>
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            style={{ 
              display: camera ? 'block' : 'none',
              width: '100%', 
              maxWidth: '640px', 
              borderRadius: '10px',
              transform: 'scaleX(-1)',
              margin: '0 auto'
            }}
          />
          {!camera && (
            <div style={{ padding: '100px', background: '#f0f0f0', borderRadius: '10px' }}>
              📹 Camera will appear here
            </div>
          )}
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
        
        <div style={{ marginTop: '30px', padding: '20px', background: '#e3f2fd', borderRadius: '10px' }}>
          <h3>🔧 Debug Info:</h3>
          <div dangerouslySetInnerHTML={{ __html: debugInfo }} />
        </div>
      </div>
    </div>
  )
}

export default SimpleHTML
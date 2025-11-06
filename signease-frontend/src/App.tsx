import { useState, useRef, useEffect } from 'react'
import './App.css'
import { ThemeProvider } from './hooks/useTheme'
import WebcamCapture, { type WebcamCaptureRef } from './components/WebcamCapture'
import GestureDisplay from './components/GestureDisplay'
import SentenceBuilder from './components/SentenceBuilder'

import DebugPanel from './components/DebugPanel'
import SpeechSettings from './components/SpeechSettings'
import ThemeToggle from './components/ThemeToggle'
import Card, { CardHeader, CardTitle, CardContent } from './components/ui/Card'
import Button from './components/ui/Button'
import Badge from './components/ui/Badge'
import AROverlay, { AROverlaySettings } from './components/AROverlay'
import { type AROverlayOptions } from './hooks/useAROverlay'
import { useMediaPipe } from './hooks/useMediaPipe'
import { useGestureAPI } from './hooks/useGestureAPI.js'
import { useSentenceBuilder } from './hooks/useSentenceBuilder'
import { useSpeech } from './hooks/useSpeech'
import { parseError, formatErrorForDisplay } from './utils/errorHandling.js'

function AppContent() {
  const [isRecognitionActive, setIsRecognitionActive] = useState(false)
  const [fps, setFps] = useState<number>(0)
  const [videoElement, setVideoElement] = useState<HTMLVideoElement | null>(null)
  const webcamRef = useRef<WebcamCaptureRef>(null)
  
  const { landmarks, isProcessing, error: mediaPipeError, startProcessing, stopProcessing } = useMediaPipe(videoElement)
  
  // Gesture API integration
  const {
    currentGesture,
    confidence,
    alternatives,
    isLoading: isAPILoading,
    error: apiError,
    isConnected,
    totalPredictions,
    averageResponseTime,
    predictGesture,
    clearError,
    testConnection
  } = useGestureAPI({ throttleMs: 200 }) // Throttle to 5 FPS for API calls

  // Sentence building
  const {
    sentence,
    gestureHistory,
    wordCount,
    characterCount,
    isBuilding,
    addGesture,
    addSpace,
    deleteLast,
    clearSentence,
    clearHistory,
    undoLast
  } = useSentenceBuilder({
    confidenceThreshold: 0.6,
    stabilityDelay: 1000,
    autoAddDelay: 1500
  })

  // Text-to-speech
  const { speak, isSupported: speechSupported, isSpeaking } = useSpeech()

  // AR Overlay state
  const [arEnabled, setArEnabled] = useState(true)
  const [arOptions, setArOptions] = useState<Partial<AROverlayOptions>>({
    displayMode: 'floating',
    minConfidence: 0.7,
    smoothingEnabled: true,
    fadeInDelay: 300,
    fadeOutDelay: 1000
  })
  
  // Get video element when webcam is ready
  useEffect(() => {
    const getVideoElement = () => {
      if (webcamRef.current) {
        const video = webcamRef.current.getVideoElement()
        if (video && video !== videoElement && video.readyState >= 1) {
          console.log('Video element ready:', video.readyState, video.videoWidth, video.videoHeight)
          setVideoElement(video)
        }
      }
    }
    
    // Check immediately and then poll
    getVideoElement()
    const interval = setInterval(getVideoElement, 1000) // Check every second
    return () => clearInterval(interval)
  }, [videoElement])
  
  // FPS calculation
  const fpsRef = useRef<number[]>([])
  const lastFrameTime = useRef<number>(Date.now())
  
  useEffect(() => {
    if (landmarks && isRecognitionActive) {
      console.log('App: Received landmarks, sending to API...', landmarks.length)
      
      // Calculate FPS
      const now = Date.now()
      const deltaTime = now - lastFrameTime.current
      lastFrameTime.current = now
      
      fpsRef.current.push(1000 / deltaTime)
      if (fpsRef.current.length > 30) {
        fpsRef.current.shift()
      }
      
      const avgFps = fpsRef.current.reduce((a, b) => a + b, 0) / fpsRef.current.length
      setFps(Math.round(avgFps))
      
      // Send landmarks to backend API for gesture recognition
      predictGesture(landmarks, 'Right').catch((error: unknown) => {
        console.error('Gesture prediction failed:', error)
      })
    }
  }, [landmarks, isRecognitionActive, predictGesture])

  // Add detected gestures to sentence builder
  useEffect(() => {
    if (currentGesture && confidence > 0.6) {
      addGesture(currentGesture, confidence)
    }
  }, [currentGesture, confidence, addGesture])
  
  const handleStartRecognition = async () => {
    if (!isRecognitionActive) {
      // Test backend connection first
      if (!isConnected) {
        const connected = await testConnection()
        if (!connected) {
          alert('Cannot connect to backend server. Please make sure it\'s running on localhost:5000')
          return
        }
      }
      
      setIsRecognitionActive(true)
      
      // Wait a moment for video element to be fully ready
      setTimeout(() => {
        startProcessing()
      }, 1000)
    } else {
      setIsRecognitionActive(false)
      stopProcessing()
      setFps(0)
    }
  }

  // Handle text-to-speech
  const handleSpeak = async (text: string) => {
    if (!speechSupported) {
      alert('Text-to-speech is not supported in your browser')
      return
    }
    
    try {
      await speak(text)
    } catch (error) {
      console.error('Speech failed:', error)
      alert('Speech failed. Please try again.')
    }
  }
  
  // Clear API errors when they occur
  useEffect(() => {
    if (apiError) {
      const timer = setTimeout(() => {
        clearError()
      }, 5000) // Clear error after 5 seconds
      
      return () => clearTimeout(timer)
    }
  }, [apiError, clearError])
  
  const getStatusText = () => {
    if (mediaPipeError) return 'MediaPipe Error: ' + mediaPipeError
    if (apiError) return 'API Error: ' + formatErrorForDisplay(parseError(apiError))
    if (!isConnected) return 'Backend disconnected'
    if (!isRecognitionActive) return 'Ready to start'
    if (isAPILoading) return 'Predicting gesture...'
    if (isProcessing && landmarks) return 'Hand detected - Recognizing...'
    if (isProcessing) return 'Waiting for hand detection'
    return 'Processing...'
  }
  
  const getStatusColor = () => {
    if (mediaPipeError || apiError) return 'text-red-600'
    if (!isConnected) return 'text-red-600'
    if (!isRecognitionActive) return 'text-gray-600'
    if (isAPILoading) return 'text-blue-600'
    if (isProcessing && landmarks) return 'text-green-600'
    if (isProcessing) return 'text-yellow-600'
    return 'text-gray-600'
  }

  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900 transition-colors duration-200 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-between mb-4">
            <div className="flex-1" />
            <h1 className="text-4xl font-bold text-gray-900 dark:text-gray-100 mb-4">
              SignEase MVP
            </h1>
            <div className="flex-1 flex justify-end">
              <ThemeToggle />
            </div>
          </div>
          <p className="text-xl text-gray-600 dark:text-gray-400">
            Real-time ASL Gesture Recognition with AI
          </p>
          <div className="flex justify-center space-x-2 mt-4">
            <Badge variant="primary">RTX 5060 Powered</Badge>
            <Badge variant="success">99.57% Accuracy</Badge>
            <Badge variant="secondary">Real-time</Badge>
          </div>
        </div>
        


        {/* Main Content Grid */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Left Column - Webcam and Gesture Display */}
          <div className="xl:col-span-2 space-y-6">
            {/* Webcam Feed */}
            <Card padding="lg" shadow="lg">
              <CardHeader>
                <CardTitle>Camera Feed</CardTitle>
                <div className="flex items-center space-x-4">
                  {fps > 0 && (
                    <span className="text-sm text-gray-500">FPS: {fps}</span>
                  )}
                  <Button
                    onClick={handleStartRecognition}
                    variant={isRecognitionActive ? 'error' : 'primary'}
                    size="md"
                  >
                    {isRecognitionActive ? 'Stop Recognition' : 'Start Recognition'}
                  </Button>
                </div>
              </CardHeader>
              
              <CardContent>
                <div className="relative">
                  <WebcamCapture 
                    ref={webcamRef}
                    className="w-full h-96 bg-gray-200 rounded-lg"
                    width={640}
                    height={480}
                  />
                  
                  {/* AR Text Overlay */}
                  <AROverlay
                    landmarks={landmarks}
                    currentGesture={currentGesture}
                    confidence={confidence}
                    containerWidth={640}
                    containerHeight={384} // Adjusted for aspect ratio
                    enabled={arEnabled && isRecognitionActive}
                    options={arOptions}
                    className="rounded-lg"
                  />
                  
                  {/* Landmark overlay */}
                  {landmarks && (
                    <div className="absolute top-2 left-2 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-sm">
                      Hand detected: {landmarks.length / 3} points
                    </div>
                  )}
                  
                  {/* AR Status Indicator */}
                  {arEnabled && isRecognitionActive && (
                    <div className="absolute top-2 right-2 bg-blue-600 bg-opacity-80 text-white px-2 py-1 rounded text-xs flex items-center space-x-1">
                      <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                      <span>AR Mode</span>
                    </div>
                  )}
                  
                  {/* Recognition Status Overlay */}
                  {!isRecognitionActive && (
                    <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center rounded-lg">
                      <div className="text-center text-white">
                        <div className="text-4xl mb-4">👋</div>
                        <div className="text-xl font-semibold mb-2">Click "Start Recognition"</div>
                        <div className="text-sm">to begin ASL gesture detection</div>
                      </div>
                    </div>
                  )}
                </div>
              
                {/* Status */}
                <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <p className={`text-sm font-medium ${getStatusColor()}`}>
                    {getStatusText()}
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Gesture Display */}
            <GestureDisplay
              currentGesture={currentGesture}
              confidence={confidence}
              alternatives={alternatives}
              isLoading={isAPILoading}
              isConnected={isConnected}
            />
          </div>

          {/* Right Column - Sentence Builder and System Status */}
          <div className="space-y-6">
            {/* Sentence Builder */}
            <SentenceBuilder
              sentence={sentence}
              gestureHistory={gestureHistory}
              wordCount={wordCount}
              characterCount={characterCount}
              isBuilding={isBuilding}
              onClear={clearSentence}
              onAddSpace={addSpace}
              onDeleteLast={deleteLast}
              onUndo={undoLast}
              onSpeak={speechSupported ? handleSpeak : undefined}
            />

            {/* System Status */}
            <Card padding="lg" shadow="lg">
              <CardHeader>
                <CardTitle>System Status</CardTitle>
              </CardHeader>
              <CardContent>
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span>Backend API:</span>
                  <span className={isConnected ? 'text-green-600' : 'text-red-600'}>
                    {isConnected ? 'Connected' : 'Disconnected'}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>MediaPipe:</span>
                  <span className={isProcessing ? 'text-green-600' : 'text-gray-500'}>
                    {isProcessing ? 'Active' : 'Inactive'}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Hand Detection:</span>
                  <span className={landmarks ? 'text-green-600' : 'text-gray-500'}>
                    {landmarks ? `${landmarks.length / 3} points` : 'None'}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Text-to-Speech:</span>
                  <span className={speechSupported ? 'text-green-600' : 'text-red-600'}>
                    {speechSupported ? 'Available' : 'Not Supported'}
                  </span>
                </div>
                {totalPredictions > 0 && (
                  <>
                    <div className="flex justify-between text-sm">
                      <span>Predictions:</span>
                      <span className="text-purple-600">{totalPredictions}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Avg Response:</span>
                      <span className="text-purple-600">{Math.round(averageResponseTime)}ms</span>
                    </div>
                  </>
                )}
              </div>
              </CardContent>
            </Card>

            {/* AR Overlay Settings */}
            <Card padding="lg" shadow="lg">
              <CardHeader>
                <CardTitle>AR Text Overlay</CardTitle>
              </CardHeader>
              <CardContent>
                <AROverlaySettings
                  enabled={arEnabled}
                  onEnabledChange={setArEnabled}
                  options={arOptions}
                  onOptionsChange={setArOptions}
                />
              </CardContent>
            </Card>

            {/* Debug Panel */}
            <Card padding="lg" shadow="lg">
              <CardHeader>
                <CardTitle>Debug Panel</CardTitle>
              </CardHeader>
              <CardContent>
                <DebugPanel
                  landmarks={landmarks}
                  isProcessing={isProcessing}
                  isRecognitionActive={isRecognitionActive}
                  currentGesture={currentGesture}
                  confidence={confidence}
                  isConnected={isConnected}
                  error={mediaPipeError || apiError}
                />
              </CardContent>
            </Card>

            {/* Speech Settings */}
            <SpeechSettings />

            {/* Quick Actions */}
            <Card padding="lg" shadow="lg">
              <CardHeader>
                <CardTitle>Quick Actions</CardTitle>
              </CardHeader>
              <CardContent>
              <div className="space-y-2">
                <Button
                  onClick={clearHistory}
                  variant="secondary"
                  size="sm"
                  fullWidth
                >
                  Clear Gesture History
                </Button>
                <Button
                  onClick={() => testConnection()}
                  variant="primary"
                  size="sm"
                  fullWidth
                >
                  Test Backend Connection
                </Button>
                {sentence && (
                  <Button
                    onClick={() => handleSpeak(sentence)}
                    disabled={isSpeaking || !speechSupported}
                    variant="success"
                    size="sm"
                    fullWidth
                    loading={isSpeaking}
                  >
                    {isSpeaking ? 'Speaking...' : 'Speak Sentence'}
                  </Button>
                )}
              </div>
              </CardContent>
            </Card>
          </div>
        </div>
        
        {/* Footer */}
        <div className="mt-8 text-center text-sm text-gray-500 dark:text-gray-400">
          <p>Backend API: <span className="font-mono">http://localhost:5000</span></p>
          <p>Frontend: <span className="font-mono">http://localhost:5173</span></p>
          <p className="mt-2">Built with React + TypeScript + MediaPipe + PyTorch</p>
        </div>
      </div>
    </div>
  )
}

// Main App component with ThemeProvider
function App() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  )
}

export default App
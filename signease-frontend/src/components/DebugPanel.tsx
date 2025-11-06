import React from 'react'

interface DebugPanelProps {
  landmarks: number[] | null
  isProcessing: boolean
  isRecognitionActive: boolean
  currentGesture: string | null
  confidence: number
  isConnected: boolean
  error: string | null
}

const DebugPanel: React.FC<DebugPanelProps> = ({
  landmarks,
  isProcessing,
  isRecognitionActive,
  currentGesture,
  confidence,
  isConnected,
  error
}) => {
  return (
    <div className="bg-gray-100 p-4 rounded-lg text-sm font-mono">
      <h4 className="font-bold mb-2">🔍 Debug Info</h4>
      <div className="space-y-1">
        <div>Recognition Active: <span className={isRecognitionActive ? 'text-green-600' : 'text-red-600'}>{isRecognitionActive ? 'YES' : 'NO'}</span></div>
        <div>MediaPipe Processing: <span className={isProcessing ? 'text-green-600' : 'text-red-600'}>{isProcessing ? 'YES' : 'NO'}</span></div>
        <div>Hand Detected: <span className={landmarks ? 'text-green-600' : 'text-red-600'}>{landmarks ? `YES (${landmarks.length} values)` : 'NO'}</span></div>
        <div>Backend Connected: <span className={isConnected ? 'text-green-600' : 'text-red-600'}>{isConnected ? 'YES' : 'NO'}</span></div>
        <div>Current Gesture: <span className="text-blue-600">{currentGesture || 'None'}</span></div>
        <div>Confidence: <span className="text-blue-600">{confidence ? (confidence * 100).toFixed(1) + '%' : '0%'}</span></div>
        {error && <div>Error: <span className="text-red-600">{error}</span></div>}
      </div>
      
      {landmarks && (
        <div className="mt-2 pt-2 border-t">
          <div>Sample landmarks: [{landmarks.slice(0, 6).map(l => l.toFixed(3)).join(', ')}...]</div>
        </div>
      )}
    </div>
  )
}

export default DebugPanel
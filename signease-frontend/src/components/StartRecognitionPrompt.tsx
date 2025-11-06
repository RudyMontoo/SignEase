import React from 'react'

interface StartRecognitionPromptProps {
  onStart: () => void
  isConnected: boolean
}

const StartRecognitionPrompt: React.FC<StartRecognitionPromptProps> = ({ onStart, isConnected }) => {
  return (
    <div className="flex flex-col items-center justify-center h-96 bg-gradient-to-br from-blue-50 to-indigo-100 rounded-lg border-2 border-dashed border-blue-300">
      <div className="text-center p-8">
        {/* Large gesture icon */}
        <div className="text-6xl mb-4">👋</div>
        
        <h3 className="text-2xl font-bold text-gray-800 mb-4">
          Ready to Start ASL Recognition
        </h3>
        
        <p className="text-gray-600 mb-6 max-w-md">
          Click the button below to start recognizing your ASL gestures in real-time using our RTX 5060 powered AI model.
        </p>
        
        {/* Status indicators */}
        <div className="flex justify-center space-x-4 mb-6">
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className="text-sm text-gray-600">
              Backend {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
            <span className="text-sm text-gray-600">Camera Ready</span>
          </div>
        </div>
        
        {/* Start button */}
        <button
          onClick={onStart}
          disabled={!isConnected}
          className={`
            px-8 py-4 rounded-lg font-semibold text-lg transition-all duration-200 transform
            ${isConnected 
              ? 'bg-blue-600 hover:bg-blue-700 text-white hover:scale-105 shadow-lg hover:shadow-xl' 
              : 'bg-gray-400 text-gray-200 cursor-not-allowed'
            }
          `}
        >
          {isConnected ? '🚀 Start Recognition' : '⚠️ Backend Disconnected'}
        </button>
        
        {!isConnected && (
          <p className="text-red-600 text-sm mt-4">
            Please make sure the backend server is running on localhost:5000
          </p>
        )}
        
        {/* Instructions */}
        <div className="mt-8 p-4 bg-white rounded-lg shadow-sm">
          <h4 className="font-semibold text-gray-800 mb-2">Quick Tips:</h4>
          <ul className="text-sm text-gray-600 space-y-1">
            <li>• Hold your hand clearly in front of the camera</li>
            <li>• Make sure you have good lighting</li>
            <li>• Try different ASL letters (A, B, C, etc.)</li>
            <li>• Keep your hand steady for better recognition</li>
          </ul>
        </div>
      </div>
    </div>
  )
}

export default StartRecognitionPrompt
/**
 * Sentence Builder Component
 * ==========================
 * 
 * Interactive sentence building with gesture history and text-to-speech
 */

import React, { useState, useEffect, useRef } from 'react'
import { type GestureDetection } from '../hooks/useSentenceBuilder'
import SpeechControls from './SpeechControls'

export interface SentenceBuilderProps {
  sentence: string
  gestureHistory: GestureDetection[]
  wordCount: number
  characterCount: number
  isBuilding: boolean
  onClear: () => void
  onAddSpace: () => void
  onDeleteLast: () => void
  onUndo: () => void
  onSpeak?: (text: string) => void
  className?: string
}

export const SentenceBuilder: React.FC<SentenceBuilderProps> = ({
  sentence,
  gestureHistory,
  wordCount,
  characterCount,
  isBuilding,
  onClear,
  onAddSpace,
  onDeleteLast,
  onUndo,
  onSpeak,
  className = ''
}) => {
  // Note: isSpeaking state is now handled by SpeechControls component
  const [showHistory, setShowHistory] = useState(false)
  const [copySuccess, setCopySuccess] = useState(false)
  const textAreaRef = useRef<HTMLTextAreaElement>(null)

  // Note: Speech handling is now done by SpeechControls component

  // Handle copy to clipboard
  const handleCopy = async () => {
    if (!sentence.trim()) return

    try {
      await navigator.clipboard.writeText(sentence)
      setCopySuccess(true)
      window.setTimeout(() => setCopySuccess(false), 2000)
    } catch (error) {
      console.error('Copy failed:', error)
    }
  }

  // Auto-resize textarea
  useEffect(() => {
    if (textAreaRef.current) {
      textAreaRef.current.style.height = 'auto'
      textAreaRef.current.style.height = `${textAreaRef.current.scrollHeight}px`
    }
  }, [sentence])

  // Format timestamp for history
  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp)
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  }

  // Get confidence color class
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600 bg-green-100'
    if (confidence >= 0.6) return 'text-yellow-600 bg-yellow-100'
    return 'text-red-600 bg-red-100'
  }

  return (
    <div className={`bg-white rounded-xl shadow-lg ${className}`}>
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-800">Sentence Builder</h3>
          <div className="flex items-center space-x-4 text-sm text-gray-500">
            <span>{wordCount} words</span>
            <span>{characterCount} characters</span>
            {isBuilding && (
              <div className="flex items-center space-x-1 text-blue-600">
                <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse"></div>
                <span>Building...</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main text area */}
      <div className="p-6">
        <div className="relative">
          <textarea
            ref={textAreaRef}
            value={sentence}
            readOnly
            placeholder="Your sentence will appear here as you sign..."
            className="w-full min-h-[120px] p-4 text-lg border-2 border-gray-200 rounded-lg resize-none focus:outline-none focus:border-blue-500 transition-colors"
            style={{ 
              fontFamily: 'system-ui, -apple-system, sans-serif',
              lineHeight: '1.5'
            }}
          />
          
          {/* Cursor indicator when building */}
          {isBuilding && sentence && (
            <div className="absolute bottom-4 right-4">
              <div className="w-0.5 h-6 bg-blue-600 animate-pulse"></div>
            </div>
          )}
        </div>

        {/* Action buttons */}
        <div className="flex flex-wrap gap-3 mt-4">
          {/* Primary actions */}
          <button
            onClick={onAddSpace}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
            title="Add space"
          >
            Space
          </button>
          
          <button
            onClick={onDeleteLast}
            disabled={!sentence}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors font-medium"
            title="Delete last character"
          >
            Delete
          </button>
          
          <button
            onClick={onUndo}
            disabled={!sentence}
            className="px-4 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors font-medium"
            title="Undo last action"
          >
            Undo
          </button>
          
          <button
            onClick={onClear}
            disabled={!sentence}
            className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors font-medium"
            title="Clear all text"
          >
            Clear All
          </button>

          {/* Secondary actions */}
          <div className="flex gap-2 ml-auto">
            <button
              onClick={handleCopy}
              disabled={!sentence.trim()}
              className="px-3 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 disabled:bg-gray-100 disabled:cursor-not-allowed transition-colors"
              title="Copy to clipboard"
            >
              {copySuccess ? (
                <span className="flex items-center space-x-1 text-green-600">
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  <span>Copied!</span>
                </span>
              ) : (
                <span className="flex items-center space-x-1">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                  <span>Copy</span>
                </span>
              )}
            </button>

            {/* Speech Controls */}
            {onSpeak && (
              <SpeechControls
                text={sentence}
                onSpeak={onSpeak}
                showAdvancedControls={false}
              />
            )}
          </div>
        </div>
      </div>

      {/* Gesture History */}
      <div className="border-t border-gray-200">
        <button
          onClick={() => setShowHistory(!showHistory)}
          className="w-full px-6 py-3 text-left text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors flex items-center justify-between"
        >
          <span>Gesture History ({gestureHistory.length})</span>
          <svg 
            className={`w-4 h-4 transition-transform ${showHistory ? 'rotate-180' : ''}`}
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {showHistory && (
          <div className="px-6 pb-4 max-h-48 overflow-y-auto">
            {gestureHistory.length === 0 ? (
              <p className="text-sm text-gray-500 text-center py-4">No gestures detected yet</p>
            ) : (
              <div className="space-y-2">
                {gestureHistory.slice(0, 10).map((detection) => (
                  <div key={detection.id} className="flex items-center justify-between p-2 bg-gray-50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <span className="text-lg font-bold text-gray-800 w-8 text-center">
                        {detection.gesture}
                      </span>
                      <span className="text-xs text-gray-500">
                        {formatTime(detection.timestamp)}
                      </span>
                    </div>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getConfidenceColor(detection.confidence)}`}>
                      {Math.round(detection.confidence * 100)}%
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default SentenceBuilder
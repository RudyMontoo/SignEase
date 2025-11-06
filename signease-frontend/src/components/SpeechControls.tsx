/**
 * Speech Controls Component
 * =========================
 * 
 * Advanced controls for text-to-speech functionality with voice settings
 */

import { useState } from 'react'
import { useSpeech } from '../hooks/useSpeech'

export interface SpeechControlsProps {
  text: string
  onSpeak?: (text: string) => void
  className?: string
  showAdvancedControls?: boolean
}

export const SpeechControls: React.FC<SpeechControlsProps> = ({
  text,
  onSpeak,
  className = '',
  showAdvancedControls = false
}) => {
  const [showSettings, setShowSettings] = useState(false)
  const {
    isSupported,
    isSpeaking,
    isPaused,
    voices,
    settings,
    error,
    speak,
    pause,
    resume,
    stop,
    updateSettings
  } = useSpeech()

  const handleSpeak = async () => {
    if (!text.trim()) return
    
    try {
      if (onSpeak) {
        onSpeak(text)
      } else {
        await speak(text)
      }
    } catch (error) {
      console.error('Speech failed:', error)
    }
  }

  const handleVoiceChange = (voiceURI: string) => {
    const selectedVoice = voices.find(voice => voice.voiceURI === voiceURI)
    updateSettings({ voice: selectedVoice || null })
  }

  const handleRateChange = (rate: number) => {
    updateSettings({ rate })
  }

  const handlePitchChange = (pitch: number) => {
    updateSettings({ pitch })
  }

  const handleVolumeChange = (volume: number) => {
    updateSettings({ volume })
  }

  if (!isSupported) {
    return (
      <div className={`p-3 bg-gray-100 rounded-lg ${className}`}>
        <p className="text-sm text-gray-600 text-center">
          Text-to-speech not supported in this browser
        </p>
      </div>
    )
  }

  return (
    <div className={`space-y-3 ${className}`}>
      {/* Main Controls */}
      <div className="flex items-center space-x-2">
        {/* Speak/Stop Button */}
        <button
          onClick={isSpeaking ? stop : handleSpeak}
          disabled={!text.trim()}
          className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
            isSpeaking
              ? 'bg-red-600 hover:bg-red-700 text-white'
              : 'bg-green-600 hover:bg-green-700 text-white disabled:bg-gray-300 disabled:cursor-not-allowed'
          }`}
          title={isSpeaking ? 'Stop speaking' : 'Speak text'}
        >
          {isSpeaking ? (
            <>
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clipRule="evenodd" />
              </svg>
              <span>Stop</span>
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 14.142M5 7h4l5-5v20l-5-5H5a1 1 0 01-1-1V8a1 1 0 011-1z" />
              </svg>
              <span>Speak</span>
            </>
          )}
        </button>

        {/* Pause/Resume Button */}
        {isSpeaking && (
          <button
            onClick={isPaused ? resume : pause}
            className="flex items-center space-x-1 px-3 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg font-medium transition-colors"
            title={isPaused ? 'Resume speaking' : 'Pause speaking'}
          >
            {isPaused ? (
              <>
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
                </svg>
                <span>Resume</span>
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
                <span>Pause</span>
              </>
            )}
          </button>
        )}

        {/* Settings Toggle */}
        {showAdvancedControls && (
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="flex items-center space-x-1 px-3 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg font-medium transition-colors"
            title="Voice settings"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            <span>Settings</span>
          </button>
        )}
      </div>

      {/* Status Display */}
      {(isSpeaking || isPaused || error) && (
        <div className="flex items-center space-x-2 text-sm">
          {isSpeaking && !isPaused && (
            <div className="flex items-center space-x-2 text-green-600">
              <div className="w-2 h-2 bg-green-600 rounded-full animate-pulse"></div>
              <span>Speaking...</span>
            </div>
          )}
          {isPaused && (
            <div className="flex items-center space-x-2 text-yellow-600">
              <div className="w-2 h-2 bg-yellow-600 rounded-full"></div>
              <span>Paused</span>
            </div>
          )}
          {error && (
            <div className="flex items-center space-x-2 text-red-600">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <span>{error}</span>
            </div>
          )}
        </div>
      )}

      {/* Advanced Settings */}
      {showSettings && showAdvancedControls && (
        <div className="p-4 bg-gray-50 rounded-lg space-y-4">
          <h4 className="font-medium text-gray-800">Voice Settings</h4>
          
          {/* Voice Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Voice
            </label>
            <select
              value={settings.voice?.voiceURI || ''}
              onChange={(e) => handleVoiceChange(e.target.value)}
              className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="">Default Voice</option>
              {voices
                .filter(voice => voice.lang.startsWith('en'))
                .map((voice) => (
                  <option key={voice.voiceURI} value={voice.voiceURI}>
                    {voice.name} ({voice.lang})
                  </option>
                ))}
            </select>
          </div>

          {/* Rate Control */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Speed: {settings.rate.toFixed(1)}x
            </label>
            <input
              type="range"
              min="0.5"
              max="2"
              step="0.1"
              value={settings.rate}
              onChange={(e) => handleRateChange(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>Slow</span>
              <span>Normal</span>
              <span>Fast</span>
            </div>
          </div>

          {/* Pitch Control */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Pitch: {settings.pitch.toFixed(1)}
            </label>
            <input
              type="range"
              min="0.5"
              max="2"
              step="0.1"
              value={settings.pitch}
              onChange={(e) => handlePitchChange(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>Low</span>
              <span>Normal</span>
              <span>High</span>
            </div>
          </div>

          {/* Volume Control */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Volume: {Math.round(settings.volume * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={settings.volume}
              onChange={(e) => handleVolumeChange(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>Quiet</span>
              <span>Normal</span>
              <span>Loud</span>
            </div>
          </div>

          {/* Test Button */}
          <button
            onClick={() => speak('This is a test of the selected voice settings.')}
            disabled={isSpeaking}
            className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white rounded-lg font-medium transition-colors"
          >
            Test Voice
          </button>
        </div>
      )}
    </div>
  )
}

export default SpeechControls
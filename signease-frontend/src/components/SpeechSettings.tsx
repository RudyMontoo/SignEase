/**
 * Speech Settings Component
 * =========================
 * 
 * Dedicated component for managing text-to-speech settings
 */

import React, { useState } from 'react'
import { useSpeech } from '../hooks/useSpeech'
import { getBrowserSpeechInfo, getSpeechPresets } from '../utils/speechUtils'

export interface SpeechSettingsProps {
  className?: string
}

export const SpeechSettings: React.FC<SpeechSettingsProps> = ({
  className = ''
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false)
  const {
    isSupported,
    voices,
    settings,
    speak,
    updateSettings
  } = useSpeech()

  const browserInfo = getBrowserSpeechInfo()
  const presets = getSpeechPresets()

  const handlePresetChange = (presetName: keyof typeof presets) => {
    const preset = presets[presetName]
    updateSettings(preset)
  }

  const handleVoiceChange = (voiceURI: string) => {
    const selectedVoice = voices.find(voice => voice.voiceURI === voiceURI)
    updateSettings({ voice: selectedVoice || null })
  }

  const testVoice = () => {
    speak('Hello! This is a test of your current voice settings. How does this sound?')
  }

  if (!isSupported) {
    return (
      <div className={`p-4 bg-red-50 border border-red-200 rounded-lg ${className}`}>
        <div className="flex items-center space-x-2 text-red-700">
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
          <div>
            <p className="font-medium">Text-to-Speech Not Available</p>
            <p className="text-sm">Your browser doesn't support text-to-speech functionality.</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className={`bg-white rounded-lg shadow-lg ${className}`}>
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-800">Speech Settings</h3>
          <div className="flex items-center space-x-2">
            <span className="text-xs text-gray-500">{browserInfo.browser}</span>
            <div className={`w-2 h-2 rounded-full ${
              browserInfo.speechSupport === 'excellent' ? 'bg-green-500' :
              browserInfo.speechSupport === 'good' ? 'bg-yellow-500' : 'bg-red-500'
            }`} />
          </div>
        </div>
      </div>

      <div className="p-6 space-y-6">
        {/* Quick Presets */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-3">
            Quick Presets
          </label>
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(presets).map(([name]) => (
              <button
                key={name}
                onClick={() => handlePresetChange(name as keyof typeof presets)}
                className="px-3 py-2 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors capitalize"
              >
                {name}
              </button>
            ))}
          </div>
        </div>

        {/* Voice Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Voice ({voices.length} available)
          </label>
          <select
            value={settings.voice?.voiceURI || ''}
            onChange={(e) => handleVoiceChange(e.target.value)}
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="">System Default</option>
            {voices
              .filter(voice => voice.lang.startsWith('en'))
              .map((voice) => (
                <option key={voice.voiceURI} value={voice.voiceURI}>
                  {voice.name} ({voice.lang}) {voice.localService ? '🏠' : '☁️'}
                </option>
              ))}
          </select>
          <p className="text-xs text-gray-500 mt-1">
            🏠 = Local voice, ☁️ = Network voice
          </p>
        </div>

        {/* Advanced Controls Toggle */}
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-gray-700">Advanced Settings</span>
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
              showAdvanced ? 'bg-blue-600' : 'bg-gray-200'
            }`}
          >
            <span
              className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                showAdvanced ? 'translate-x-6' : 'translate-x-1'
              }`}
            />
          </button>
        </div>

        {/* Advanced Settings */}
        {showAdvanced && (
          <div className="space-y-4 p-4 bg-gray-50 rounded-lg">
            {/* Speed Control */}
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
                onChange={(e) => updateSettings({ rate: parseFloat(e.target.value) })}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>0.5x</span>
                <span>1.0x</span>
                <span>2.0x</span>
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
                onChange={(e) => updateSettings({ pitch: parseFloat(e.target.value) })}
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
                onChange={(e) => updateSettings({ volume: parseFloat(e.target.value) })}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
              </div>
            </div>
          </div>
        )}

        {/* Test Button */}
        <button
          onClick={testVoice}
          className="w-full py-3 px-4 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors flex items-center justify-center space-x-2"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 14.142M5 7h4l5-5v20l-5-5H5a1 1 0 01-1-1V8a1 1 0 011-1z" />
          </svg>
          <span>Test Voice Settings</span>
        </button>

        {/* Browser Info */}
        <div className="text-xs text-gray-500 space-y-1">
          <p>Browser: {browserInfo.browser} ({browserInfo.speechSupport} support)</p>
          <p>Available voices: {browserInfo.voiceCount}</p>
          <p>Current voice: {settings.voice?.name || 'System default'}</p>
        </div>
      </div>
    </div>
  )
}

export default SpeechSettings
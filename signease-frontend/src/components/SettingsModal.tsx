/**
 * Settings Modal Component
 * ========================
 * 
 * Advanced settings modal with detailed configuration options
 */

import React, { useState } from 'react'
import Button from './ui/Button'
import Switch from './ui/Switch'
import type { ControlPanelSettings } from './ControlPanel'

interface SettingsModalProps {
  isOpen: boolean
  onClose: () => void
  settings: ControlPanelSettings
  onSettingsChange: (settings: Partial<ControlPanelSettings>) => void
}

const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  settings,
  onSettingsChange
}) => {
  const [localSettings, setLocalSettings] = useState(settings)

  if (!isOpen) return null

  const handleChange = (key: keyof ControlPanelSettings, value: any) => {
    const newSettings = { ...localSettings, [key]: value }
    setLocalSettings(newSettings)
  }

  const handleSave = () => {
    onSettingsChange(localSettings)
    onClose()
  }

  const handleCancel = () => {
    setLocalSettings(settings) // Reset to original settings
    onClose()
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
            Advanced Settings
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Recognition Settings */}
          <div className="space-y-4">
            <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
              Recognition Settings
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Confidence Threshold */}
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Confidence Threshold: {Math.round(localSettings.confidenceThreshold * 100)}%
                </label>
                <input
                  type="range"
                  min="0.3"
                  max="0.95"
                  step="0.05"
                  value={localSettings.confidenceThreshold}
                  onChange={(e) => handleChange('confidenceThreshold', parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                />
                <p className="text-xs text-gray-500">
                  Higher values require more confident predictions
                </p>
              </div>

              {/* Stability Delay */}
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Stability Delay: {localSettings.stabilityDelay}ms
                </label>
                <input
                  type="range"
                  min="500"
                  max="3000"
                  step="100"
                  value={localSettings.stabilityDelay}
                  onChange={(e) => handleChange('stabilityDelay', parseInt(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                />
                <p className="text-xs text-gray-500">
                  Time to wait before confirming a gesture
                </p>
              </div>
            </div>
          </div>

          {/* Display Settings */}
          <div className="space-y-4">
            <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
              Display Settings
            </h3>
            
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    AR Text Overlay
                  </label>
                  <p className="text-xs text-gray-500">
                    Show floating text over detected hand
                  </p>
                </div>
                <Switch
                  checked={localSettings.arOverlayEnabled}
                  onChange={(checked) => handleChange('arOverlayEnabled', checked)}
                />
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Gesture Guide Overlay
                  </label>
                  <p className="text-xs text-gray-500">
                    Show ASL alphabet reference on screen
                  </p>
                </div>
                <Switch
                  checked={localSettings.gestureGuideEnabled}
                  onChange={(checked) => handleChange('gestureGuideEnabled', checked)}
                />
              </div>
            </div>
          </div>

          {/* Interaction Settings */}
          <div className="space-y-4">
            <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
              Interaction Settings
            </h3>
            
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Auto-Add Gestures
                  </label>
                  <p className="text-xs text-gray-500">
                    Automatically add recognized gestures to sentence
                  </p>
                </div>
                <Switch
                  checked={localSettings.autoAddEnabled}
                  onChange={(checked) => handleChange('autoAddEnabled', checked)}
                />
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Text-to-Speech
                  </label>
                  <p className="text-xs text-gray-500">
                    Enable voice synthesis for recognized text
                  </p>
                </div>
                <Switch
                  checked={localSettings.speechEnabled}
                  onChange={(checked) => handleChange('speechEnabled', checked)}
                />
              </div>
            </div>
          </div>

          {/* Performance Settings */}
          <div className="space-y-4">
            <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
              Performance Settings
            </h3>
            
            <div className="space-y-3">
              <div>
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 block">
                  Performance Mode
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { value: 'quality', label: 'Quality', desc: 'Best accuracy, slower' },
                    { value: 'balanced', label: 'Balanced', desc: 'Good balance' },
                    { value: 'performance', label: 'Performance', desc: 'Fastest, lower accuracy' }
                  ].map((mode) => (
                    <button
                      key={mode.value}
                      onClick={() => handleChange('performanceMode', mode.value)}
                      className={`p-3 rounded-lg border text-left transition-colors ${
                        localSettings.performanceMode === mode.value
                          ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                          : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                      }`}
                    >
                      <div className="font-medium text-sm">{mode.label}</div>
                      <div className="text-xs text-gray-500">{mode.desc}</div>
                    </button>
                  ))}
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Debug Mode
                  </label>
                  <p className="text-xs text-gray-500">
                    Show detailed performance and debugging information
                  </p>
                </div>
                <Switch
                  checked={localSettings.debugMode}
                  onChange={(checked) => handleChange('debugMode', checked)}
                />
              </div>
            </div>
          </div>

          {/* Preset Configurations */}
          <div className="space-y-4">
            <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
              Quick Presets
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <Button
                onClick={() => {
                  const preset = {
                    confidenceThreshold: 0.9,
                    stabilityDelay: 1500,
                    performanceMode: 'quality' as const,
                    autoAddEnabled: false
                  }
                  setLocalSettings({ ...localSettings, ...preset })
                }}
                variant="secondary"
                size="sm"
                fullWidth
              >
                High Accuracy
              </Button>
              
              <Button
                onClick={() => {
                  const preset = {
                    confidenceThreshold: 0.7,
                    stabilityDelay: 1000,
                    performanceMode: 'balanced' as const,
                    autoAddEnabled: true
                  }
                  setLocalSettings({ ...localSettings, ...preset })
                }}
                variant="secondary"
                size="sm"
                fullWidth
              >
                Balanced
              </Button>
              
              <Button
                onClick={() => {
                  const preset = {
                    confidenceThreshold: 0.5,
                    stabilityDelay: 500,
                    performanceMode: 'performance' as const,
                    autoAddEnabled: true
                  }
                  setLocalSettings({ ...localSettings, ...preset })
                }}
                variant="secondary"
                size="sm"
                fullWidth
              >
                Fast Response
              </Button>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end space-x-3 p-6 border-t border-gray-200 dark:border-gray-700">
          <Button
            onClick={handleCancel}
            variant="secondary"
            size="md"
          >
            Cancel
          </Button>
          <Button
            onClick={handleSave}
            variant="primary"
            size="md"
          >
            Save Settings
          </Button>
        </div>
      </div>
    </div>
  )
}

export default SettingsModal
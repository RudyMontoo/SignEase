/**
 * Control Panel Component
 * ======================
 * 
 * Main settings panel with toggles and controls for the SignEase application
 */

import React, { useState, useEffect } from 'react'
import Card, { CardHeader, CardTitle, CardContent } from './ui/Card'
import Button from './ui/Button'
import Switch from './ui/Switch'
import Badge from './ui/Badge'
import SettingsModal from './SettingsModal'
import GestureGuide from './GestureGuide'

export interface ControlPanelSettings {
  confidenceThreshold: number
  stabilityDelay: number
  autoAddEnabled: boolean
  arOverlayEnabled: boolean
  speechEnabled: boolean
  debugMode: boolean
  performanceMode: 'balanced' | 'performance' | 'quality'
  gestureGuideEnabled: boolean
}

interface ControlPanelProps {
  settings: ControlPanelSettings
  onSettingsChange: (settings: Partial<ControlPanelSettings>) => void
  isConnected: boolean
  totalPredictions: number
  averageResponseTime: number
  currentFPS: number
}

const DEFAULT_SETTINGS: ControlPanelSettings = {
  confidenceThreshold: 0.7,
  stabilityDelay: 1000,
  autoAddEnabled: true,
  arOverlayEnabled: true,
  speechEnabled: true,
  debugMode: false,
  performanceMode: 'balanced',
  gestureGuideEnabled: false
}

const ControlPanel: React.FC<ControlPanelProps> = ({
  settings,
  onSettingsChange,
  isConnected,
  totalPredictions,
  averageResponseTime,
  currentFPS
}) => {
  const [showSettingsModal, setShowSettingsModal] = useState(false)
  const [showGestureGuide, setShowGestureGuide] = useState(false)
  const [localSettings, setLocalSettings] = useState<ControlPanelSettings>(settings)

  // Load settings from localStorage on mount
  useEffect(() => {
    const savedSettings = localStorage.getItem('signease-settings')
    if (savedSettings) {
      try {
        const parsed = JSON.parse(savedSettings)
        const mergedSettings = { ...DEFAULT_SETTINGS, ...parsed }
        setLocalSettings(mergedSettings)
        onSettingsChange(mergedSettings)
      } catch (error) {
        console.error('Failed to load settings:', error)
      }
    }
  }, [])

  // Save settings to localStorage when they change
  useEffect(() => {
    localStorage.setItem('signease-settings', JSON.stringify(localSettings))
  }, [localSettings])

  const handleSettingChange = (key: keyof ControlPanelSettings, value: any) => {
    const newSettings = { ...localSettings, [key]: value }
    setLocalSettings(newSettings)
    onSettingsChange({ [key]: value })
  }

  const resetSettings = () => {
    setLocalSettings(DEFAULT_SETTINGS)
    onSettingsChange(DEFAULT_SETTINGS)
    localStorage.removeItem('signease-settings')
  }

  const getPerformanceStatus = () => {
    if (currentFPS >= 25) return { status: 'excellent', color: 'text-green-600' }
    if (currentFPS >= 20) return { status: 'good', color: 'text-blue-600' }
    if (currentFPS >= 15) return { status: 'fair', color: 'text-yellow-600' }
    return { status: 'poor', color: 'text-red-600' }
  }

  const performanceStatus = getPerformanceStatus()

  return (
    <>
      <Card padding="lg" shadow="lg">
        <CardHeader>
          <CardTitle>Control Panel</CardTitle>
          <div className="flex space-x-2">
            <Button
              onClick={() => setShowSettingsModal(true)}
              variant="secondary"
              size="sm"
            >
              Advanced Settings
            </Button>
            <Button
              onClick={() => setShowGestureGuide(true)}
              variant="primary"
              size="sm"
            >
              Gesture Guide
            </Button>
          </div>
        </CardHeader>
        
        <CardContent>
          <div className="space-y-4">
            {/* Quick Settings */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  AR Text Overlay
                </label>
                <Switch
                  checked={localSettings.arOverlayEnabled}
                  onChange={(checked) => handleSettingChange('arOverlayEnabled', checked)}
                />
              </div>
              
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Auto-Add Gestures
                </label>
                <Switch
                  checked={localSettings.autoAddEnabled}
                  onChange={(checked) => handleSettingChange('autoAddEnabled', checked)}
                />
              </div>
              
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Text-to-Speech
                </label>
                <Switch
                  checked={localSettings.speechEnabled}
                  onChange={(checked) => handleSettingChange('speechEnabled', checked)}
                />
              </div>
              
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Debug Mode
                </label>
                <Switch
                  checked={localSettings.debugMode}
                  onChange={(checked) => handleSettingChange('debugMode', checked)}
                />
              </div>
            </div>

            {/* Confidence Threshold Slider */}
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
                onChange={(e) => handleSettingChange('confidenceThreshold', parseFloat(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>Less Strict (30%)</span>
                <span>More Strict (95%)</span>
              </div>
            </div>

            {/* Performance Mode */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Performance Mode
              </label>
              <div className="flex space-x-2">
                {(['balanced', 'performance', 'quality'] as const).map((mode) => (
                  <Button
                    key={mode}
                    onClick={() => handleSettingChange('performanceMode', mode)}
                    variant={localSettings.performanceMode === mode ? 'primary' : 'secondary'}
                    size="sm"
                  >
                    {mode.charAt(0).toUpperCase() + mode.slice(1)}
                  </Button>
                ))}
              </div>
            </div>

            {/* Performance Metrics */}
            <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 space-y-3">
              <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300">Performance Metrics</h4>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">Connection:</span>
                  <div className="flex items-center space-x-1">
                    <div className={`w-2 h-2 rounded-full ${
                      isConnected ? 'bg-green-500' : 'bg-red-500'
                    }`} />
                    <span className={isConnected ? 'text-green-600' : 'text-red-600'}>
                      {isConnected ? 'Online' : 'Offline'}
                    </span>
                  </div>
                </div>
                
                <div>
                  <span className="text-gray-500">FPS:</span>
                  <div className={`font-medium ${performanceStatus.color}`}>
                    {currentFPS} ({performanceStatus.status})
                  </div>
                </div>
                
                <div>
                  <span className="text-gray-500">Predictions:</span>
                  <div className="font-medium text-purple-600">
                    {totalPredictions.toLocaleString()}
                  </div>
                </div>
                
                <div>
                  <span className="text-gray-500">Avg Response:</span>
                  <div className="font-medium text-blue-600">
                    {Math.round(averageResponseTime)}ms
                  </div>
                </div>
              </div>
              
              {/* Performance Badges */}
              <div className="flex flex-wrap gap-2">
                {currentFPS >= 25 && <Badge variant="success">High FPS</Badge>}
                {averageResponseTime < 100 && <Badge variant="success">Fast Response</Badge>}
                {isConnected && <Badge variant="primary">Connected</Badge>}
                {localSettings.debugMode && <Badge variant="secondary">Debug Mode</Badge>}
              </div>
            </div>

            {/* Quick Actions */}
            <div className="flex flex-wrap gap-2">
              <Button
                onClick={resetSettings}
                variant="secondary"
                size="sm"
              >
                Reset to Defaults
              </Button>
              
              <Button
                onClick={() => {
                  const settings = {
                    timestamp: new Date().toISOString(),
                    settings: localSettings,
                    performance: {
                      fps: currentFPS,
                      responseTime: averageResponseTime,
                      predictions: totalPredictions,
                      connected: isConnected
                    }
                  }
                  
                  const blob = new Blob([JSON.stringify(settings, null, 2)], {
                    type: 'application/json'
                  })
                  
                  const url = URL.createObjectURL(blob)
                  const a = document.createElement('a')
                  a.href = url
                  a.download = `signease-settings-${Date.now()}.json`
                  a.click()
                  URL.revokeObjectURL(url)
                }}
                variant="primary"
                size="sm"
              >
                Export Settings
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={showSettingsModal}
        onClose={() => setShowSettingsModal(false)}
        settings={localSettings}
        onSettingsChange={(newSettings) => {
          const updated = { ...localSettings, ...newSettings }
          setLocalSettings(updated)
          onSettingsChange(newSettings)
        }}
      />

      {/* Gesture Guide Modal */}
      <GestureGuide
        isOpen={showGestureGuide}
        onClose={() => setShowGestureGuide(false)}
        enabled={localSettings.gestureGuideEnabled}
        onEnabledChange={(enabled) => handleSettingChange('gestureGuideEnabled', enabled)}
      />
    </>
  )
}

export default ControlPanel
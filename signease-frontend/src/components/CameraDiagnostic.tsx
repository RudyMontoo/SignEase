import React, { useState, useEffect } from 'react'

interface CameraInfo {
  deviceId: string
  label: string
  kind: string
}

const CameraDiagnostic: React.FC = () => {
  const [cameras, setCameras] = useState<CameraInfo[]>([])
  const [permissions, setPermissions] = useState<string>('unknown')
  const [browserSupport, setBrowserSupport] = useState<boolean>(false)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    checkCameraSupport()
  }, [])

  const checkCameraSupport = async () => {
    setIsLoading(true)
    setError(null)

    try {
      // Check browser support
      const hasGetUserMedia = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)
      setBrowserSupport(hasGetUserMedia)

      if (!hasGetUserMedia) {
        setError('Your browser does not support camera access')
        setIsLoading(false)
        return
      }

      // Check permissions
      try {
        const permissionStatus = await navigator.permissions.query({ name: 'camera' as PermissionName })
        setPermissions(permissionStatus.state)
        
        permissionStatus.onchange = () => {
          setPermissions(permissionStatus.state)
        }
      } catch (e) {
        console.log('Permission API not supported')
      }

      // Get available cameras
      try {
        const devices = await navigator.mediaDevices.enumerateDevices()
        const videoDevices = devices
          .filter(device => device.kind === 'videoinput')
          .map(device => ({
            deviceId: device.deviceId,
            label: device.label || `Camera ${device.deviceId.slice(0, 8)}`,
            kind: device.kind
          }))
        
        setCameras(videoDevices)
      } catch (e) {
        console.error('Error enumerating devices:', e)
        setError('Unable to detect cameras')
      }

    } catch (e) {
      console.error('Camera diagnostic error:', e)
      setError('Camera diagnostic failed')
    }

    setIsLoading(false)
  }

  const requestCameraPermission = async () => {
    try {
      setError(null)
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 640 }, 
          height: { ideal: 480 },
          facingMode: 'user'
        } 
      })
      
      // Stop the stream immediately - we just wanted permission
      stream.getTracks().forEach(track => track.stop())
      
      // Refresh the diagnostic
      await checkCameraSupport()
      
      // Reload the page to refresh camera access
      window.location.reload()
      
    } catch (e: any) {
      console.error('Permission request failed:', e)
      
      let errorMessage = 'Camera permission denied'
      if (e.name === 'NotFoundError') {
        errorMessage = 'No camera found on this device'
      } else if (e.name === 'NotReadableError') {
        errorMessage = 'Camera is being used by another application'
      } else if (e.name === 'OverconstrainedError') {
        errorMessage = 'Camera constraints not supported'
      }
      
      setError(errorMessage)
    }
  }

  const getPermissionIcon = () => {
    switch (permissions) {
      case 'granted':
        return '✅'
      case 'denied':
        return '❌'
      case 'prompt':
        return '❓'
      default:
        return '⚠️'
    }
  }

  const getPermissionColor = () => {
    switch (permissions) {
      case 'granted':
        return 'text-green-600'
      case 'denied':
        return 'text-red-600'
      case 'prompt':
        return 'text-yellow-600'
      default:
        return 'text-gray-600'
    }
  }

  if (isLoading) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-lg">
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-3">Checking camera status...</span>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white p-6 rounded-lg shadow-lg max-w-md mx-auto">
      <h3 className="text-lg font-semibold mb-4 flex items-center">
        🎥 Camera Diagnostic
      </h3>

      {/* Browser Support */}
      <div className="mb-4">
        <div className="flex items-center justify-between">
          <span>Browser Support:</span>
          <span className={browserSupport ? 'text-green-600' : 'text-red-600'}>
            {browserSupport ? '✅ Supported' : '❌ Not Supported'}
          </span>
        </div>
      </div>

      {/* Permissions */}
      <div className="mb-4">
        <div className="flex items-center justify-between">
          <span>Camera Permission:</span>
          <span className={getPermissionColor()}>
            {getPermissionIcon()} {permissions}
          </span>
        </div>
      </div>

      {/* Available Cameras */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <span>Available Cameras:</span>
          <span className="text-blue-600">{cameras.length} found</span>
        </div>
        {cameras.length > 0 ? (
          <ul className="text-sm text-gray-600 space-y-1">
            {cameras.map((camera, index) => (
              <li key={camera.deviceId} className="truncate">
                📹 {camera.label || `Camera ${index + 1}`}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-gray-500">No cameras detected</p>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded">
          <p className="text-red-800 text-sm">{error}</p>
        </div>
      )}

      {/* Action Buttons */}
      <div className="space-y-2">
        {permissions !== 'granted' && (
          <button
            onClick={requestCameraPermission}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded transition-colors"
          >
            Request Camera Permission
          </button>
        )}
        
        <button
          onClick={checkCameraSupport}
          className="w-full bg-gray-600 hover:bg-gray-700 text-white py-2 px-4 rounded transition-colors"
        >
          Refresh Diagnostic
        </button>
      </div>

      {/* Instructions */}
      <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded">
        <p className="text-blue-800 text-sm">
          <strong>Troubleshooting:</strong>
        </p>
        <ul className="text-blue-700 text-xs mt-1 space-y-1">
          <li>• Click the camera icon in your browser's address bar</li>
          <li>• Select "Allow" for camera access</li>
          <li>• Make sure no other apps are using your camera</li>
          <li>• Try refreshing the page after granting permission</li>
        </ul>
      </div>
    </div>
  )
}

export default CameraDiagnostic
/**
 * Camera Test Component
 * ====================
 * 
 * Simple component to test camera access and diagnose issues
 */

import { useRef, useEffect, useState } from 'react'

const CameraTest = () => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [status, setStatus] = useState<string>('Initializing...')
  const [error, setError] = useState<string | null>(null)
  const [permissions, setPermissions] = useState<string>('Unknown')

  useEffect(() => {
    testCameraAccess()
  }, [])

  const testCameraAccess = async () => {
    try {
      setStatus('Checking permissions...')
      
      // Check if getUserMedia is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('getUserMedia not supported in this browser')
      }

      // Check permissions
      try {
        const permissionStatus = await navigator.permissions.query({ name: 'camera' as PermissionName })
        setPermissions(permissionStatus.state)
      } catch (e) {
        setPermissions('Unable to check')
      }

      setStatus('Requesting camera access...')
      
      // Request camera access
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        }
      })

      setStatus('Camera access granted!')
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.onloadedmetadata = () => {
          setStatus('Camera stream active')
          setError(null)
        }
      }

    } catch (err) {
      console.error('Camera test error:', err)
      setError(err instanceof Error ? err.message : 'Unknown camera error')
      setStatus('Camera access failed')
    }
  }

  const retryCamera = () => {
    setError(null)
    setStatus('Retrying...')
    testCameraAccess()
  }

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg max-w-md mx-auto">
      <h3 className="text-lg font-semibold mb-4">Camera Diagnostic Test</h3>
      
      <div className="space-y-3 mb-4">
        <div className="flex justify-between">
          <span>Status:</span>
          <span className={error ? 'text-red-600' : 'text-green-600'}>{status}</span>
        </div>
        <div className="flex justify-between">
          <span>Permissions:</span>
          <span className={permissions === 'granted' ? 'text-green-600' : 'text-yellow-600'}>
            {permissions}
          </span>
        </div>
        <div className="flex justify-between">
          <span>HTTPS:</span>
          <span className={location.protocol === 'https:' || location.hostname === 'localhost' ? 'text-green-600' : 'text-red-600'}>
            {location.protocol === 'https:' || location.hostname === 'localhost' ? 'OK' : 'Required'}
          </span>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded p-3 mb-4">
          <p className="text-red-800 text-sm">{error}</p>
        </div>
      )}

      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full h-48 bg-gray-200 rounded mb-4"
        style={{ transform: 'scaleX(-1)' }}
      />

      <button
        onClick={retryCamera}
        className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded"
      >
        Test Camera Access
      </button>
    </div>
  )
}

export default CameraTest
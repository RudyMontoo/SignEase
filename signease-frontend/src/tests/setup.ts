/**
 * Test Setup
 * ==========
 * 
 * Global test configuration and mocks
 */

import { vi } from 'vitest'
import '@testing-library/jest-dom'
import React from 'react'

// Mock MediaDevices API
Object.defineProperty(navigator, 'mediaDevices', {
  writable: true,
  value: {
    getUserMedia: vi.fn().mockResolvedValue({
      getTracks: () => [{ stop: vi.fn() }],
      getVideoTracks: () => [{ stop: vi.fn() }],
      getAudioTracks: () => []
    }),
    enumerateDevices: vi.fn().mockResolvedValue([
      { deviceId: 'camera1', kind: 'videoinput', label: 'Mock Camera' }
    ])
  }
})

// Mock SpeechSynthesis API
Object.defineProperty(window, 'speechSynthesis', {
  writable: true,
  value: {
    speak: vi.fn(),
    cancel: vi.fn(),
    pause: vi.fn(),
    resume: vi.fn(),
    getVoices: vi.fn().mockReturnValue([
      { name: 'Mock Voice', lang: 'en-US', default: true }
    ]),
    speaking: false,
    pending: false,
    paused: false
  }
})

// Mock SpeechSynthesisUtterance
global.SpeechSynthesisUtterance = vi.fn().mockImplementation((text) => ({
  text,
  lang: 'en-US',
  voice: null,
  volume: 1,
  rate: 1,
  pitch: 1,
  onstart: null,
  onend: null,
  onerror: null,
  onpause: null,
  onresume: null,
  onmark: null,
  onboundary: null
}))

// Mock Performance API
Object.defineProperty(performance, 'memory', {
  writable: true,
  value: {
    usedJSHeapSize: 1000000,
    totalJSHeapSize: 2000000,
    jsHeapSizeLimit: 4000000
  }
})

// Mock PerformanceObserver
global.PerformanceObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  disconnect: vi.fn(),
  takeRecords: vi.fn().mockReturnValue([])
}))

// Mock ResizeObserver
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn()
}))

// Mock IntersectionObserver
global.IntersectionObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn()
}))

// Mock requestAnimationFrame
global.requestAnimationFrame = vi.fn().mockImplementation((cb) => {
  setTimeout(cb, 16) // ~60fps
  return 1
})

global.cancelAnimationFrame = vi.fn()

// Mock localStorage
const localStorageMock = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
  length: 0,
  key: vi.fn()
}

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock
})

// Mock sessionStorage
Object.defineProperty(window, 'sessionStorage', {
  value: localStorageMock
})

// Mock IndexedDB
const indexedDBMock = {
  open: vi.fn().mockImplementation(() => ({
    onsuccess: null,
    onerror: null,
    onupgradeneeded: null,
    result: {
      name: 'test-db',
      close: vi.fn(),
      createObjectStore: vi.fn(),
      objectStoreNames: { contains: vi.fn().mockReturnValue(false) }
    }
  })),
  deleteDatabase: vi.fn()
}

Object.defineProperty(window, 'indexedDB', {
  value: indexedDBMock
})

// Mock WebGL context
const webglMock = {
  getParameter: vi.fn().mockReturnValue('WebGL 1.0'),
  createShader: vi.fn(),
  shaderSource: vi.fn(),
  compileShader: vi.fn(),
  createProgram: vi.fn(),
  attachShader: vi.fn(),
  linkProgram: vi.fn(),
  useProgram: vi.fn(),
  getAttribLocation: vi.fn(),
  getUniformLocation: vi.fn(),
  enableVertexAttribArray: vi.fn(),
  vertexAttribPointer: vi.fn(),
  uniform1f: vi.fn(),
  uniform2f: vi.fn(),
  uniform3f: vi.fn(),
  uniform4f: vi.fn(),
  drawArrays: vi.fn(),
  clear: vi.fn(),
  clearColor: vi.fn(),
  enable: vi.fn(),
  disable: vi.fn(),
  viewport: vi.fn()
}

// Mock HTMLCanvasElement.getContext
HTMLCanvasElement.prototype.getContext = vi.fn().mockImplementation((contextType) => {
  if (contextType === '2d') {
    return {
      fillRect: vi.fn(),
      clearRect: vi.fn(),
      getImageData: vi.fn().mockReturnValue({
        data: new Uint8ClampedArray(4),
        width: 1,
        height: 1
      }),
      putImageData: vi.fn(),
      createImageData: vi.fn(),
      setTransform: vi.fn(),
      drawImage: vi.fn(),
      save: vi.fn(),
      restore: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      closePath: vi.fn(),
      stroke: vi.fn(),
      fill: vi.fn(),
      measureText: vi.fn().mockReturnValue({ width: 0 }),
      fillText: vi.fn(),
      strokeText: vi.fn()
    }
  }
  
  if (contextType === 'webgl' || contextType === 'experimental-webgl' || contextType === 'webgl2') {
    return webglMock
  }
  
  return null
})

// Mock HTMLVideoElement
Object.defineProperty(HTMLVideoElement.prototype, 'play', {
  writable: true,
  value: vi.fn().mockResolvedValue(undefined)
})

Object.defineProperty(HTMLVideoElement.prototype, 'pause', {
  writable: true,
  value: vi.fn()
})

Object.defineProperty(HTMLVideoElement.prototype, 'load', {
  writable: true,
  value: vi.fn()
})

// Mock URL.createObjectURL
global.URL.createObjectURL = vi.fn().mockReturnValue('mock-object-url')
global.URL.revokeObjectURL = vi.fn()

// Mock Blob
const MockBlob = function(content: any, options: any) {
  return {
    size: content ? content.length : 0,
    type: options?.type || '',
    arrayBuffer: vi.fn().mockResolvedValue(new ArrayBuffer(0)),
    text: vi.fn().mockResolvedValue(''),
    stream: vi.fn()
  }
}
global.Blob = MockBlob as any

// Mock File
const MockFile = function(content: any, name: string, options: any) {
  return {
    ...MockBlob(content, options),
    name,
    lastModified: Date.now()
  }
}
global.File = MockFile as any

// Mock fetch
global.fetch = vi.fn().mockResolvedValue({
  ok: true,
  status: 200,
  json: vi.fn().mockResolvedValue({}),
  text: vi.fn().mockResolvedValue(''),
  blob: vi.fn().mockResolvedValue(MockBlob([], {})),
  arrayBuffer: vi.fn().mockResolvedValue(new ArrayBuffer(0))
})

// Mock console methods for cleaner test output
const originalConsole = { ...console }
global.console = {
  ...console,
  log: vi.fn(),
  warn: vi.fn(),
  error: vi.fn(),
  info: vi.fn(),
  debug: vi.fn()
}

// Restore console for test debugging when needed
export const restoreConsole = () => {
  global.console = originalConsole
}

// Mock MediaPipe (if used)
vi.mock('@mediapipe/hands', () => ({
  Hands: vi.fn().mockImplementation(() => ({
    setOptions: vi.fn(),
    onResults: vi.fn(),
    send: vi.fn().mockResolvedValue(undefined),
    close: vi.fn()
  })),
  Camera: vi.fn().mockImplementation(() => ({
    start: vi.fn().mockResolvedValue(undefined),
    stop: vi.fn()
  })),
  HAND_CONNECTIONS: []
}))

// Mock React Router (if used)
vi.mock('react-router-dom', () => ({
  useNavigate: () => vi.fn(),
  useLocation: () => ({ pathname: '/' }),
  BrowserRouter: ({ children }: { children: React.ReactNode }) => children,
  Routes: ({ children }: { children: React.ReactNode }) => children,
  Route: ({ element }: { element: React.ReactNode }) => element,
  Link: ({ children, to }: { children: React.ReactNode; to: string }) => 
    React.createElement('a', { href: to }, children)
}))

// Setup cleanup
afterEach(() => {
  vi.clearAllMocks()
  localStorageMock.clear()
})

// Global test utilities
export const mockGestureAPI = {
  predict: vi.fn().mockResolvedValue({
    prediction: 'A',
    confidence: 0.95,
    alternatives: []
  }),
  health: vi.fn().mockResolvedValue({
    status: 'healthy',
    latency: 50
  })
}

export const mockMediaPipeResults = {
  multiHandLandmarks: [[
    { x: 0.5, y: 0.5, z: 0 },
    // ... 20 more landmarks
  ]],
  multiHandedness: [{
    index: 0,
    score: 0.95,
    label: 'Right'
  }]
}

export const createMockVideoElement = () => {
  const video = document.createElement('video')
  Object.defineProperty(video, 'videoWidth', { value: 640, writable: true })
  Object.defineProperty(video, 'videoHeight', { value: 480, writable: true })
  Object.defineProperty(video, 'readyState', { value: 4, writable: true }) // HAVE_ENOUGH_DATA
  return video
}

export const createMockCanvas = () => {
  const canvas = document.createElement('canvas')
  canvas.width = 640
  canvas.height = 480
  return canvas
}
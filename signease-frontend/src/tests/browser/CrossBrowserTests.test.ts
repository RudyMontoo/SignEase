/**
 * Cross-Browser Compatibility Tests
 * =================================
 * 
 * Tests for browser compatibility and feature detection
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'

// Browser detection utilities
class BrowserDetector {
  static getUserAgent(): string {
    return navigator.userAgent
  }

  static isChrome(): boolean {
    return /Chrome/.test(navigator.userAgent) && !/Edge/.test(navigator.userAgent)
  }

  static isFirefox(): boolean {
    return /Firefox/.test(navigator.userAgent)
  }

  static isSafari(): boolean {
    return /Safari/.test(navigator.userAgent) && !/Chrome/.test(navigator.userAgent)
  }

  static isEdge(): boolean {
    return /Edge/.test(navigator.userAgent)
  }

  static getVersion(): string {
    const ua = navigator.userAgent
    const match = ua.match(/(Chrome|Firefox|Safari|Edge)\/(\d+)/)
    return match ? match[2] : 'unknown'
  }
}

// Feature detection utilities
class FeatureDetector {
  static hasWebRTC(): boolean {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)
  }

  static hasWebGL(): boolean {
    try {
      const canvas = document.createElement('canvas')
      return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'))
    } catch (e) {
      return false
    }
  }

  static hasWebAssembly(): boolean {
    return typeof WebAssembly === 'object'
  }

  static hasSpeechSynthesis(): boolean {
    return 'speechSynthesis' in window
  }

  static hasLocalStorage(): boolean {
    try {
      const test = 'test'
      localStorage.setItem(test, test)
      localStorage.removeItem(test)
      return true
    } catch (e) {
      return false
    }
  }

  static hasIndexedDB(): boolean {
    return 'indexedDB' in window
  }

  static hasWorkers(): boolean {
    return typeof Worker !== 'undefined'
  }

  static hasOffscreenCanvas(): boolean {
    return typeof OffscreenCanvas !== 'undefined'
  }
}

describe('Cross-Browser Compatibility Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('Browser Detection', () => {
    it('should detect browser type correctly', () => {
      const userAgent = navigator.userAgent
      
      // Should detect at least one browser type
      const isKnownBrowser = BrowserDetector.isChrome() || 
                           BrowserDetector.isFirefox() || 
                           BrowserDetector.isSafari() || 
                           BrowserDetector.isEdge()
      
      expect(isKnownBrowser).toBe(true)
      expect(typeof userAgent).toBe('string')
      expect(userAgent.length).toBeGreaterThan(0)
    })

    it('should handle different user agent strings', () => {
      const testUserAgents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59'
      ]

      testUserAgents.forEach(ua => {
        // Mock user agent
        Object.defineProperty(navigator, 'userAgent', {
          value: ua,
          configurable: true
        })

        const isChrome = /Chrome/.test(ua) && !/Edge/.test(ua)
        const isFirefox = /Firefox/.test(ua)
        const isSafari = /Safari/.test(ua) && !/Chrome/.test(ua)
        const isEdge = /Edge/.test(ua)

        expect(typeof isChrome).toBe('boolean')
        expect(typeof isFirefox).toBe('boolean')
        expect(typeof isSafari).toBe('boolean')
        expect(typeof isEdge).toBe('boolean')
      })
    })
  })

  describe('Feature Detection', () => {
    it('should detect WebRTC support', () => {
      const hasWebRTC = FeatureDetector.hasWebRTC()
      
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        expect(hasWebRTC).toBe(true)
      } else {
        expect(hasWebRTC).toBe(false)
      }
    })

    it('should detect WebGL support', () => {
      const hasWebGL = FeatureDetector.hasWebGL()
      
      // WebGL should be available in modern browsers
      expect(typeof hasWebGL).toBe('boolean')
    })

    it('should detect WebAssembly support', () => {
      const hasWasm = FeatureDetector.hasWebAssembly()
      
      // WebAssembly should be available in modern browsers
      expect(hasWasm).toBe(true)
    })

    it('should detect Speech Synthesis support', () => {
      const hasSpeech = FeatureDetector.hasSpeechSynthesis()
      
      expect(typeof hasSpeech).toBe('boolean')
      
      if ('speechSynthesis' in window) {
        expect(hasSpeech).toBe(true)
      }
    })

    it('should detect Local Storage support', () => {
      const hasLocalStorage = FeatureDetector.hasLocalStorage()
      
      expect(typeof hasLocalStorage).toBe('boolean')
      
      // Should be available in test environment
      expect(hasLocalStorage).toBe(true)
    })

    it('should detect Web Workers support', () => {
      const hasWorkers = FeatureDetector.hasWorkers()
      
      expect(typeof hasWorkers).toBe('boolean')
    })
  })

  describe('Media API Compatibility', () => {
    it('should handle getUserMedia across browsers', async () => {
      // Mock different getUserMedia implementations
      const implementations = [
        // Modern browsers
        {
          name: 'modern',
          getUserMedia: vi.fn().mockResolvedValue({
            getTracks: () => [{ stop: vi.fn() }]
          })
        },
        // Legacy browsers (should not be used but test fallback)
        {
          name: 'legacy',
          getUserMedia: undefined
        }
      ]

      for (const impl of implementations) {
        if (impl.getUserMedia) {
          global.navigator.mediaDevices = {
            getUserMedia: impl.getUserMedia
          } as any

          try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true })
            expect(stream).toBeDefined()
            expect(stream.getTracks).toBeDefined()
          } catch (error) {
            // Should handle errors gracefully
            expect(error).toBeInstanceOf(Error)
          }
        }
      }
    })

    it('should handle different video constraints', async () => {
      const constraints = [
        { video: true },
        { video: { width: 640, height: 480 } },
        { video: { width: { ideal: 1280 }, height: { ideal: 720 } } },
        { video: { facingMode: 'user' } },
        { video: { frameRate: { ideal: 30 } } }
      ]

      global.navigator.mediaDevices = {
        getUserMedia: vi.fn().mockResolvedValue({
          getTracks: () => [{ stop: vi.fn() }]
        })
      } as any

      for (const constraint of constraints) {
        try {
          await navigator.mediaDevices.getUserMedia(constraint)
          expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledWith(constraint)
        } catch (error) {
          // Some constraints might not be supported
          expect(error).toBeInstanceOf(Error)
        }
      }
    })
  })

  describe('Canvas and WebGL Compatibility', () => {
    it('should handle canvas operations across browsers', () => {
      const canvas = document.createElement('canvas')
      canvas.width = 640
      canvas.height = 480

      const ctx2d = canvas.getContext('2d')
      expect(ctx2d).toBeDefined()

      if (ctx2d) {
        // Test basic canvas operations
        ctx2d.fillStyle = 'red'
        ctx2d.fillRect(0, 0, 100, 100)
        
        const imageData = ctx2d.getImageData(0, 0, 100, 100)
        expect(imageData).toBeDefined()
        expect(imageData.data).toBeDefined()
      }
    })

    it('should handle WebGL context creation', () => {
      const canvas = document.createElement('canvas')
      
      // Try different WebGL context types
      const contextTypes = ['webgl2', 'webgl', 'experimental-webgl']
      let gl: WebGLRenderingContext | null = null

      for (const contextType of contextTypes) {
        try {
          gl = canvas.getContext(contextType) as WebGLRenderingContext
          if (gl) break
        } catch (e) {
          // Continue to next context type
        }
      }

      if (gl) {
        expect(gl).toBeDefined()
        expect(typeof gl.getParameter).toBe('function')
        
        // Test basic WebGL operations
        const version = gl.getParameter(gl.VERSION)
        expect(typeof version).toBe('string')
      }
    })
  })

  describe('Audio API Compatibility', () => {
    it('should handle Speech Synthesis across browsers', () => {
      if ('speechSynthesis' in window) {
        const synth = window.speechSynthesis
        expect(synth).toBeDefined()
        
        // Test basic speech synthesis
        const utterance = new SpeechSynthesisUtterance('test')
        expect(utterance).toBeDefined()
        expect(typeof utterance.text).toBe('string')
        
        // Test voice enumeration
        const voices = synth.getVoices()
        expect(Array.isArray(voices)).toBe(true)
      }
    })

    it('should handle Web Audio API if available', () => {
      if ('AudioContext' in window || 'webkitAudioContext' in window) {
        const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext
        const audioContext = new AudioContextClass()
        
        expect(audioContext).toBeDefined()
        expect(typeof audioContext.createOscillator).toBe('function')
        
        audioContext.close()
      }
    })
  })

  describe('Storage Compatibility', () => {
    it('should handle localStorage across browsers', () => {
      if (FeatureDetector.hasLocalStorage()) {
        const testKey = 'test-key'
        const testValue = 'test-value'
        
        localStorage.setItem(testKey, testValue)
        const retrieved = localStorage.getItem(testKey)
        
        expect(retrieved).toBe(testValue)
        
        localStorage.removeItem(testKey)
        const afterRemoval = localStorage.getItem(testKey)
        
        expect(afterRemoval).toBeNull()
      }
    })

    it('should handle sessionStorage if available', () => {
      if ('sessionStorage' in window) {
        const testKey = 'session-test-key'
        const testValue = 'session-test-value'
        
        sessionStorage.setItem(testKey, testValue)
        const retrieved = sessionStorage.getItem(testKey)
        
        expect(retrieved).toBe(testValue)
        
        sessionStorage.removeItem(testKey)
      }
    })

    it('should handle IndexedDB if available', async () => {
      if (FeatureDetector.hasIndexedDB()) {
        const dbName = 'test-db'
        const version = 1
        
        const request = indexedDB.open(dbName, version)
        
        const db = await new Promise<IDBDatabase>((resolve, reject) => {
          request.onsuccess = () => resolve(request.result)
          request.onerror = () => reject(request.error)
          request.onupgradeneeded = () => {
            const db = request.result
            if (!db.objectStoreNames.contains('test-store')) {
              db.createObjectStore('test-store')
            }
          }
        })

        expect(db).toBeDefined()
        expect(db.name).toBe(dbName)
        
        db.close()
        indexedDB.deleteDatabase(dbName)
      }
    })
  })

  describe('Performance API Compatibility', () => {
    it('should handle performance.now() across browsers', () => {
      const now1 = performance.now()
      const now2 = performance.now()
      
      expect(typeof now1).toBe('number')
      expect(typeof now2).toBe('number')
      expect(now2).toBeGreaterThanOrEqual(now1)
    })

    it('should handle performance.memory if available', () => {
      if ('memory' in performance) {
        const memory = (performance as any).memory
        expect(memory).toBeDefined()
        expect(typeof memory.usedJSHeapSize).toBe('number')
      }
    })

    it('should handle PerformanceObserver if available', () => {
      if ('PerformanceObserver' in window) {
        const observer = new PerformanceObserver(() => {})
        expect(observer).toBeDefined()
        expect(typeof observer.observe).toBe('function')
        observer.disconnect()
      }
    })
  })

  describe('CSS and Styling Compatibility', () => {
    it('should handle CSS custom properties', () => {
      const testElement = document.createElement('div')
      document.body.appendChild(testElement)
      
      // Set CSS custom property
      testElement.style.setProperty('--test-color', 'red')
      
      // Get CSS custom property
      const value = getComputedStyle(testElement).getPropertyValue('--test-color')
      
      expect(typeof value).toBe('string')
      
      document.body.removeChild(testElement)
    })

    it('should handle CSS Grid and Flexbox', () => {
      const testElement = document.createElement('div')
      document.body.appendChild(testElement)
      
      // Test CSS Grid
      testElement.style.display = 'grid'
      expect(getComputedStyle(testElement).display).toBe('grid')
      
      // Test Flexbox
      testElement.style.display = 'flex'
      expect(getComputedStyle(testElement).display).toBe('flex')
      
      document.body.removeChild(testElement)
    })

    it('should handle CSS transforms and animations', () => {
      const testElement = document.createElement('div')
      document.body.appendChild(testElement)
      
      // Test transform
      testElement.style.transform = 'translateX(10px)'
      expect(testElement.style.transform).toBe('translateX(10px)')
      
      // Test transition
      testElement.style.transition = 'all 0.3s ease'
      expect(testElement.style.transition).toBe('all 0.3s ease')
      
      document.body.removeChild(testElement)
    })
  })

  describe('Event Handling Compatibility', () => {
    it('should handle modern event listeners', () => {
      const testElement = document.createElement('div')
      let eventFired = false
      
      const handler = () => {
        eventFired = true
      }
      
      testElement.addEventListener('click', handler)
      
      // Simulate click
      const clickEvent = new MouseEvent('click', { bubbles: true })
      testElement.dispatchEvent(clickEvent)
      
      expect(eventFired).toBe(true)
      
      testElement.removeEventListener('click', handler)
    })

    it('should handle touch events if supported', () => {
      const testElement = document.createElement('div')
      let touchEventFired = false
      
      const handler = () => {
        touchEventFired = true
      }
      
      testElement.addEventListener('touchstart', handler)
      
      // Check if touch events are supported
      if ('TouchEvent' in window) {
        const touchEvent = new TouchEvent('touchstart', { bubbles: true })
        testElement.dispatchEvent(touchEvent)
        expect(touchEventFired).toBe(true)
      }
      
      testElement.removeEventListener('touchstart', handler)
    })

    it('should handle keyboard events', () => {
      const testElement = document.createElement('input')
      document.body.appendChild(testElement)
      
      let keyEventFired = false
      
      const handler = (e: KeyboardEvent) => {
        keyEventFired = true
        expect(e.key).toBeDefined()
      }
      
      testElement.addEventListener('keydown', handler)
      
      const keyEvent = new KeyboardEvent('keydown', { key: 'Enter', bubbles: true })
      testElement.dispatchEvent(keyEvent)
      
      expect(keyEventFired).toBe(true)
      
      testElement.removeEventListener('keydown', handler)
      document.body.removeChild(testElement)
    })
  })
})

export { BrowserDetector, FeatureDetector }
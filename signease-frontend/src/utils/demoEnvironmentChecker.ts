/**
 * Demo Environment Checker
 * ========================
 * 
 * Comprehensive demo environment validation and setup
 * Ensures optimal presentation conditions and fallback strategies
 */

export interface DemoEnvironmentCheck {
  name: string
  status: 'pass' | 'warning' | 'fail'
  message: string
  recommendation?: string
  critical: boolean
}

export interface DemoEnvironmentReport {
  overall: 'ready' | 'needs_attention' | 'not_ready'
  checks: DemoEnvironmentCheck[]
  systemInfo: {
    browser: string
    version: string
    platform: string
    screenResolution: string
    deviceMemory?: number
    hardwareConcurrency: number
    connectionType?: string
    batteryLevel?: number
  }
  recommendations: string[]
  contingencyPlans: string[]
}

export interface DemoConfig {
  targetFPS: number
  minBatteryLevel: number
  requiredPermissions: string[]
  fallbackOptions: {
    enableOfflineMode: boolean
    useBackupVideo: boolean
    simplifiedUI: boolean
  }
}

class DemoEnvironmentChecker {
  private config: DemoConfig = {
    targetFPS: 30,
    minBatteryLevel: 20,
    requiredPermissions: ['camera', 'microphone'],
    fallbackOptions: {
      enableOfflineMode: true,
      useBackupVideo: true,
      simplifiedUI: false
    }
  }

  async runFullCheck(): Promise<DemoEnvironmentReport> {
    console.log('🎯 Running comprehensive demo environment check...')
    
    const checks: DemoEnvironmentCheck[] = []
    
    // System checks
    checks.push(...await this.checkSystemRequirements())
    
    // Browser checks
    checks.push(...await this.checkBrowserCapabilities())
    
    // Hardware checks
    checks.push(...await this.checkHardwareCapabilities())
    
    // Network checks
    checks.push(...await this.checkNetworkConditions())
    
    // Permission checks
    checks.push(...await this.checkPermissions())
    
    // Performance checks
    checks.push(...await this.checkPerformance())
    
    // Audio/Video checks
    checks.push(...await this.checkAudioVideoSetup())
    
    // Lighting and environment checks
    checks.push(...await this.checkEnvironmentalConditions())

    const systemInfo = await this.getSystemInfo()
    const overall = this.determineOverallStatus(checks)
    const recommendations = this.generateRecommendations(checks)
    const contingencyPlans = this.generateContingencyPlans(checks)

    const report: DemoEnvironmentReport = {
      overall,
      checks,
      systemInfo,
      recommendations,
      contingencyPlans
    }

    console.log(`🎯 Demo environment check complete: ${overall}`)
    return report
  }

  private async checkSystemRequirements(): Promise<DemoEnvironmentCheck[]> {
    const checks: DemoEnvironmentCheck[] = []

    // Check available memory
    if ('memory' in performance) {
      const memory = (performance as any).memory
      const availableMemory = memory.jsHeapSizeLimit / 1024 / 1024 // MB
      
      checks.push({
        name: 'Available Memory',
        status: availableMemory > 512 ? 'pass' : availableMemory > 256 ? 'warning' : 'fail',
        message: `${availableMemory.toFixed(0)}MB available`,
        recommendation: availableMemory < 512 ? 'Close other browser tabs and applications' : undefined,
        critical: availableMemory < 256
      })
    }

    // Check CPU cores
    const cores = navigator.hardwareConcurrency || 1
    checks.push({
      name: 'CPU Cores',
      status: cores >= 4 ? 'pass' : cores >= 2 ? 'warning' : 'fail',
      message: `${cores} cores available`,
      recommendation: cores < 4 ? 'Consider using a more powerful device for optimal performance' : undefined,
      critical: cores < 2
    })

    // Check device memory (if available)
    if ('deviceMemory' in navigator) {
      const deviceMemory = (navigator as any).deviceMemory
      checks.push({
        name: 'Device Memory',
        status: deviceMemory >= 4 ? 'pass' : deviceMemory >= 2 ? 'warning' : 'fail',
        message: `${deviceMemory}GB device memory`,
        recommendation: deviceMemory < 4 ? 'Device may struggle with complex operations' : undefined,
        critical: deviceMemory < 2
      })
    }

    return checks
  }

  private async checkBrowserCapabilities(): Promise<DemoEnvironmentCheck[]> {
    const checks: DemoEnvironmentCheck[] = []

    // Check WebRTC support
    const hasWebRTC = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)
    checks.push({
      name: 'WebRTC Support',
      status: hasWebRTC ? 'pass' : 'fail',
      message: hasWebRTC ? 'WebRTC supported' : 'WebRTC not supported',
      recommendation: !hasWebRTC ? 'Use a modern browser (Chrome, Firefox, Safari, Edge)' : undefined,
      critical: !hasWebRTC
    })

    // Check WebGL support
    const hasWebGL = this.checkWebGLSupport()
    checks.push({
      name: 'WebGL Support',
      status: hasWebGL ? 'pass' : 'warning',
      message: hasWebGL ? 'WebGL supported' : 'WebGL not supported',
      recommendation: !hasWebGL ? 'GPU acceleration unavailable, performance may be reduced' : undefined,
      critical: false
    })

    // Check WebAssembly support
    const hasWasm = typeof WebAssembly === 'object'
    checks.push({
      name: 'WebAssembly Support',
      status: hasWasm ? 'pass' : 'warning',
      message: hasWasm ? 'WebAssembly supported' : 'WebAssembly not supported',
      recommendation: !hasWasm ? 'Some optimizations unavailable' : undefined,
      critical: false
    })

    // Check Speech Synthesis support
    const hasSpeech = 'speechSynthesis' in window
    checks.push({
      name: 'Speech Synthesis',
      status: hasSpeech ? 'pass' : 'warning',
      message: hasSpeech ? 'Speech synthesis supported' : 'Speech synthesis not supported',
      recommendation: !hasSpeech ? 'Text-to-speech features will be unavailable' : undefined,
      critical: false
    })

    return checks
  }

  private async checkHardwareCapabilities(): Promise<DemoEnvironmentCheck[]> {
    const checks: DemoEnvironmentCheck[] = []

    // Check screen resolution
    const screenWidth = screen.width
    const screenHeight = screen.height
    const isHighRes = screenWidth >= 1920 && screenHeight >= 1080
    
    checks.push({
      name: 'Screen Resolution',
      status: isHighRes ? 'pass' : screenWidth >= 1280 ? 'warning' : 'fail',
      message: `${screenWidth}x${screenHeight}`,
      recommendation: !isHighRes ? 'Higher resolution recommended for better demo visibility' : undefined,
      critical: screenWidth < 1024
    })

    // Check battery level (if available)
    if ('getBattery' in navigator) {
      try {
        const battery = await (navigator as any).getBattery()
        const batteryLevel = Math.round(battery.level * 100)
        
        checks.push({
          name: 'Battery Level',
          status: batteryLevel > 50 ? 'pass' : batteryLevel > this.config.minBatteryLevel ? 'warning' : 'fail',
          message: `${batteryLevel}% battery`,
          recommendation: batteryLevel < 50 ? 'Connect to power source for demo' : undefined,
          critical: batteryLevel < this.config.minBatteryLevel
        })
      } catch (error) {
        // Battery API not available or blocked
      }
    }

    return checks
  }

  private async checkNetworkConditions(): Promise<DemoEnvironmentCheck[]> {
    const checks: DemoEnvironmentCheck[] = []

    // Check connection type
    if ('connection' in navigator) {
      const connection = (navigator as any).connection
      const effectiveType = connection.effectiveType
      
      checks.push({
        name: 'Network Connection',
        status: ['4g', 'wifi'].includes(effectiveType) ? 'pass' : 
                effectiveType === '3g' ? 'warning' : 'fail',
        message: `${effectiveType} connection`,
        recommendation: !['4g', 'wifi'].includes(effectiveType) ? 
          'Consider using offline mode or backup video for unreliable connections' : undefined,
        critical: effectiveType === 'slow-2g'
      })
    }

    // Test API connectivity
    try {
      const startTime = performance.now()
      const response = await fetch('/api/health', { 
        method: 'GET',
        cache: 'no-cache'
      })
      const endTime = performance.now()
      const latency = endTime - startTime

      checks.push({
        name: 'API Connectivity',
        status: response.ok && latency < 200 ? 'pass' : 
                response.ok && latency < 500 ? 'warning' : 'fail',
        message: response.ok ? `${latency.toFixed(0)}ms latency` : 'API unreachable',
        recommendation: !response.ok ? 'Enable offline mode or use backup demo' : 
                       latency > 200 ? 'High latency detected, consider local demo' : undefined,
        critical: !response.ok
      })
    } catch (error) {
      checks.push({
        name: 'API Connectivity',
        status: 'fail',
        message: 'API unreachable',
        recommendation: 'Enable offline mode or use backup demo',
        critical: true
      })
    }

    return checks
  }

  private async checkPermissions(): Promise<DemoEnvironmentCheck[]> {
    const checks: DemoEnvironmentCheck[] = []

    // Check camera permission
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      stream.getTracks().forEach(track => track.stop())
      
      checks.push({
        name: 'Camera Permission',
        status: 'pass',
        message: 'Camera access granted',
        critical: true
      })
    } catch (error) {
      checks.push({
        name: 'Camera Permission',
        status: 'fail',
        message: 'Camera access denied or unavailable',
        recommendation: 'Grant camera permission in browser settings',
        critical: true
      })
    }

    // Check microphone permission (for speech features)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      stream.getTracks().forEach(track => track.stop())
      
      checks.push({
        name: 'Microphone Permission',
        status: 'pass',
        message: 'Microphone access granted',
        critical: false
      })
    } catch (error) {
      checks.push({
        name: 'Microphone Permission',
        status: 'warning',
        message: 'Microphone access denied',
        recommendation: 'Grant microphone permission for full speech features',
        critical: false
      })
    }

    return checks
  }

  private async checkPerformance(): Promise<DemoEnvironmentCheck[]> {
    const checks: DemoEnvironmentCheck[] = []

    // Run performance benchmark
    const benchmark = await this.runPerformanceBenchmark()
    
    checks.push({
      name: 'Rendering Performance',
      status: benchmark.fps >= this.config.targetFPS ? 'pass' : 
              benchmark.fps >= this.config.targetFPS * 0.8 ? 'warning' : 'fail',
      message: `${benchmark.fps.toFixed(1)} FPS`,
      recommendation: benchmark.fps < this.config.targetFPS ? 
        'Close other applications and browser tabs for better performance' : undefined,
      critical: benchmark.fps < this.config.targetFPS * 0.6
    })

    checks.push({
      name: 'Memory Usage',
      status: benchmark.memoryUsage < 80 ? 'pass' : 
              benchmark.memoryUsage < 90 ? 'warning' : 'fail',
      message: `${benchmark.memoryUsage.toFixed(1)}% memory usage`,
      recommendation: benchmark.memoryUsage > 80 ? 
        'High memory usage detected, consider restarting browser' : undefined,
      critical: benchmark.memoryUsage > 95
    })

    return checks
  }

  private async checkAudioVideoSetup(): Promise<DemoEnvironmentCheck[]> {
    const checks: DemoEnvironmentCheck[] = []

    try {
      // Test camera quality
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 1280, height: 720 } 
      })
      
      const videoTrack = stream.getVideoTracks()[0]
      const settings = videoTrack.getSettings()
      
      checks.push({
        name: 'Camera Quality',
        status: (settings.width || 0) >= 720 ? 'pass' : 
                (settings.width || 0) >= 480 ? 'warning' : 'fail',
        message: `${settings.width}x${settings.height} @ ${settings.frameRate}fps`,
        recommendation: (settings.width || 0) < 720 ? 
          'Consider using external camera for better quality' : undefined,
        critical: (settings.width || 0) < 480
      })

      stream.getTracks().forEach(track => track.stop())
    } catch (error) {
      checks.push({
        name: 'Camera Quality',
        status: 'fail',
        message: 'Unable to test camera quality',
        recommendation: 'Check camera connection and permissions',
        critical: true
      })
    }

    // Test speech synthesis
    if ('speechSynthesis' in window) {
      const voices = speechSynthesis.getVoices()
      checks.push({
        name: 'Speech Voices',
        status: voices.length > 0 ? 'pass' : 'warning',
        message: `${voices.length} voices available`,
        recommendation: voices.length === 0 ? 'Speech synthesis may not work properly' : undefined,
        critical: false
      })
    }

    return checks
  }

  private async checkEnvironmentalConditions(): Promise<DemoEnvironmentCheck[]> {
    const checks: DemoEnvironmentCheck[] = []

    try {
      // Test lighting conditions using camera
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      const video = document.createElement('video')
      video.srcObject = stream
      await video.play()

      // Create canvas to analyze lighting
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')!
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      
      ctx.drawImage(video, 0, 0)
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
      
      // Calculate average brightness
      let totalBrightness = 0
      for (let i = 0; i < imageData.data.length; i += 4) {
        const r = imageData.data[i]
        const g = imageData.data[i + 1]
        const b = imageData.data[i + 2]
        totalBrightness += (r + g + b) / 3
      }
      
      const avgBrightness = totalBrightness / (imageData.data.length / 4)
      
      checks.push({
        name: 'Lighting Conditions',
        status: avgBrightness > 100 && avgBrightness < 200 ? 'pass' :
                avgBrightness > 50 && avgBrightness < 250 ? 'warning' : 'fail',
        message: `Brightness level: ${avgBrightness.toFixed(0)}`,
        recommendation: avgBrightness < 100 ? 'Increase lighting for better gesture recognition' :
                       avgBrightness > 200 ? 'Reduce harsh lighting to avoid glare' : undefined,
        critical: avgBrightness < 50 || avgBrightness > 250
      })

      stream.getTracks().forEach(track => track.stop())
    } catch (error) {
      checks.push({
        name: 'Lighting Conditions',
        status: 'warning',
        message: 'Unable to test lighting conditions',
        recommendation: 'Ensure adequate lighting for gesture recognition',
        critical: false
      })
    }

    return checks
  }

  private async runPerformanceBenchmark(): Promise<{ fps: number; memoryUsage: number }> {
    return new Promise((resolve) => {
      let frameCount = 0
      const startTime = performance.now()
      const duration = 1000 // 1 second test

      const testFrame = () => {
        frameCount++
        
        if (performance.now() - startTime < duration) {
          requestAnimationFrame(testFrame)
        } else {
          const fps = frameCount / (duration / 1000)
          
          let memoryUsage = 0
          if ('memory' in performance) {
            const memory = (performance as any).memory
            memoryUsage = (memory.usedJSHeapSize / memory.jsHeapSizeLimit) * 100
          }
          
          resolve({ fps, memoryUsage })
        }
      }
      
      requestAnimationFrame(testFrame)
    })
  }

  private checkWebGLSupport(): boolean {
    try {
      const canvas = document.createElement('canvas')
      return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'))
    } catch (e) {
      return false
    }
  }

  private async getSystemInfo(): Promise<DemoEnvironmentReport['systemInfo']> {
    const ua = navigator.userAgent
    let browser = 'Unknown'
    let version = 'Unknown'

    if (ua.includes('Chrome')) {
      browser = 'Chrome'
      version = ua.match(/Chrome\/(\d+)/)?.[1] || 'Unknown'
    } else if (ua.includes('Firefox')) {
      browser = 'Firefox'
      version = ua.match(/Firefox\/(\d+)/)?.[1] || 'Unknown'
    } else if (ua.includes('Safari') && !ua.includes('Chrome')) {
      browser = 'Safari'
      version = ua.match(/Version\/(\d+)/)?.[1] || 'Unknown'
    } else if (ua.includes('Edge')) {
      browser = 'Edge'
      version = ua.match(/Edge\/(\d+)/)?.[1] || 'Unknown'
    }

    let batteryLevel: number | undefined
    if ('getBattery' in navigator) {
      try {
        const battery = await (navigator as any).getBattery()
        batteryLevel = Math.round(battery.level * 100)
      } catch (error) {
        // Battery API not available
      }
    }

    return {
      browser,
      version,
      platform: navigator.platform,
      screenResolution: `${screen.width}x${screen.height}`,
      deviceMemory: (navigator as any).deviceMemory,
      hardwareConcurrency: navigator.hardwareConcurrency,
      connectionType: (navigator as any).connection?.effectiveType,
      batteryLevel
    }
  }

  private determineOverallStatus(checks: DemoEnvironmentCheck[]): 'ready' | 'needs_attention' | 'not_ready' {
    const criticalFailures = checks.filter(c => c.critical && c.status === 'fail')
    const warnings = checks.filter(c => c.status === 'warning')
    const failures = checks.filter(c => c.status === 'fail')

    if (criticalFailures.length > 0) {
      return 'not_ready'
    } else if (failures.length > 2 || warnings.length > 5) {
      return 'needs_attention'
    } else {
      return 'ready'
    }
  }

  private generateRecommendations(checks: DemoEnvironmentCheck[]): string[] {
    const recommendations: string[] = []
    
    checks.forEach(check => {
      if (check.recommendation) {
        recommendations.push(`${check.name}: ${check.recommendation}`)
      }
    })

    // Add general recommendations
    if (checks.some(c => c.status === 'fail' || c.status === 'warning')) {
      recommendations.push('Test the demo multiple times before presentation')
      recommendations.push('Have backup plans ready for technical issues')
      recommendations.push('Consider using a dedicated demo device')
    }

    return recommendations
  }

  private generateContingencyPlans(checks: DemoEnvironmentCheck[]): string[] {
    const plans: string[] = []

    // Camera issues
    if (checks.some(c => c.name === 'Camera Permission' && c.status === 'fail')) {
      plans.push('Use pre-recorded demo video if camera fails')
      plans.push('Have backup device with working camera ready')
    }

    // Performance issues
    if (checks.some(c => c.name === 'Rendering Performance' && c.status !== 'pass')) {
      plans.push('Enable simplified UI mode for better performance')
      plans.push('Close all other applications before demo')
    }

    // Network issues
    if (checks.some(c => c.name === 'API Connectivity' && c.status === 'fail')) {
      plans.push('Enable offline mode with cached responses')
      plans.push('Use local development server as backup')
    }

    // Lighting issues
    if (checks.some(c => c.name === 'Lighting Conditions' && c.status === 'fail')) {
      plans.push('Adjust room lighting or move to better location')
      plans.push('Use external lighting setup if available')
    }

    // General contingencies
    plans.push('Have demo video ready as ultimate fallback')
    plans.push('Practice demo script for smooth delivery')
    plans.push('Test all equipment 30 minutes before presentation')

    return plans
  }

  // Quick health check for ongoing monitoring
  async quickHealthCheck(): Promise<{ status: 'good' | 'warning' | 'critical'; message: string }> {
    try {
      // Check critical systems quickly
      const hasCamera = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)
      const hasWebGL = this.checkWebGLSupport()
      
      let apiStatus = false
      try {
        const response = await fetch('/api/health', { 
          method: 'GET',
          cache: 'no-cache',
          signal: AbortSignal.timeout(2000)
        })
        apiStatus = response.ok
      } catch (error) {
        // API not available
      }

      if (!hasCamera) {
        return { status: 'critical', message: 'Camera not available' }
      } else if (!apiStatus) {
        return { status: 'warning', message: 'API not responding' }
      } else if (!hasWebGL) {
        return { status: 'warning', message: 'WebGL not available' }
      } else {
        return { status: 'good', message: 'All systems operational' }
      }
    } catch (error) {
      return { status: 'critical', message: 'Health check failed' }
    }
  }

  exportReport(report: DemoEnvironmentReport): string {
    return JSON.stringify({
      ...report,
      timestamp: new Date().toISOString(),
      checkedBy: 'SignEase Demo Environment Checker v1.0'
    }, null, 2)
  }
}

// Global demo environment checker instance
export const demoEnvironmentChecker = new DemoEnvironmentChecker()

// React hook for demo environment monitoring
export function useDemoEnvironmentChecker() {
  const [report, setReport] = React.useState<DemoEnvironmentReport | null>(null)
  const [isChecking, setIsChecking] = React.useState(false)
  const [healthStatus, setHealthStatus] = React.useState<{ status: 'good' | 'warning' | 'critical'; message: string } | null>(null)

  const runCheck = React.useCallback(async () => {
    setIsChecking(true)
    try {
      const newReport = await demoEnvironmentChecker.runFullCheck()
      setReport(newReport)
    } catch (error) {
      console.error('Demo environment check failed:', error)
    } finally {
      setIsChecking(false)
    }
  }, [])

  React.useEffect(() => {
    // Run initial check
    runCheck()

    // Set up periodic health checks
    const healthCheckInterval = setInterval(async () => {
      const health = await demoEnvironmentChecker.quickHealthCheck()
      setHealthStatus(health)
    }, 30000) // Every 30 seconds

    return () => clearInterval(healthCheckInterval)
  }, [runCheck])

  return {
    report,
    isChecking,
    healthStatus,
    runCheck,
    exportReport: report ? () => demoEnvironmentChecker.exportReport(report) : null
  }
}

export default DemoEnvironmentChecker
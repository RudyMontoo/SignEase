/**
 * Demo Video Recorder
 * ===================
 * 
 * Records backup demo videos for presentation contingency
 */

export interface RecordingOptions {
  videoBitsPerSecond?: number
  audioBitsPerSecond?: number
  mimeType?: string
  maxDuration?: number // seconds
  includeAudio?: boolean
}

export interface RecordingMetadata {
  id: string
  title: string
  description: string
  duration: number
  size: number
  timestamp: number
  quality: 'low' | 'medium' | 'high'
}

class DemoVideoRecorder {
  private mediaRecorder: MediaRecorder | null = null
  private recordedChunks: Blob[] = []
  private isRecording: boolean = false
  private startTime: number = 0
  private stream: MediaStream | null = null
  private recordings: Map<string, { blob: Blob; metadata: RecordingMetadata }> = new Map()

  async startRecording(options: RecordingOptions = {}): Promise<string> {
    if (this.isRecording) {
      throw new Error('Recording already in progress')
    }

    const defaultOptions: RecordingOptions = {
      videoBitsPerSecond: 2500000, // 2.5 Mbps
      audioBitsPerSecond: 128000,  // 128 kbps
      mimeType: 'video/webm;codecs=vp9',
      maxDuration: 300, // 5 minutes
      includeAudio: true
    }

    const recordingOptions = { ...defaultOptions, ...options }

    try {
      // Get screen and camera streams
      const screenStream = await navigator.mediaDevices.getDisplayMedia({
        video: {
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          frameRate: { ideal: 30 }
        },
        audio: recordingOptions.includeAudio
      })

      let cameraStream: MediaStream | null = null
      try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            frameRate: { ideal: 30 }
          },
          audio: false // Audio from screen capture is enough
        })
      } catch (error) {
        console.warn('Camera not available for recording:', error)
      }

      // Combine streams if camera is available
      if (cameraStream) {
        this.stream = this.combineStreams(screenStream, cameraStream)
      } else {
        this.stream = screenStream
      }

      // Setup MediaRecorder
      const mediaRecorderOptions: MediaRecorderOptions = {
        mimeType: this.getSupportedMimeType(recordingOptions.mimeType),
        videoBitsPerSecond: recordingOptions.videoBitsPerSecond,
        audioBitsPerSecond: recordingOptions.audioBitsPerSecond
      }

      this.mediaRecorder = new MediaRecorder(this.stream, mediaRecorderOptions)
      this.recordedChunks = []

      // Setup event handlers
      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.recordedChunks.push(event.data)
        }
      }

      this.mediaRecorder.onstop = () => {
        this.handleRecordingStop()
      }

      // Start recording
      this.mediaRecorder.start(1000) // Collect data every second
      this.isRecording = true
      this.startTime = Date.now()

      // Auto-stop after max duration
      if (recordingOptions.maxDuration) {
        setTimeout(() => {
          if (this.isRecording) {
            this.stopRecording()
          }
        }, recordingOptions.maxDuration * 1000)
      }

      const recordingId = this.generateRecordingId()
      console.log(`🎥 Started demo recording: ${recordingId}`)
      
      return recordingId

    } catch (error) {
      console.error('Failed to start recording:', error)
      throw new Error(`Recording failed: ${error.message}`)
    }
  }

  async stopRecording(): Promise<string> {
    if (!this.isRecording || !this.mediaRecorder) {
      throw new Error('No recording in progress')
    }

    return new Promise((resolve, reject) => {
      if (!this.mediaRecorder) {
        reject(new Error('MediaRecorder not available'))
        return
      }

      const recordingId = this.generateRecordingId()

      this.mediaRecorder.onstop = () => {
        try {
          const result = this.handleRecordingStop()
          resolve(result)
        } catch (error) {
          reject(error)
        }
      }

      this.mediaRecorder.stop()
      
      // Stop all tracks
      if (this.stream) {
        this.stream.getTracks().forEach(track => track.stop())
      }

      this.isRecording = false
    })
  }

  private handleRecordingStop(): string {
    const duration = (Date.now() - this.startTime) / 1000
    const blob = new Blob(this.recordedChunks, { type: 'video/webm' })
    const recordingId = this.generateRecordingId()

    const metadata: RecordingMetadata = {
      id: recordingId,
      title: `Demo Recording ${new Date().toLocaleString()}`,
      description: 'SignEase demo backup recording',
      duration,
      size: blob.size,
      timestamp: Date.now(),
      quality: this.determineQuality(blob.size, duration)
    }

    this.recordings.set(recordingId, { blob, metadata })

    console.log(`🎥 Recording completed: ${recordingId} (${duration.toFixed(1)}s, ${(blob.size / 1024 / 1024).toFixed(1)}MB)`)

    // Cleanup
    this.recordedChunks = []
    this.stream = null
    this.mediaRecorder = null

    return recordingId
  }

  private combineStreams(screenStream: MediaStream, cameraStream: MediaStream): MediaStream {
    // Create a canvas to combine screen and camera
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')!
    
    canvas.width = 1920
    canvas.height = 1080

    // Create video elements
    const screenVideo = document.createElement('video')
    const cameraVideo = document.createElement('video')
    
    screenVideo.srcObject = screenStream
    cameraVideo.srcObject = cameraStream
    
    screenVideo.play()
    cameraVideo.play()

    // Combine streams on canvas
    const drawFrame = () => {
      if (screenVideo.readyState >= 2) {
        // Draw screen capture (full size)
        ctx.drawImage(screenVideo, 0, 0, canvas.width, canvas.height)
      }
      
      if (cameraVideo.readyState >= 2) {
        // Draw camera in bottom-right corner (picture-in-picture)
        const cameraWidth = 320
        const cameraHeight = 240
        const x = canvas.width - cameraWidth - 20
        const y = canvas.height - cameraHeight - 20
        
        // Add border
        ctx.strokeStyle = '#ffffff'
        ctx.lineWidth = 3
        ctx.strokeRect(x - 2, y - 2, cameraWidth + 4, cameraHeight + 4)
        
        // Draw camera feed
        ctx.drawImage(cameraVideo, x, y, cameraWidth, cameraHeight)
      }
      
      if (this.isRecording) {
        requestAnimationFrame(drawFrame)
      }
    }

    drawFrame()

    // Get stream from canvas
    const combinedStream = canvas.captureStream(30)
    
    // Add audio from screen stream
    const audioTracks = screenStream.getAudioTracks()
    audioTracks.forEach(track => combinedStream.addTrack(track))

    return combinedStream
  }

  private getSupportedMimeType(preferred?: string): string {
    const types = [
      preferred,
      'video/webm;codecs=vp9',
      'video/webm;codecs=vp8',
      'video/webm',
      'video/mp4'
    ].filter(Boolean) as string[]

    for (const type of types) {
      if (MediaRecorder.isTypeSupported(type)) {
        return type
      }
    }

    return 'video/webm' // Fallback
  }

  private generateRecordingId(): string {
    return `demo_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  private determineQuality(size: number, duration: number): 'low' | 'medium' | 'high' {
    const bitrate = (size * 8) / duration // bits per second
    const mbps = bitrate / 1000000

    if (mbps > 5) return 'high'
    if (mbps > 2) return 'medium'
    return 'low'
  }

  // Public API methods
  isCurrentlyRecording(): boolean {
    return this.isRecording
  }

  getRecordingDuration(): number {
    if (!this.isRecording) return 0
    return (Date.now() - this.startTime) / 1000
  }

  getRecordings(): RecordingMetadata[] {
    return Array.from(this.recordings.values()).map(r => r.metadata)
  }

  async downloadRecording(recordingId: string, filename?: string): Promise<void> {
    const recording = this.recordings.get(recordingId)
    if (!recording) {
      throw new Error(`Recording ${recordingId} not found`)
    }

    const url = URL.createObjectURL(recording.blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename || `${recording.metadata.title}.webm`
    a.click()
    
    URL.revokeObjectURL(url)
  }

  async playRecording(recordingId: string): Promise<HTMLVideoElement> {
    const recording = this.recordings.get(recordingId)
    if (!recording) {
      throw new Error(`Recording ${recordingId} not found`)
    }

    const video = document.createElement('video')
    video.src = URL.createObjectURL(recording.blob)
    video.controls = true
    video.style.width = '100%'
    video.style.maxWidth = '800px'
    
    return video
  }

  deleteRecording(recordingId: string): boolean {
    return this.recordings.delete(recordingId)
  }

  clearAllRecordings(): void {
    this.recordings.clear()
    console.log('🗑️ All demo recordings cleared')
  }

  // Create a quick demo recording with predefined settings
  async recordQuickDemo(durationSeconds: number = 60): Promise<string> {
    const recordingId = await this.startRecording({
      videoBitsPerSecond: 1500000, // Lower bitrate for quick demo
      maxDuration: durationSeconds,
      includeAudio: true
    })

    console.log(`🎥 Quick demo recording started (${durationSeconds}s max)`)
    return recordingId
  }

  // Export recording metadata for backup
  exportRecordingList(): string {
    const recordings = this.getRecordings()
    const exportData = {
      timestamp: new Date().toISOString(),
      totalRecordings: recordings.length,
      recordings: recordings.map(r => ({
        ...r,
        sizeFormatted: `${(r.size / 1024 / 1024).toFixed(1)}MB`,
        durationFormatted: `${Math.floor(r.duration / 60)}:${(r.duration % 60).toFixed(0).padStart(2, '0')}`
      }))
    }

    return JSON.stringify(exportData, null, 2)
  }
}

// Global demo video recorder instance
export const demoVideoRecorder = new DemoVideoRecorder()

// React hook for demo video recording
export function useDemoVideoRecorder() {
  const [isRecording, setIsRecording] = React.useState(false)
  const [recordings, setRecordings] = React.useState<RecordingMetadata[]>([])
  const [currentDuration, setCurrentDuration] = React.useState(0)

  React.useEffect(() => {
    setRecordings(demoVideoRecorder.getRecordings())

    let interval: number | null = null
    if (demoVideoRecorder.isCurrentlyRecording()) {
      setIsRecording(true)
      interval = window.setInterval(() => {
        setCurrentDuration(demoVideoRecorder.getRecordingDuration())
      }, 1000)
    } else {
      setIsRecording(false)
      setCurrentDuration(0)
    }

    return () => {
      if (interval) clearInterval(interval)
    }
  }, [])

  const startRecording = React.useCallback(async (options?: RecordingOptions) => {
    try {
      const recordingId = await demoVideoRecorder.startRecording(options)
      setIsRecording(true)
      return recordingId
    } catch (error) {
      console.error('Failed to start recording:', error)
      throw error
    }
  }, [])

  const stopRecording = React.useCallback(async () => {
    try {
      const recordingId = await demoVideoRecorder.stopRecording()
      setIsRecording(false)
      setCurrentDuration(0)
      setRecordings(demoVideoRecorder.getRecordings())
      return recordingId
    } catch (error) {
      console.error('Failed to stop recording:', error)
      throw error
    }
  }, [])

  const recordQuickDemo = React.useCallback(async (duration: number = 60) => {
    return demoVideoRecorder.recordQuickDemo(duration)
  }, [])

  return {
    isRecording,
    recordings,
    currentDuration,
    startRecording,
    stopRecording,
    recordQuickDemo,
    downloadRecording: demoVideoRecorder.downloadRecording.bind(demoVideoRecorder),
    playRecording: demoVideoRecorder.playRecording.bind(demoVideoRecorder),
    deleteRecording: demoVideoRecorder.deleteRecording.bind(demoVideoRecorder),
    clearAllRecordings: demoVideoRecorder.clearAllRecordings.bind(demoVideoRecorder),
    exportRecordingList: demoVideoRecorder.exportRecordingList.bind(demoVideoRecorder)
  }
}

export default DemoVideoRecorder
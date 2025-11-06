/**
 * Speech Utilities
 * ================
 * 
 * Utility functions for text-to-speech functionality
 */

export interface VoiceInfo {
  name: string
  lang: string
  gender: 'male' | 'female' | 'unknown'
  quality: 'high' | 'medium' | 'low'
  isDefault: boolean
  isLocal: boolean
}

/**
 * Check if text-to-speech is supported in the current browser
 */
export const isSpeechSupported = (): boolean => {
  return 'speechSynthesis' in window && 'SpeechSynthesisUtterance' in window
}

/**
 * Get available voices with enhanced information
 */
export const getAvailableVoices = (): VoiceInfo[] => {
  if (!isSpeechSupported()) {
    return []
  }

  const voices = speechSynthesis.getVoices()
  
  return voices.map(voice => ({
    name: voice.name,
    lang: voice.lang,
    gender: detectVoiceGender(voice.name),
    quality: detectVoiceQuality(voice),
    isDefault: voice.default,
    isLocal: voice.localService
  }))
}

/**
 * Get the best voice for a given language
 */
export const getBestVoiceForLanguage = (lang: string = 'en-US'): SpeechSynthesisVoice | null => {
  if (!isSpeechSupported()) {
    return null
  }

  const voices = speechSynthesis.getVoices()
  
  // Filter voices by language
  const languageVoices = voices.filter(voice => 
    voice.lang.toLowerCase().startsWith(lang.toLowerCase().split('-')[0])
  )

  if (languageVoices.length === 0) {
    return voices[0] || null
  }

  // Prioritize local voices
  const localVoices = languageVoices.filter(voice => voice.localService)
  if (localVoices.length > 0) {
    return localVoices.find(voice => voice.default) || localVoices[0]
  }

  // Fall back to any voice in the language
  return languageVoices.find(voice => voice.default) || languageVoices[0]
}

/**
 * Detect voice gender based on name patterns
 */
const detectVoiceGender = (voiceName: string): 'male' | 'female' | 'unknown' => {
  const name = voiceName.toLowerCase()
  
  // Common female voice names
  const femalePatterns = [
    'female', 'woman', 'girl', 'lady', 'samantha', 'victoria', 'susan', 'karen', 
    'alice', 'emma', 'sophia', 'olivia', 'ava', 'isabella', 'mia', 'zira', 'hazel'
  ]
  
  // Common male voice names
  const malePatterns = [
    'male', 'man', 'boy', 'guy', 'alex', 'daniel', 'david', 'fred', 'tom', 'thomas',
    'james', 'john', 'michael', 'william', 'richard', 'mark', 'paul'
  ]

  if (femalePatterns.some(pattern => name.includes(pattern))) {
    return 'female'
  }
  
  if (malePatterns.some(pattern => name.includes(pattern))) {
    return 'male'
  }
  
  return 'unknown'
}

/**
 * Detect voice quality based on voice properties
 */
const detectVoiceQuality = (voice: SpeechSynthesisVoice): 'high' | 'medium' | 'low' => {
  // Local voices are generally higher quality
  if (voice.localService) {
    return 'high'
  }
  
  // Default voices are usually good quality
  if (voice.default) {
    return 'medium'
  }
  
  // Network voices might be lower quality
  return 'low'
}

/**
 * Prepare text for speech by cleaning and formatting
 */
export const prepareTextForSpeech = (text: string): string => {
  return text
    // Remove extra whitespace
    .trim()
    .replace(/\s+/g, ' ')
    // Add pauses for punctuation
    .replace(/\./g, '. ')
    .replace(/,/g, ', ')
    .replace(/;/g, '; ')
    .replace(/:/g, ': ')
    .replace(/!/g, '! ')
    .replace(/\?/g, '? ')
    // Handle abbreviations
    .replace(/\bDr\./g, 'Doctor')
    .replace(/\bMr\./g, 'Mister')
    .replace(/\bMrs\./g, 'Missus')
    .replace(/\bMs\./g, 'Miss')
    // Handle numbers
    .replace(/\b(\d+)\b/g, (match) => {
      const num = parseInt(match)
      if (num < 100) {
        return match // Keep small numbers as is
      }
      return match // For now, keep larger numbers as is
    })
}

/**
 * Estimate speech duration in milliseconds
 */
export const estimateSpeechDuration = (text: string, rate: number = 1): number => {
  const wordsPerMinute = 150 * rate // Average speaking rate
  const words = text.trim().split(/\s+/).length
  const minutes = words / wordsPerMinute
  return Math.round(minutes * 60 * 1000) // Convert to milliseconds
}

/**
 * Break long text into chunks for better speech synthesis
 */
export const chunkTextForSpeech = (text: string, maxChunkLength: number = 200): string[] => {
  if (text.length <= maxChunkLength) {
    return [text]
  }

  const chunks: string[] = []
  const sentences = text.split(/[.!?]+/)
  
  let currentChunk = ''
  
  for (const sentence of sentences) {
    const trimmedSentence = sentence.trim()
    if (!trimmedSentence) continue
    
    const sentenceWithPunctuation = trimmedSentence + '.'
    
    if (currentChunk.length + sentenceWithPunctuation.length <= maxChunkLength) {
      currentChunk += (currentChunk ? ' ' : '') + sentenceWithPunctuation
    } else {
      if (currentChunk) {
        chunks.push(currentChunk)
      }
      currentChunk = sentenceWithPunctuation
    }
  }
  
  if (currentChunk) {
    chunks.push(currentChunk)
  }
  
  return chunks.length > 0 ? chunks : [text]
}

/**
 * Get speech synthesis settings optimized for different use cases
 */
export const getSpeechPresets = () => {
  return {
    normal: {
      rate: 1.0,
      pitch: 1.0,
      volume: 1.0
    },
    slow: {
      rate: 0.7,
      pitch: 1.0,
      volume: 1.0
    },
    fast: {
      rate: 1.3,
      pitch: 1.0,
      volume: 1.0
    },
    expressive: {
      rate: 0.9,
      pitch: 1.1,
      volume: 1.0
    },
    calm: {
      rate: 0.8,
      pitch: 0.9,
      volume: 0.8
    }
  }
}

/**
 * Test speech synthesis capability
 */
export const testSpeechSynthesis = (): Promise<boolean> => {
  return new Promise((resolve) => {
    if (!isSpeechSupported()) {
      resolve(false)
      return
    }

    try {
      const utterance = new SpeechSynthesisUtterance('Test')
      utterance.volume = 0 // Silent test
      
      utterance.onend = () => resolve(true)
      utterance.onerror = () => resolve(false)
      
      speechSynthesis.speak(utterance)
      
      // Timeout after 5 seconds
      setTimeout(() => resolve(false), 5000)
    } catch {
      resolve(false)
    }
  })
}

/**
 * Get browser-specific speech synthesis information
 */
export const getBrowserSpeechInfo = () => {
  const userAgent = navigator.userAgent.toLowerCase()
  
  let browser = 'unknown'
  let speechSupport = 'unknown'
  
  if (userAgent.includes('chrome')) {
    browser = 'Chrome'
    speechSupport = 'excellent'
  } else if (userAgent.includes('firefox')) {
    browser = 'Firefox'
    speechSupport = 'good'
  } else if (userAgent.includes('safari')) {
    browser = 'Safari'
    speechSupport = 'good'
  } else if (userAgent.includes('edge')) {
    browser = 'Edge'
    speechSupport = 'excellent'
  }
  
  return {
    browser,
    speechSupport,
    isSupported: isSpeechSupported(),
    voiceCount: speechSynthesis.getVoices().length
  }
}
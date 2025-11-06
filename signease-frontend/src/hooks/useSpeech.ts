/**
 * Text-to-Speech Hook
 * ===================
 * 
 * Web Speech API wrapper for text-to-speech functionality
 */

import { useState, useCallback, useRef, useEffect } from 'react'

export interface SpeechSettings {
  rate: number
  pitch: number
  volume: number
  voice: SpeechSynthesisVoice | null
  lang: string
}

export interface SpeechState {
  isSupported: boolean
  isSpeaking: boolean
  isPaused: boolean
  voices: SpeechSynthesisVoice[]
  settings: SpeechSettings
  error: string | null
}

export interface SpeechActions {
  speak: (text: string) => Promise<void>
  pause: () => void
  resume: () => void
  stop: () => void
  updateSettings: (settings: Partial<SpeechSettings>) => void
}

export const useSpeech = (): SpeechState & SpeechActions => {
  const [state, setState] = useState<SpeechState>({
    isSupported: 'speechSynthesis' in window,
    isSpeaking: false,
    isPaused: false,
    voices: [],
    settings: {
      rate: 1,
      pitch: 1,
      volume: 1,
      voice: null,
      lang: 'en-US'
    },
    error: null
  })

  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null)

  // Load available voices
  const loadVoices = useCallback(() => {
    if (!state.isSupported) return

    const voices = speechSynthesis.getVoices()
    setState(prev => {
      const englishVoices = voices.filter(voice => voice.lang.startsWith('en'))
      const defaultVoice = englishVoices.find(voice => voice.default) || englishVoices[0] || voices[0]
      
      return {
        ...prev,
        voices,
        settings: {
          ...prev.settings,
          voice: prev.settings.voice || defaultVoice
        }
      }
    })
  }, [state.isSupported])

  // Initialize voices
  useEffect(() => {
    if (!state.isSupported) return

    loadVoices()

    // Some browsers load voices asynchronously
    if (speechSynthesis.onvoiceschanged !== undefined) {
      speechSynthesis.onvoiceschanged = loadVoices
    }

    return () => {
      if (speechSynthesis.onvoiceschanged !== undefined) {
        speechSynthesis.onvoiceschanged = null
      }
    }
  }, [loadVoices, state.isSupported])

  // Speak text
  const speak = useCallback(async (text: string): Promise<void> => {
    if (!state.isSupported) {
      throw new Error('Speech synthesis not supported')
    }

    if (!text.trim()) {
      throw new Error('No text to speak')
    }

    // Stop any current speech
    speechSynthesis.cancel()

    return new Promise((resolve, reject) => {
      try {
        const utterance = new SpeechSynthesisUtterance(text)
        
        // Apply settings
        utterance.rate = state.settings.rate
        utterance.pitch = state.settings.pitch
        utterance.volume = state.settings.volume
        utterance.lang = state.settings.lang
        
        if (state.settings.voice) {
          utterance.voice = state.settings.voice
        }

        // Event handlers
        utterance.onstart = () => {
          setState(prev => ({ ...prev, isSpeaking: true, isPaused: false, error: null }))
        }

        utterance.onend = () => {
          setState(prev => ({ ...prev, isSpeaking: false, isPaused: false }))
          utteranceRef.current = null
          resolve()
        }

        utterance.onerror = (event) => {
          const errorMessage = `Speech error: ${event.error}`
          setState(prev => ({ 
            ...prev, 
            isSpeaking: false, 
            isPaused: false, 
            error: errorMessage 
          }))
          utteranceRef.current = null
          reject(new Error(errorMessage))
        }

        utterance.onpause = () => {
          setState(prev => ({ ...prev, isPaused: true }))
        }

        utterance.onresume = () => {
          setState(prev => ({ ...prev, isPaused: false }))
        }

        utteranceRef.current = utterance
        speechSynthesis.speak(utterance)

      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown speech error'
        setState(prev => ({ ...prev, error: errorMessage }))
        reject(new Error(errorMessage))
      }
    })
  }, [state.isSupported, state.settings])

  // Pause speech
  const pause = useCallback(() => {
    if (state.isSupported && state.isSpeaking && !state.isPaused) {
      speechSynthesis.pause()
    }
  }, [state.isSupported, state.isSpeaking, state.isPaused])

  // Resume speech
  const resume = useCallback(() => {
    if (state.isSupported && state.isSpeaking && state.isPaused) {
      speechSynthesis.resume()
    }
  }, [state.isSupported, state.isSpeaking, state.isPaused])

  // Stop speech
  const stop = useCallback(() => {
    if (state.isSupported) {
      speechSynthesis.cancel()
      setState(prev => ({ ...prev, isSpeaking: false, isPaused: false }))
      utteranceRef.current = null
    }
  }, [state.isSupported])

  // Update settings
  const updateSettings = useCallback((newSettings: Partial<SpeechSettings>) => {
    setState(prev => ({
      ...prev,
      settings: { ...prev.settings, ...newSettings },
      error: null
    }))
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (state.isSupported) {
        speechSynthesis.cancel()
      }
    }
  }, [state.isSupported])

  return {
    ...state,
    speak,
    pause,
    resume,
    stop,
    updateSettings
  }
}
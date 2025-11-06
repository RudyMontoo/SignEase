/**
 * Gesture Guide Component
 * =======================
 * 
 * Interactive ASL alphabet guide with visual demonstrations
 */

import React, { useState } from 'react'
import Button from './ui/Button'
import Badge from './ui/Badge'
import Switch from './ui/Switch'

interface GestureGuideProps {
  isOpen: boolean
  onClose: () => void
  enabled: boolean
  onEnabledChange: (enabled: boolean) => void
}

// ASL Alphabet data with descriptions
const ASL_ALPHABET = [
  { letter: 'A', description: 'Closed fist with thumb on the side' },
  { letter: 'B', description: 'Flat hand with fingers together, thumb across palm' },
  { letter: 'C', description: 'Curved hand forming a C shape' },
  { letter: 'D', description: 'Index finger up, other fingers and thumb form O' },
  { letter: 'E', description: 'Fingers curled down, thumb across fingertips' },
  { letter: 'F', description: 'Index and thumb form circle, other fingers up' },
  { letter: 'G', description: 'Index finger and thumb pointing sideways' },
  { letter: 'H', description: 'Index and middle fingers sideways together' },
  { letter: 'I', description: 'Pinky finger up, other fingers down' },
  { letter: 'J', description: 'Pinky finger up, draw a J in the air' },
  { letter: 'K', description: 'Index and middle up, thumb between them' },
  { letter: 'L', description: 'Index finger up, thumb out (L shape)' },
  { letter: 'M', description: 'Thumb under first three fingers' },
  { letter: 'N', description: 'Thumb under first two fingers' },
  { letter: 'O', description: 'All fingers and thumb form circle' },
  { letter: 'P', description: 'Like K but pointing down' },
  { letter: 'Q', description: 'Index finger and thumb pointing down' },
  { letter: 'R', description: 'Index and middle fingers crossed' },
  { letter: 'S', description: 'Closed fist with thumb over fingers' },
  { letter: 'T', description: 'Thumb between index and middle finger' },
  { letter: 'U', description: 'Index and middle fingers up together' },
  { letter: 'V', description: 'Index and middle fingers up, separated' },
  { letter: 'W', description: 'Index, middle, and ring fingers up' },
  { letter: 'X', description: 'Index finger crooked like a hook' },
  { letter: 'Y', description: 'Thumb and pinky extended (hang loose)' },
  { letter: 'Z', description: 'Index finger draws Z in the air' }
]

const PRACTICE_WORDS = [
  { word: 'HELLO', letters: ['H', 'E', 'L', 'L', 'O'] },
  { word: 'WORLD', letters: ['W', 'O', 'R', 'L', 'D'] },
  { word: 'PEACE', letters: ['P', 'E', 'A', 'C', 'E'] },
  { word: 'LOVE', letters: ['L', 'O', 'V', 'E'] },
  { word: 'HELP', letters: ['H', 'E', 'L', 'P'] },
  { word: 'THANK', letters: ['T', 'H', 'A', 'N', 'K'] },
  { word: 'YOU', letters: ['Y', 'O', 'U'] }
]

const GestureGuide: React.FC<GestureGuideProps> = ({
  isOpen,
  onClose,
  enabled,
  onEnabledChange
}) => {
  const [selectedLetter, setSelectedLetter] = useState<string | null>(null)
  const [currentTab, setCurrentTab] = useState<'alphabet' | 'practice'>('alphabet')
  const [selectedWord, setSelectedWord] = useState<string | null>(null)

  if (!isOpen) return null

  const selectedGesture = selectedLetter ? ASL_ALPHABET.find(g => g.letter === selectedLetter) : null
  const selectedPracticeWord = selectedWord ? PRACTICE_WORDS.find(w => w.word === selectedWord) : null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center space-x-4">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
              ASL Gesture Guide
            </h2>
            <Badge variant="primary">Interactive</Badge>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Show on screen
              </label>
              <Switch
                checked={enabled}
                onChange={(e) => onEnabledChange(e.target.checked)}
              />
            </div>
            
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-200 dark:border-gray-700">
          <button
            onClick={() => setCurrentTab('alphabet')}
            className={`px-6 py-3 text-sm font-medium transition-colors ${
              currentTab === 'alphabet'
                ? 'text-blue-600 border-b-2 border-blue-600'
                : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
            }`}
          >
            ASL Alphabet
          </button>
          <button
            onClick={() => setCurrentTab('practice')}
            className={`px-6 py-3 text-sm font-medium transition-colors ${
              currentTab === 'practice'
                ? 'text-blue-600 border-b-2 border-blue-600'
                : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
            }`}
          >
            Practice Words
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {currentTab === 'alphabet' && (
            <div className="space-y-6">
              {/* Alphabet Grid */}
              <div className="grid grid-cols-6 md:grid-cols-13 gap-2">
                {ASL_ALPHABET.map((gesture) => (
                  <button
                    key={gesture.letter}
                    onClick={() => setSelectedLetter(
                      selectedLetter === gesture.letter ? null : gesture.letter
                    )}
                    className={`aspect-square rounded-lg border-2 font-bold text-lg transition-all ${
                      selectedLetter === gesture.letter
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 text-blue-600'
                        : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600 text-gray-700 dark:text-gray-300'
                    }`}
                  >
                    {gesture.letter}
                  </button>
                ))}
              </div>

              {/* Selected Letter Details */}
              {selectedGesture && (
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
                  <div className="flex items-start space-x-4">
                    <div className="text-6xl font-bold text-blue-600 bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm">
                      {selectedGesture.letter}
                    </div>
                    <div className="flex-1">
                      <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
                        Letter {selectedGesture.letter}
                      </h3>
                      <p className="text-gray-700 dark:text-gray-300 mb-4">
                        {selectedGesture.description}
                      </p>
                      
                      {/* Practice Tips */}
                      <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                        <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">
                          Practice Tips:
                        </h4>
                        <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                          <li>• Hold the gesture steady for 1-2 seconds</li>
                          <li>• Keep your hand clearly visible to the camera</li>
                          <li>• Ensure good lighting for better recognition</li>
                          <li>• Practice the motion slowly at first</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Quick Reference */}
              <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
                <h3 className="font-medium text-gray-900 dark:text-gray-100 mb-3">
                  Quick Reference Tips
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-600 dark:text-gray-400">
                  <div>
                    <strong>Hand Position:</strong>
                    <ul className="mt-1 space-y-1">
                      <li>• Keep hand centered in camera view</li>
                      <li>• Maintain consistent distance from camera</li>
                      <li>• Use your dominant hand</li>
                    </ul>
                  </div>
                  <div>
                    <strong>Recognition Tips:</strong>
                    <ul className="mt-1 space-y-1">
                      <li>• Hold gestures for 1-2 seconds</li>
                      <li>• Avoid rapid movements</li>
                      <li>• Check confidence scores</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          )}

          {currentTab === 'practice' && (
            <div className="space-y-6">
              {/* Practice Words */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {PRACTICE_WORDS.map((word) => (
                  <button
                    key={word.word}
                    onClick={() => setSelectedWord(
                      selectedWord === word.word ? null : word.word
                    )}
                    className={`p-4 rounded-lg border-2 text-left transition-all ${
                      selectedWord === word.word
                        ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                    }`}
                  >
                    <div className="font-bold text-lg text-gray-900 dark:text-gray-100">
                      {word.word}
                    </div>
                    <div className="text-sm text-gray-500 mt-1">
                      {word.letters.length} letters
                    </div>
                    <div className="flex space-x-1 mt-2">
                      {word.letters.map((letter, index) => (
                        <span
                          key={index}
                          className="w-6 h-6 bg-gray-200 dark:bg-gray-700 rounded text-xs flex items-center justify-center font-medium"
                        >
                          {letter}
                        </span>
                      ))}
                    </div>
                  </button>
                ))}
              </div>

              {/* Selected Word Practice */}
              {selectedPracticeWord && (
                <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-4">
                    Practice: {selectedPracticeWord.word}
                  </h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Letter Sequence */}
                    <div>
                      <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-3">
                        Letter Sequence:
                      </h4>
                      <div className="space-y-2">
                        {selectedPracticeWord.letters.map((letter, index) => {
                          const gesture = ASL_ALPHABET.find(g => g.letter === letter)
                          return (
                            <div key={index} className="flex items-center space-x-3 p-2 bg-white dark:bg-gray-800 rounded">
                              <div className="w-8 h-8 bg-green-100 dark:bg-green-800 rounded flex items-center justify-center font-bold text-green-600 dark:text-green-400">
                                {letter}
                              </div>
                              <div className="text-sm text-gray-600 dark:text-gray-400">
                                {gesture?.description}
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                    
                    {/* Practice Instructions */}
                    <div>
                      <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-3">
                        Practice Instructions:
                      </h4>
                      <div className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                        <div className="flex items-start space-x-2">
                          <span className="font-bold text-green-600">1.</span>
                          <span>Start recognition in the main app</span>
                        </div>
                        <div className="flex items-start space-x-2">
                          <span className="font-bold text-green-600">2.</span>
                          <span>Sign each letter slowly and clearly</span>
                        </div>
                        <div className="flex items-start space-x-2">
                          <span className="font-bold text-green-600">3.</span>
                          <span>Wait for recognition before next letter</span>
                        </div>
                        <div className="flex items-start space-x-2">
                          <span className="font-bold text-green-600">4.</span>
                          <span>Check that the word builds correctly</span>
                        </div>
                        <div className="flex items-start space-x-2">
                          <span className="font-bold text-green-600">5.</span>
                          <span>Use text-to-speech to hear the result</span>
                        </div>
                      </div>
                      
                      <div className="mt-4 p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded">
                        <p className="text-xs text-yellow-700 dark:text-yellow-300">
                          <strong>Tip:</strong> If recognition is inconsistent, try adjusting the confidence threshold in settings or improving lighting conditions.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-6 border-t border-gray-200 dark:border-gray-700">
          <div className="text-sm text-gray-500">
            Click any letter or word to see detailed instructions
          </div>
          <Button
            onClick={onClose}
            variant="primary"
            size="md"
          >
            Close Guide
          </Button>
        </div>
      </div>
    </div>
  )
}

export default GestureGuide
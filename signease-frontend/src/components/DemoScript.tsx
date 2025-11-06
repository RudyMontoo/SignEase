/**
 * Demo Script Component
 * =====================
 * 
 * Interactive demo script with timing, cues, and fallback options
 */

import React, { useState, useEffect, useRef } from 'react'
import { useTheme } from '../hooks/useTheme'
import Card, { CardHeader, CardTitle, CardContent } from './ui/Card'
import Button from './ui/Button'
import Badge from './ui/Badge'

export interface DemoStep {
  id: string
  title: string
  duration: number // seconds
  description: string
  actions: string[]
  cues: string[]
  fallbackOptions: string[]
  criticalStep: boolean
}

export interface DemoScriptProps {
  className?: string
  onStepComplete?: (stepId: string) => void
  onDemoComplete?: () => void
}

const DEMO_SCRIPT: DemoStep[] = [
  {
    id: 'intro',
    title: 'Introduction & Problem Statement',
    duration: 30,
    description: 'Hook the audience with the communication barrier problem',
    actions: [
      'Start with compelling statistic about deaf/hard-of-hearing community',
      'Highlight current communication challenges',
      'Introduce SignEase as the solution'
    ],
    cues: [
      'Make eye contact with audience',
      'Use confident, clear voice',
      'Show passion for the problem'
    ],
    fallbackOptions: [
      'Have backup slides ready',
      'Memorize key statistics',
      'Practice smooth transitions'
    ],
    criticalStep: true
  },
  {
    id: 'solution-overview',
    title: 'Solution Overview',
    duration: 45,
    description: 'Present SignEase capabilities and unique value proposition',
    actions: [
      'Explain real-time ASL recognition technology',
      'Highlight 99.57% accuracy achievement',
      'Show key features: AR overlay, speech synthesis, sentence building'
    ],
    cues: [
      'Use hand gestures to demonstrate concept',
      'Emphasize "real-time" and "high accuracy"',
      'Build excitement for live demo'
    ],
    fallbackOptions: [
      'Have feature screenshots ready',
      'Prepare accuracy comparison charts',
      'Use demo video if needed'
    ],
    criticalStep: true
  },
  {
    id: 'live-demo',
    title: 'Live Demo',
    duration: 90,
    description: 'Demonstrate SignEase in action with real ASL gestures',
    actions: [
      'Open SignEase application',
      'Show camera activation and hand detection',
      'Demonstrate spelling "HELLO" with ASL',
      'Show AR overlay and text-to-speech features',
      'Demonstrate sentence building with "HELLO WORLD"',
      'Show settings and customization options'
    ],
    cues: [
      'Ensure good lighting and camera angle',
      'Speak clearly while signing',
      'Point out real-time recognition',
      'Highlight smooth user experience'
    ],
    fallbackOptions: [
      'Use backup demo video if camera fails',
      'Have pre-recorded gesture sequences',
      'Switch to offline mode if API fails',
      'Use simplified UI if performance issues'
    ],
    criticalStep: true
  },
  {
    id: 'technical-architecture',
    title: 'Technical Architecture',
    duration: 45,
    description: 'Showcase the technical innovation and implementation',
    actions: [
      'Show MediaPipe hand tracking integration',
      'Explain ML model training and optimization',
      'Highlight performance optimizations',
      'Demonstrate cross-platform compatibility'
    ],
    cues: [
      'Use technical terms appropriately for audience',
      'Show confidence in technical decisions',
      'Highlight innovation aspects'
    ],
    fallbackOptions: [
      'Have architecture diagrams ready',
      'Prepare performance metrics slides',
      'Show code snippets if appropriate'
    ],
    criticalStep: false
  },
  {
    id: 'impact-future',
    title: 'Impact & Future Vision',
    duration: 30,
    description: 'Present the broader impact and future roadmap',
    actions: [
      'Quantify potential impact on deaf/hard-of-hearing community',
      'Present future features: mobile app, more sign languages',
      'Discuss scalability and commercial potential'
    ],
    cues: [
      'Show enthusiasm for future possibilities',
      'Connect back to original problem',
      'End with strong call to action'
    ],
    fallbackOptions: [
      'Have impact statistics ready',
      'Prepare roadmap timeline',
      'Show market opportunity data'
    ],
    criticalStep: true
  }
]

export const DemoScript: React.FC<DemoScriptProps> = ({
  className = '',
  onStepComplete,
  onDemoComplete
}) => {
  const { isDark } = useTheme()
  const [currentStepIndex, setCurrentStepIndex] = useState(0)
  const [isRunning, setIsRunning] = useState(false)
  const [timeRemaining, setTimeRemaining] = useState(0)
  const [totalElapsed, setTotalElapsed] = useState(0)
  const [completedSteps, setCompletedSteps] = useState<Set<string>>(new Set())
  const timerRef = useRef<number | null>(null)
  const startTimeRef = useRef<number>(0)

  const currentStep = DEMO_SCRIPT[currentStepIndex]
  const totalDuration = DEMO_SCRIPT.reduce((sum, step) => sum + step.duration, 0)

  useEffect(() => {
    if (isRunning && timeRemaining > 0) {
      timerRef.current = window.setTimeout(() => {
        setTimeRemaining(prev => prev - 1)
        setTotalElapsed(prev => prev + 1)
      }, 1000)
    } else if (isRunning && timeRemaining === 0) {
      // Auto-advance to next step
      handleNextStep()
    }

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current)
      }
    }
  }, [isRunning, timeRemaining])

  const startDemo = () => {
    setIsRunning(true)
    setCurrentStepIndex(0)
    setTimeRemaining(DEMO_SCRIPT[0].duration)
    setTotalElapsed(0)
    setCompletedSteps(new Set())
    startTimeRef.current = Date.now()
  }

  const pauseDemo = () => {
    setIsRunning(false)
  }

  const resumeDemo = () => {
    setIsRunning(true)
  }

  const resetDemo = () => {
    setIsRunning(false)
    setCurrentStepIndex(0)
    setTimeRemaining(0)
    setTotalElapsed(0)
    setCompletedSteps(new Set())
  }

  const handleNextStep = () => {
    const currentStepId = currentStep.id
    const newCompletedSteps = new Set(completedSteps)
    newCompletedSteps.add(currentStepId)
    setCompletedSteps(newCompletedSteps)

    if (onStepComplete) {
      onStepComplete(currentStepId)
    }

    if (currentStepIndex < DEMO_SCRIPT.length - 1) {
      const nextIndex = currentStepIndex + 1
      setCurrentStepIndex(nextIndex)
      setTimeRemaining(DEMO_SCRIPT[nextIndex].duration)
    } else {
      // Demo complete
      setIsRunning(false)
      if (onDemoComplete) {
        onDemoComplete()
      }
    }
  }

  const handlePreviousStep = () => {
    if (currentStepIndex > 0) {
      const prevIndex = currentStepIndex - 1
      setCurrentStepIndex(prevIndex)
      setTimeRemaining(DEMO_SCRIPT[prevIndex].duration)
      
      // Remove current step from completed if going back
      const newCompletedSteps = new Set(completedSteps)
      newCompletedSteps.delete(currentStep.id)
      setCompletedSteps(newCompletedSteps)
    }
  }

  const jumpToStep = (stepIndex: number) => {
    setCurrentStepIndex(stepIndex)
    setTimeRemaining(DEMO_SCRIPT[stepIndex].duration)
  }

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const getStepStatus = (stepId: string, index: number): 'completed' | 'current' | 'upcoming' => {
    if (completedSteps.has(stepId)) return 'completed'
    if (index === currentStepIndex) return 'current'
    return 'upcoming'
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success'
      case 'current': return 'primary'
      case 'upcoming': return 'secondary'
      default: return 'secondary'
    }
  }

  return (
    <Card className={`${className}`}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Demo Script & Timer</CardTitle>
          <div className="flex items-center space-x-2">
            <Badge variant={isRunning ? 'success' : 'secondary'} size="sm">
              {isRunning ? 'Running' : 'Stopped'}
            </Badge>
            <Badge variant="outline" size="sm">
              {formatTime(totalElapsed)} / {formatTime(totalDuration)}
            </Badge>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Timer and Controls */}
        <div className={`p-4 rounded-lg border ${
          isDark ? 'border-gray-600 bg-gray-800' : 'border-gray-200 bg-gray-50'
        }`}>
          <div className="flex items-center justify-between mb-4">
            <div className="text-2xl font-mono">
              {formatTime(timeRemaining)}
            </div>
            <div className="flex items-center space-x-2">
              {!isRunning ? (
                <Button variant="primary" size="sm" onClick={startDemo}>
                  ▶️ Start Demo
                </Button>
              ) : (
                <>
                  <Button variant="secondary" size="sm" onClick={pauseDemo}>
                    ⏸️ Pause
                  </Button>
                  <Button variant="outline" size="sm" onClick={resumeDemo}>
                    ▶️ Resume
                  </Button>
                </>
              )}
              <Button variant="outline" size="sm" onClick={resetDemo}>
                🔄 Reset
              </Button>
            </div>
          </div>

          {/* Progress bar */}
          <div className={`w-full h-2 rounded-full ${
            isDark ? 'bg-gray-700' : 'bg-gray-200'
          }`}>
            <div 
              className="h-full bg-blue-600 rounded-full transition-all duration-1000"
              style={{ width: `${(totalElapsed / totalDuration) * 100}%` }}
            />
          </div>
        </div>

        {/* Step Overview */}
        <div className="grid grid-cols-5 gap-2">
          {DEMO_SCRIPT.map((step, index) => {
            const status = getStepStatus(step.id, index)
            return (
              <button
                key={step.id}
                onClick={() => jumpToStep(index)}
                className={`p-2 text-xs rounded-lg border transition-colors ${
                  status === 'current' 
                    ? isDark ? 'border-blue-500 bg-blue-900/50' : 'border-blue-500 bg-blue-50'
                    : status === 'completed'
                    ? isDark ? 'border-green-500 bg-green-900/50' : 'border-green-500 bg-green-50'
                    : isDark ? 'border-gray-600 bg-gray-800' : 'border-gray-200 bg-gray-50'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <Badge variant={getStatusColor(status)} size="sm">
                    {index + 1}
                  </Badge>
                  <span className="text-xs">{step.duration}s</span>
                </div>
                <div className="font-medium text-xs truncate">
                  {step.title}
                </div>
              </button>
            )
          })}
        </div>

        {/* Current Step Details */}
        {currentStep && (
          <div className={`p-4 rounded-lg border-l-4 ${
            currentStep.criticalStep 
              ? 'border-red-500 bg-red-50 dark:bg-red-900/20'
              : 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
          }`}>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-lg font-semibold">{currentStep.title}</h3>
              <div className="flex items-center space-x-2">
                {currentStep.criticalStep && (
                  <Badge variant="error" size="sm">Critical</Badge>
                )}
                <Badge variant="outline" size="sm">
                  Step {currentStepIndex + 1} of {DEMO_SCRIPT.length}
                </Badge>
              </div>
            </div>

            <p className={`mb-4 ${
              isDark ? 'text-gray-300' : 'text-gray-700'
            }`}>
              {currentStep.description}
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Actions */}
              <div>
                <h4 className="font-medium mb-2 text-green-600">Actions</h4>
                <ul className="space-y-1">
                  {currentStep.actions.map((action, index) => (
                    <li key={index} className="text-sm flex items-start">
                      <span className="mr-2">✓</span>
                      <span>{action}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Cues */}
              <div>
                <h4 className="font-medium mb-2 text-blue-600">Cues</h4>
                <ul className="space-y-1">
                  {currentStep.cues.map((cue, index) => (
                    <li key={index} className="text-sm flex items-start">
                      <span className="mr-2">💡</span>
                      <span>{cue}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Fallback Options */}
              <div>
                <h4 className="font-medium mb-2 text-orange-600">Fallbacks</h4>
                <ul className="space-y-1">
                  {currentStep.fallbackOptions.map((fallback, index) => (
                    <li key={index} className="text-sm flex items-start">
                      <span className="mr-2">🛡️</span>
                      <span>{fallback}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* Navigation */}
        <div className="flex items-center justify-between">
          <Button 
            variant="outline" 
            size="sm" 
            onClick={handlePreviousStep}
            disabled={currentStepIndex === 0}
          >
            ← Previous Step
          </Button>

          <div className="flex items-center space-x-2">
            <span className={`text-sm ${
              isDark ? 'text-gray-400' : 'text-gray-600'
            }`}>
              Progress: {completedSteps.size} / {DEMO_SCRIPT.length} steps
            </span>
          </div>

          <Button 
            variant="primary" 
            size="sm" 
            onClick={handleNextStep}
            disabled={currentStepIndex === DEMO_SCRIPT.length - 1 && completedSteps.has(currentStep.id)}
          >
            {currentStepIndex === DEMO_SCRIPT.length - 1 ? 'Complete Demo' : 'Next Step →'}
          </Button>
        </div>

        {/* Demo Complete */}
        {completedSteps.size === DEMO_SCRIPT.length && (
          <div className={`p-4 rounded-lg border ${
            isDark ? 'border-green-600 bg-green-900/20' : 'border-green-500 bg-green-50'
          }`}>
            <div className="flex items-center space-x-2">
              <span className="text-2xl">🎉</span>
              <div>
                <h3 className="font-semibold text-green-600">Demo Complete!</h3>
                <p className="text-sm">
                  Total time: {formatTime(totalElapsed)} 
                  {totalElapsed <= totalDuration && (
                    <span className="text-green-600 ml-2">
                      (Under target time!)
                    </span>
                  )}
                </p>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default DemoScript
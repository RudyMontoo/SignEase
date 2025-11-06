/**
 * Coordinate Utilities
 * ====================
 * 
 * Utilities for converting between different coordinate systems for AR overlay positioning
 */

export interface Point2D {
  x: number
  y: number
}

export interface Point3D extends Point2D {
  z: number
}

export interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
}

export interface HandLandmarks {
  landmarks: Point3D[]
  handedness: 'Left' | 'Right'
  confidence: number
}

/**
 * Convert MediaPipe normalized coordinates (0-1) to pixel coordinates
 */
export const normalizedToPixel = (
  normalizedPoint: Point2D,
  containerWidth: number,
  containerHeight: number
): Point2D => {
  return {
    x: normalizedPoint.x * containerWidth,
    y: normalizedPoint.y * containerHeight
  }
}

/**
 * Convert pixel coordinates to normalized coordinates (0-1)
 */
export const pixelToNormalized = (
  pixelPoint: Point2D,
  containerWidth: number,
  containerHeight: number
): Point2D => {
  return {
    x: pixelPoint.x / containerWidth,
    y: pixelPoint.y / containerHeight
  }
}

/**
 * Calculate the center point of a hand from landmarks
 */
export const getHandCenter = (landmarks: Point3D[]): Point3D => {
  if (landmarks.length === 0) {
    return { x: 0.5, y: 0.5, z: 0 }
  }

  // Use wrist (landmark 0) as the base, but adjust for better visual positioning
  const wrist = landmarks[0]
  const middleFingerMCP = landmarks[9] // Middle finger MCP joint
  
  // Calculate center between wrist and middle finger MCP for better positioning
  return {
    x: (wrist.x + middleFingerMCP.x) / 2,
    y: (wrist.y + middleFingerMCP.y) / 2,
    z: (wrist.z + middleFingerMCP.z) / 2
  }
}

/**
 * Calculate the bounding box of hand landmarks
 */
export const getHandBoundingBox = (landmarks: Point3D[]): BoundingBox => {
  if (landmarks.length === 0) {
    return { x: 0, y: 0, width: 0, height: 0 }
  }

  let minX = landmarks[0].x
  let maxX = landmarks[0].x
  let minY = landmarks[0].y
  let maxY = landmarks[0].y

  landmarks.forEach(landmark => {
    minX = Math.min(minX, landmark.x)
    maxX = Math.max(maxX, landmark.x)
    minY = Math.min(minY, landmark.y)
    maxY = Math.max(maxY, landmark.y)
  })

  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY
  }
}

/**
 * Calculate optimal text position relative to hand
 */
export const calculateTextPosition = (
  handCenter: Point3D,
  handBoundingBox: BoundingBox,
  containerWidth: number,
  containerHeight: number,
  textWidth: number = 100,
  textHeight: number = 40,
  offset: number = 20
): Point2D => {
  // Convert normalized coordinates to pixels
  const centerPixel = normalizedToPixel(handCenter, containerWidth, containerHeight)
  const boundingBoxPixel = {
    x: handBoundingBox.x * containerWidth,
    y: handBoundingBox.y * containerHeight,
    width: handBoundingBox.width * containerWidth,
    height: handBoundingBox.height * containerHeight
  }

  // Try different positions in order of preference
  const positions = [
    // Above hand
    {
      x: centerPixel.x - textWidth / 2,
      y: boundingBoxPixel.y - textHeight - offset,
      priority: 1
    },
    // Below hand
    {
      x: centerPixel.x - textWidth / 2,
      y: boundingBoxPixel.y + boundingBoxPixel.height + offset,
      priority: 2
    },
    // Right of hand
    {
      x: boundingBoxPixel.x + boundingBoxPixel.width + offset,
      y: centerPixel.y - textHeight / 2,
      priority: 3
    },
    // Left of hand
    {
      x: boundingBoxPixel.x - textWidth - offset,
      y: centerPixel.y - textHeight / 2,
      priority: 4
    }
  ]

  // Find the best position that fits within the container
  for (const position of positions.sort((a, b) => a.priority - b.priority)) {
    if (
      position.x >= 0 &&
      position.y >= 0 &&
      position.x + textWidth <= containerWidth &&
      position.y + textHeight <= containerHeight
    ) {
      return { x: position.x, y: position.y }
    }
  }

  // Fallback: clamp to container bounds
  const fallbackX = Math.max(0, Math.min(centerPixel.x - textWidth / 2, containerWidth - textWidth))
  const fallbackY = Math.max(0, Math.min(centerPixel.y - textHeight / 2, containerHeight - textHeight))

  return { x: fallbackX, y: fallbackY }
}

/**
 * Calculate smooth interpolation between two points
 */
export const interpolatePoint = (
  from: Point2D,
  to: Point2D,
  factor: number
): Point2D => {
  const clampedFactor = Math.max(0, Math.min(1, factor))
  
  return {
    x: from.x + (to.x - from.x) * clampedFactor,
    y: from.y + (to.y - from.y) * clampedFactor
  }
}

/**
 * Calculate distance between two points
 */
export const getDistance = (point1: Point2D, point2: Point2D): number => {
  const dx = point2.x - point1.x
  const dy = point2.y - point1.y
  return Math.sqrt(dx * dx + dy * dy)
}

/**
 * Check if a point is within the viewport bounds
 */
export const isPointInBounds = (
  point: Point2D,
  width: number,
  height: number,
  margin: number = 0
): boolean => {
  return (
    point.x >= margin &&
    point.y >= margin &&
    point.x <= width - margin &&
    point.y <= height - margin
  )
}

/**
 * Apply smoothing to reduce jitter in hand tracking
 */
export class PositionSmoother {
  private history: Point2D[] = []
  private maxHistorySize: number
  private smoothingFactor: number

  constructor(maxHistorySize: number = 5, smoothingFactor: number = 0.3) {
    this.maxHistorySize = maxHistorySize
    this.smoothingFactor = smoothingFactor
  }

  smooth(newPoint: Point2D): Point2D {
    this.history.push(newPoint)
    
    if (this.history.length > this.maxHistorySize) {
      this.history.shift()
    }

    if (this.history.length === 1) {
      return newPoint
    }

    // Calculate weighted average with more weight on recent positions
    let totalWeight = 0
    let weightedX = 0
    let weightedY = 0

    this.history.forEach((point, index) => {
      const weight = Math.pow(this.smoothingFactor, this.history.length - 1 - index)
      totalWeight += weight
      weightedX += point.x * weight
      weightedY += point.y * weight
    })

    return {
      x: weightedX / totalWeight,
      y: weightedY / totalWeight
    }
  }

  reset(): void {
    this.history = []
  }
}

/**
 * Calculate hand velocity for motion-based effects
 */
export const calculateVelocity = (
  currentPosition: Point2D,
  previousPosition: Point2D,
  deltaTime: number
): Point2D => {
  if (deltaTime <= 0) {
    return { x: 0, y: 0 }
  }

  return {
    x: (currentPosition.x - previousPosition.x) / deltaTime,
    y: (currentPosition.y - previousPosition.y) / deltaTime
  }
}

/**
 * Get hand landmark by name for easier access
 */
export const getHandLandmarkByName = (
  landmarks: Point3D[],
  landmarkName: keyof typeof HAND_LANDMARK_INDICES
): Point3D | null => {
  const index = HAND_LANDMARK_INDICES[landmarkName]
  return landmarks[index] || null
}

/**
 * MediaPipe hand landmark indices
 */
export const HAND_LANDMARK_INDICES = {
  WRIST: 0,
  THUMB_CMC: 1,
  THUMB_MCP: 2,
  THUMB_IP: 3,
  THUMB_TIP: 4,
  INDEX_FINGER_MCP: 5,
  INDEX_FINGER_PIP: 6,
  INDEX_FINGER_DIP: 7,
  INDEX_FINGER_TIP: 8,
  MIDDLE_FINGER_MCP: 9,
  MIDDLE_FINGER_PIP: 10,
  MIDDLE_FINGER_DIP: 11,
  MIDDLE_FINGER_TIP: 12,
  RING_FINGER_MCP: 13,
  RING_FINGER_PIP: 14,
  RING_FINGER_DIP: 15,
  RING_FINGER_TIP: 16,
  PINKY_MCP: 17,
  PINKY_PIP: 18,
  PINKY_DIP: 19,
  PINKY_TIP: 20
} as const
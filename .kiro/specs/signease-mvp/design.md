# Design Document

## Overview

SignEase is a real-time sign language translation system built using a hybrid architecture that combines browser-based hand detection with GPU-accelerated gesture classification. The system leverages MediaPipe for robust hand tracking and a custom PyTorch neural network trained on ASL data for accurate gesture recognition. The architecture is designed for low latency (<500ms end-to-end), high accuracy (90%+), and scalability to support additional gestures and features.

### Key Design Principles

1. **Separation of Concerns**: Hand detection (MediaPipe) and gesture classification (PyTorch) are independent modules
2. **GPU Acceleration**: Leverage RTX 5060 for both training and real-time inference
3. **Graceful Degradation**: Fallback mechanisms ensure functionality even if components fail
4. **Real-time Performance**: Optimize for <500ms latency from gesture to text display
5. **Modularity**: Easy to swap models, add features, or extend gesture vocabulary

## Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User's Browser                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              React Frontend (Vite)                     │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │ │
│  │  │   Webcam     │  │  MediaPipe   │  │   UI Layer  │ │ │
│  │  │   Capture    │→ │  Hands.js    │→ │  (Display)  │ │ │
│  │  └──────────────┘  └──────────────┘  └─────────────┘ │ │
│  │         ↓                  ↓                  ↑        │ │
│  │    Video Feed      Hand Landmarks         Text/Voice  │ │
│  └────────────────────────────────────────────────────────┘ │
│                            ↓ HTTP/WebSocket ↑               │
└────────────────────────────────────────────────────────────┘
                             ↓                ↑
┌────────────────────────────────────────────────────────────┐
│                    Flask Backend (Python)                   │
│  ┌────────────────────────────────────────────────────────┐│
│  │              API Layer (Flask + CORS)                   ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  ││
│  │  │   /predict   │  │   /health    │  │  /metrics   │  ││
│  │  │   endpoint   │  │   endpoint   │  │  endpoint   │  ││
│  │  └──────────────┘  └──────────────┘  └─────────────┘  ││
│  └────────────────────────────────────────────────────────┘│
│                            ↓                                │
│  ┌────────────────────────────────────────────────────────┐│
│  │           ML Inference Engine (PyTorch)                 ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  ││
│  │  │ Model Loader │→ │  GPU Tensor  │→ │ Classifier  │  ││
│  │  │   (.pth)     │  │  Processing  │  │   Output    │  ││
│  │  └──────────────┘  └──────────────┘  └─────────────┘  ││
│  └────────────────────────────────────────────────────────┘│
│                            ↓                                │
│                    RTX 5060 GPU (CUDA)                      │
└────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

```
1. User performs gesture in front of webcam
2. React captures video frame (30 FPS)
3. MediaPipe.js detects hand and extracts 21 landmarks
4. Frontend sends landmarks to Flask backend via HTTP POST
5. Backend preprocesses landmarks and converts to PyTorch tensor
6. PyTorch model runs inference on GPU (<50ms)
7. Backend returns prediction with confidence score
8. Frontend displays letter and updates sentence
9. User clicks "Speak" → Web Speech API converts text to speech
```

## Components and Interfaces

### 1. Frontend Components (React + Vite)

#### 1.1 WebcamCapture Component

**Purpose**: Manages webcam access and video stream display
**Dependencies**: MediaPipe.js, React hooks (useRef, useEffect)

```typescript
interface WebcamCaptureProps {
  onLandmarksDetected: (landmarks: HandLandmarks[]) => void;
  isActive: boolean;
  showOverlay: boolean;
}

interface HandLandmarks {
  landmarks: Array<{x: number, y: number, z: number}>;
  handedness: 'Left' | 'Right';
  confidence: number;
}
```

**Key Methods**:
- `startCamera()`: Initialize webcam with optimal settings (1280x720, 30fps)
- `stopCamera()`: Clean up resources and stop video stream
- `processFrame()`: Run MediaPipe hand detection on each frame
- `drawLandmarks()`: Overlay hand landmarks on video (debug mode)

#### 1.2 GestureDisplay Component

**Purpose**: Shows detected gestures, builds sentences, manages text output
**Dependencies**: Web Speech API, React state management

```typescript
interface GestureDisplayProps {
  currentGesture: string;
  confidence: number;
  sentence: string;
  onClear: () => void;
  onSpeak: () => void;
  onAddToSentence: (letter: string) => void;
}

interface GestureState {
  currentLetter: string;
  confidence: number;
  sentence: string;
  isProcessing: boolean;
  lastDetectionTime: number;
}
```

**Key Features**:
- Real-time gesture display with confidence indicator
- Sentence building with space/delete functionality
- Text-to-speech integration
- Gesture history (last 10 detections)
- Auto-add letters after stable detection (1 second threshold)

#### 1.3 AROverlay Component

**Purpose**: Floating text overlay on video feed for AR effect
**Dependencies**: CSS transforms, React Portal

```typescript
interface AROverlayProps {
  text: string;
  handPosition: {x: number, y: number};
  isVisible: boolean;
  animationType: 'fade' | 'slide' | 'bounce';
}
```

**Key Features**:
- Position text relative to detected hand
- Smooth animations and transitions
- Multiple display modes (floating, fixed, following)
- Customizable styling and colors

#### 1.4 ControlPanel Component

**Purpose**: User controls and settings
**Dependencies**: React state, local storage

```typescript
interface ControlPanelProps {
  isRecording: boolean;
  onToggleRecording: () => void;
  onClearSentence: () => void;
  onToggleARMode: () => void;
  settings: AppSettings;
  onSettingsChange: (settings: AppSettings) => void;
}

interface AppSettings {
  confidenceThreshold: number;
  detectionDelay: number;
  arMode: boolean;
  darkMode: boolean;
  voiceEnabled: boolean;
  gestureGuide: boolean;
}
```

### 2. Backend Components (Flask + PyTorch)

#### 2.1 Flask API Server

**Purpose**: HTTP API for gesture prediction and system health
**Dependencies**: Flask, Flask-CORS, PyTorch

```python
# API Endpoints
@app.route('/predict', methods=['POST'])
def predict_gesture():
    """
    Input: JSON with hand landmarks
    Output: JSON with gesture prediction and confidence
    """
    pass

@app.route('/health', methods=['GET'])
def health_check():
    """
    Output: System status, GPU availability, model loaded
    """
    pass

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Output: Performance metrics, prediction counts, accuracy stats
    """
    pass

@app.route('/retrain', methods=['POST'])
def trigger_retrain():
    """
    Input: New training data
    Output: Training job status
    """
    pass
```

**Request/Response Schemas**:

```python
# Prediction Request
{
    "landmarks": [
        {"x": 0.5, "y": 0.3, "z": 0.1},
        # ... 20 more landmarks
    ],
    "handedness": "Right",
    "timestamp": 1699123456789
}

# Prediction Response
{
    "gesture": "A",
    "confidence": 0.94,
    "alternatives": [
        {"gesture": "S", "confidence": 0.03},
        {"gesture": "E", "confidence": 0.02}
    ],
    "processing_time_ms": 23,
    "model_version": "v1.2.0"
}
```

#### 2.2 ML Inference Engine

**Purpose**: PyTorch model loading, preprocessing, and GPU inference
**Dependencies**: PyTorch, CUDA, NumPy

```python
class ASLInferenceEngine:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model = self.load_model(model_path)
        self.preprocessor = LandmarkPreprocessor()
        self.class_names = self.load_class_names()
        
    def predict(self, landmarks: List[Dict]) -> Dict:
        """
        Process landmarks and return prediction
        """
        pass
        
    def preprocess_landmarks(self, landmarks: List[Dict]) -> torch.Tensor:
        """
        Convert landmarks to model input format
        """
        pass
        
    def postprocess_output(self, output: torch.Tensor) -> Dict:
        """
        Convert model output to human-readable format
        """
        pass
```

**Model Architecture**:

```python
class ASLClassifier(nn.Module):
    def __init__(self, input_size=63, hidden_sizes=[128, 64, 32], num_classes=29):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(0.3))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(0.2))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        self.layers.append(nn.Softmax(dim=1))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

#### 2.3 Data Processing Pipeline

**Purpose**: Landmark normalization, feature extraction, data augmentation
**Dependencies**: NumPy, scikit-learn

```python
class LandmarkPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_extractor = HandFeatureExtractor()
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks relative to wrist position and hand size
        """
        pass
    
    def extract_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract geometric features (angles, distances, ratios)
        """
        pass
    
    def augment_data(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation (rotation, scaling, noise)
        """
        pass
```

### 3. Model Training Pipeline

#### 3.1 Dataset Management

**Purpose**: Load, preprocess, and manage ASL datasets
**Dependencies**: PyTorch Dataset, PIL, pandas

```python
class ASLDataset(Dataset):
    def __init__(self, data_path: str, transform=None, mode='landmarks'):
        self.data_path = data_path
        self.transform = transform
        self.mode = mode  # 'landmarks' or 'images'
        self.samples = self.load_samples()
        self.class_to_idx = self.create_class_mapping()
    
    def __getitem__(self, idx):
        if self.mode == 'landmarks':
            return self.get_landmark_sample(idx)
        else:
            return self.get_image_sample(idx)
    
    def load_samples(self):
        """Load dataset from Kaggle ASL Alphabet dataset"""
        pass
```

#### 3.2 Training Configuration

**Purpose**: Training hyperparameters and experiment tracking
**Dependencies**: PyTorch, Weights & Biases (optional)

```python
@dataclass
class TrainingConfig:
    # Model parameters
    model_type: str = 'mlp'  # 'mlp' or 'cnn'
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout_rate: float = 0.3
    
    # Training parameters
    batch_size: int = 128
    learning_rate: float = 0.001
    num_epochs: int = 50
    weight_decay: float = 1e-4
    
    # Hardware
    device: str = 'cuda'
    num_workers: int = 4
    pin_memory: bool = True
    
    # Paths
    data_path: str = './data/asl_alphabet'
    model_save_path: str = './models/asl_classifier.pth'
    log_dir: str = './logs'
    
    # Early stopping
    patience: int = 10
    min_delta: float = 0.001
```

## Data Flow and State Management

### Frontend State Management

```typescript
// Global App State
interface AppState {
  // Camera state
  isCameraActive: boolean;
  cameraError: string | null;
  
  // Detection state
  currentGesture: string;
  confidence: number;
  isProcessing: boolean;
  
  // Sentence building
  sentence: string;
  gestureHistory: GestureDetection[];
  
  // UI state
  arMode: boolean;
  darkMode: boolean;
  showSettings: boolean;
  
  // Backend connection
  isConnected: boolean;
  apiError: string | null;
}

// Actions
type AppAction = 
  | { type: 'CAMERA_START' }
  | { type: 'CAMERA_STOP' }
  | { type: 'GESTURE_DETECTED'; payload: GestureDetection }
  | { type: 'ADD_TO_SENTENCE'; payload: string }
  | { type: 'CLEAR_SENTENCE' }
  | { type: 'TOGGLE_AR_MODE' }
  | { type: 'SET_ERROR'; payload: string };
```

### Backend State Management

```python
# Application State
class AppState:
    def __init__(self):
        self.model_loaded = False
        self.gpu_available = torch.cuda.is_available()
        self.prediction_count = 0
        self.average_inference_time = 0.0
        self.model_version = "v1.0.0"
        self.last_prediction_time = None
        
    def update_metrics(self, inference_time: float):
        """Update performance metrics"""
        self.prediction_count += 1
        self.average_inference_time = (
            (self.average_inference_time * (self.prediction_count - 1) + inference_time)
            / self.prediction_count
        )
        self.last_prediction_time = time.time()
```

## Performance Requirements

### Latency Requirements

| Component | Target Latency | Maximum Acceptable |
|-----------|----------------|-------------------|
| MediaPipe Hand Detection | <33ms (30 FPS) | <50ms |
| Backend API Call | <100ms | <200ms |
| PyTorch Inference | <50ms | <100ms |
| End-to-End (Gesture → Display) | <300ms | <500ms |
| Text-to-Speech | <200ms | <500ms |

### Accuracy Requirements

| Metric | Target | Minimum Acceptable |
|--------|--------|--------------------|
| Gesture Classification | >92% | >88% |
| Hand Detection Rate | >95% | >90% |
| False Positive Rate | <5% | <10% |
| Sentence Accuracy (5 letters) | >85% | >80% |

### Resource Requirements

| Resource | Development | Production |
|----------|-------------|------------|
| GPU Memory | 4GB (RTX 5060) | 2GB minimum |
| System RAM | 16GB | 8GB minimum |
| CPU | Intel Ultra 9 285HX | 4+ cores |
| Storage | 10GB (with datasets) | 2GB |
| Network | Local development | 10 Mbps minimum |

## Security and Privacy

### Data Privacy

1. **No Data Storage**: Landmarks are processed in real-time and not stored
2. **Local Processing**: All video processing happens locally (no cloud)
3. **Minimal Data Transfer**: Only landmark coordinates sent to backend
4. **User Consent**: Clear camera permission requests and usage explanation

### Security Measures

1. **CORS Configuration**: Restrict API access to frontend domain
2. **Input Validation**: Validate all landmark data before processing
3. **Rate Limiting**: Prevent API abuse with request throttling
4. **Error Handling**: Graceful degradation without exposing internals

```python
# Input validation example
def validate_landmarks(landmarks):
    if not isinstance(landmarks, list) or len(landmarks) != 21:
        raise ValueError("Invalid landmarks format")
    
    for landmark in landmarks:
        if not all(key in landmark for key in ['x', 'y', 'z']):
            raise ValueError("Missing landmark coordinates")
        
        if not all(0 <= landmark[key] <= 1 for key in ['x', 'y']):
            raise ValueError("Landmark coordinates out of range")
```

## Error Handling and Fallbacks

### Frontend Error Handling

```typescript
// Error boundary for React components
class GestureErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  
  componentDidCatch(error, errorInfo) {
    console.error('Gesture detection error:', error, errorInfo);
    // Fallback to basic gesture recognition
    this.props.onFallback();
  }
}

// API error handling
const handleAPIError = (error) => {
  if (error.code === 'NETWORK_ERROR') {
    // Switch to offline mode
    setOfflineMode(true);
  } else if (error.code === 'MODEL_ERROR') {
    // Use fallback gesture recognition
    useFallbackModel();
  }
};
```

### Backend Error Handling

```python
# Graceful error handling
@app.errorhandler(Exception)
def handle_error(error):
    if isinstance(error, torch.cuda.OutOfMemoryError):
        # Fall back to CPU inference
        return fallback_to_cpu_inference()
    elif isinstance(error, ModelLoadError):
        # Use backup model
        return load_backup_model()
    else:
        # Generic error response
        return jsonify({
            'error': 'Internal server error',
            'fallback_available': True
        }), 500
```

### Fallback Strategies

1. **GPU → CPU Fallback**: If GPU fails, switch to CPU inference
2. **Custom Model → MediaPipe Fallback**: If custom model fails, use MediaPipe GestureRecognizer
3. **Real-time → Batch Processing**: If real-time fails, process in batches
4. **Network → Offline Mode**: If backend fails, use client-side processing

## Testing Strategy

### Unit Tests

```python
# Backend model tests
class TestASLClassifier(unittest.TestCase):
    def setUp(self):
        self.model = ASLClassifier()
        self.sample_landmarks = generate_test_landmarks()
    
    def test_model_forward_pass(self):
        output = self.model(self.sample_landmarks)
        self.assertEqual(output.shape, (1, 29))
        self.assertTrue(torch.allclose(output.sum(dim=1), torch.ones(1)))
    
    def test_preprocessing(self):
        preprocessor = LandmarkPreprocessor()
        processed = preprocessor.normalize_landmarks(self.sample_landmarks)
        self.assertEqual(processed.shape, (63,))
```

```typescript
// Frontend component tests
describe('GestureDisplay', () => {
  it('should display current gesture', () => {
    render(<GestureDisplay currentGesture="A" confidence={0.95} />);
    expect(screen.getByText('A')).toBeInTheDocument();
    expect(screen.getByText('95%')).toBeInTheDocument();
  });
  
  it('should build sentence correctly', () => {
    const { rerender } = render(<GestureDisplay sentence="" />);
    fireEvent.click(screen.getByText('Add to Sentence'));
    rerender(<GestureDisplay sentence="A" />);
    expect(screen.getByText('A')).toBeInTheDocument();
  });
});
```

### Integration Tests

```python
# API integration tests
class TestAPIIntegration(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.sample_request = {
            "landmarks": generate_test_landmarks(),
            "handedness": "Right"
        }
    
    def test_prediction_endpoint(self):
        response = self.client.post('/predict', 
                                  json=self.sample_request)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('gesture', data)
        self.assertIn('confidence', data)
```

### Performance Tests

```python
# Load testing
def test_inference_performance():
    model = ASLClassifier()
    landmarks = torch.randn(100, 63)  # Batch of 100
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model(landmarks)
    end_time = time.time()
    
    avg_time_per_sample = (end_time - start_time) / 100
    assert avg_time_per_sample < 0.01  # <10ms per sample
```

### End-to-End Tests

```typescript
// E2E testing with Playwright
test('complete gesture recognition flow', async ({ page }) => {
  await page.goto('http://localhost:3000');
  
  // Start camera
  await page.click('[data-testid="start-camera"]');
  await expect(page.locator('video')).toBeVisible();
  
  // Mock gesture detection
  await page.evaluate(() => {
    window.mockGestureDetection('A', 0.95);
  });
  
  // Verify display
  await expect(page.locator('[data-testid="current-gesture"]')).toHaveText('A');
  
  // Add to sentence
  await page.click('[data-testid="add-to-sentence"]');
  await expect(page.locator('[data-testid="sentence"]')).toHaveText('A');
  
  // Test speech
  await page.click('[data-testid="speak-button"]');
  // Verify speech API was called
});
```

## Deployment Architecture

### Development Environment

```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
    environment:
      - REACT_APP_API_URL=http://localhost:5000
  
  backend:
    build: ./backend
    ports:
      - "5000:5000"
    volumes:
      - ./backend:/app
      - ./models:/app/models
    environment:
      - CUDA_VISIBLE_DEVICES=0
    runtime: nvidia
```

### Production Deployment

```yaml
# Frontend (Vercel)
# vercel.json
{
  "builds": [
    {
      "src": "package.json",
      "use": "@vercel/static-build"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "https://signease-api.herokuapp.com/api/$1"
    },
    {
      "src": "/(.*)",
      "dest": "/index.html"
    }
  ]
}

# Backend (Local/Cloud GPU instance)
# Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

## Monitoring and Observability

### Metrics Collection

```python
# Performance metrics
class MetricsCollector:
    def __init__(self):
        self.prediction_times = []
        self.accuracy_scores = []
        self.error_counts = defaultdict(int)
    
    def record_prediction(self, time_ms: float, confidence: float):
        self.prediction_times.append(time_ms)
        if confidence > 0.8:
            self.accuracy_scores.append(1.0)
        else:
            self.accuracy_scores.append(0.0)
    
    def get_metrics(self):
        return {
            'avg_prediction_time': np.mean(self.prediction_times),
            'p95_prediction_time': np.percentile(self.prediction_times, 95),
            'accuracy_rate': np.mean(self.accuracy_scores),
            'total_predictions': len(self.prediction_times),
            'error_counts': dict(self.error_counts)
        }
```

### Logging Configuration

```python
# logging_config.py
import logging
import sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('signease.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Separate logger for ML operations
    ml_logger = logging.getLogger('ml_inference')
    ml_handler = logging.FileHandler('ml_inference.log')
    ml_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    ml_logger.addHandler(ml_handler)
```

This completes the comprehensive design document. The architecture is optimized for your RTX 5060 GPU and provides a solid foundation for building the MVP within 24 hours while maintaining scalability for future enhancements.

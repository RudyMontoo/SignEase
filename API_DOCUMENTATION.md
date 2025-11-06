# SignEase MVP API Documentation

Complete API reference for the SignEase MVP backend services.

## 🌐 Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://signease-api.vercel.app`

## 🔐 Authentication

Currently, the API does not require authentication. All endpoints are publicly accessible for the MVP version.

## 📊 API Overview

### Core Endpoints
- **Gesture Recognition**: `/api/predict` - Real-time ASL gesture prediction
- **Health Check**: `/api/health` - System health and status
- **Performance Metrics**: `/api/metrics` - System performance data
- **Model Information**: `/api/model/info` - ML model details

### Response Format
All API responses follow a consistent JSON format:

```json
{
  "success": true,
  "data": { ... },
  "message": "Success message",
  "timestamp": "2024-01-01T12:00:00Z",
  "request_id": "req_123456789"
}
```

Error responses:
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": { ... }
  },
  "timestamp": "2024-01-01T12:00:00Z",
  "request_id": "req_123456789"
}
```

## 🤖 Gesture Recognition API

### Predict Gesture

Analyzes hand landmarks and returns ASL gesture prediction.

**Endpoint**: `POST /api/predict`

**Request Body**:
```json
{
  "landmarks": [
    [0.5, 0.3, 0.0],
    [0.45, 0.35, -0.01],
    ...
  ],
  "confidence_threshold": 0.7,
  "max_alternatives": 3,
  "enable_caching": true
}
```

**Parameters**:
- `landmarks` (required): Array of 21 hand landmark points, each with [x, y, z] coordinates
- `confidence_threshold` (optional): Minimum confidence for prediction (default: 0.7)
- `max_alternatives` (optional): Maximum alternative predictions to return (default: 3)
- `enable_caching` (optional): Enable result caching for identical inputs (default: true)

**Response**:
```json
{
  "success": true,
  "data": {
    "prediction": "A",
    "confidence": 0.95,
    "alternatives": [
      {
        "prediction": "S",
        "confidence": 0.12
      },
      {
        "prediction": "T",
        "confidence": 0.08
      }
    ],
    "processing_time": 45.2,
    "model_version": "1.0.0",
    "cached": false
  },
  "message": "Gesture predicted successfully",
  "timestamp": "2024-01-01T12:00:00Z",
  "request_id": "req_123456789"
}
```

**Error Responses**:

*400 Bad Request*:
```json
{
  "success": false,
  "error": {
    "code": "INVALID_LANDMARKS",
    "message": "Invalid landmark data provided",
    "details": {
      "expected_points": 21,
      "received_points": 15,
      "required_format": "[[x, y, z], ...]"
    }
  }
}
```

*422 Unprocessable Entity*:
```json
{
  "success": false,
  "error": {
    "code": "CONFIDENCE_TOO_LOW",
    "message": "Prediction confidence below threshold",
    "details": {
      "confidence": 0.45,
      "threshold": 0.7,
      "suggestion": "Lower threshold or improve hand positioning"
    }
  }
}
```

*500 Internal Server Error*:
```json
{
  "success": false,
  "error": {
    "code": "MODEL_ERROR",
    "message": "Model inference failed",
    "details": {
      "error_type": "RuntimeError",
      "suggestion": "Try again or contact support"
    }
  }
}
```

### Batch Predict

Process multiple gesture predictions in a single request for improved performance.

**Endpoint**: `POST /api/predict/batch`

**Request Body**:
```json
{
  "requests": [
    {
      "id": "req_1",
      "landmarks": [[0.5, 0.3, 0.0], ...],
      "confidence_threshold": 0.7
    },
    {
      "id": "req_2", 
      "landmarks": [[0.6, 0.4, 0.1], ...],
      "confidence_threshold": 0.8
    }
  ]
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "id": "req_1",
        "prediction": "A",
        "confidence": 0.95,
        "alternatives": [...],
        "processing_time": 23.1
      },
      {
        "id": "req_2",
        "prediction": "B", 
        "confidence": 0.88,
        "alternatives": [...],
        "processing_time": 21.8
      }
    ],
    "total_processing_time": 44.9,
    "batch_size": 2
  }
}
```

## 🏥 Health Check API

### System Health

Check the overall health and status of the API service.

**Endpoint**: `GET /api/health`

**Response**:
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "uptime": 3600.5,
    "timestamp": "2024-01-01T12:00:00Z",
    "components": {
      "model": {
        "status": "healthy",
        "loaded": true,
        "version": "1.0.0",
        "last_prediction": "2024-01-01T11:59:30Z"
      },
      "gpu": {
        "status": "available",
        "memory_used": 1.2,
        "memory_total": 8.0,
        "utilization": 15
      },
      "cache": {
        "status": "healthy",
        "hit_rate": 0.85,
        "size": 1024,
        "max_size": 10000
      }
    }
  }
}
```

**Health Status Values**:
- `healthy`: All systems operational
- `degraded`: Some non-critical issues
- `unhealthy`: Critical issues detected

### Detailed Health Check

Get comprehensive system diagnostics.

**Endpoint**: `GET /api/health/detailed`

**Response**:
```json
{
  "success": true,
  "data": {
    "overall_status": "healthy",
    "checks": [
      {
        "name": "Model Loading",
        "status": "pass",
        "duration": 0.5,
        "details": "Model loaded successfully"
      },
      {
        "name": "GPU Availability", 
        "status": "pass",
        "duration": 0.1,
        "details": "CUDA device available"
      },
      {
        "name": "Memory Usage",
        "status": "pass", 
        "duration": 0.1,
        "details": "Memory usage within limits"
      }
    ],
    "system_info": {
      "python_version": "3.9.7",
      "tensorflow_version": "2.13.0",
      "cuda_version": "11.8",
      "platform": "Linux-5.4.0-x86_64"
    }
  }
}
```

## 📈 Performance Metrics API

### Current Metrics

Get real-time performance metrics.

**Endpoint**: `GET /api/metrics`

**Response**:
```json
{
  "success": true,
  "data": {
    "requests": {
      "total": 15420,
      "per_second": 25.3,
      "per_minute": 1518,
      "per_hour": 91080
    },
    "latency": {
      "average": 47.8,
      "p50": 42.1,
      "p95": 89.2,
      "p99": 156.7
    },
    "accuracy": {
      "overall": 0.9957,
      "last_100": 0.9960,
      "by_letter": {
        "A": 0.998,
        "B": 0.995,
        "C": 0.992
      }
    },
    "resources": {
      "cpu_usage": 45.2,
      "memory_usage": 2.1,
      "gpu_memory": 1.8,
      "gpu_utilization": 23.5
    },
    "cache": {
      "hit_rate": 0.85,
      "size": 1024,
      "evictions": 45
    }
  }
}
```

### Historical Metrics

Get performance metrics over time.

**Endpoint**: `GET /api/metrics/history`

**Query Parameters**:
- `period`: Time period (`1h`, `24h`, `7d`, `30d`)
- `metric`: Specific metric (`latency`, `accuracy`, `requests`)
- `granularity`: Data granularity (`1m`, `5m`, `1h`)

**Example**: `GET /api/metrics/history?period=24h&metric=latency&granularity=1h`

**Response**:
```json
{
  "success": true,
  "data": {
    "metric": "latency",
    "period": "24h",
    "granularity": "1h",
    "data_points": [
      {
        "timestamp": "2024-01-01T00:00:00Z",
        "value": 45.2,
        "count": 1250
      },
      {
        "timestamp": "2024-01-01T01:00:00Z", 
        "value": 48.7,
        "count": 1180
      }
    ],
    "summary": {
      "min": 42.1,
      "max": 67.3,
      "average": 47.8,
      "total_requests": 28450
    }
  }
}
```

## 🧠 Model Information API

### Model Details

Get information about the current ML model.

**Endpoint**: `GET /api/model/info`

**Response**:
```json
{
  "success": true,
  "data": {
    "name": "ASL Gesture Classifier",
    "version": "1.0.0",
    "architecture": "CNN",
    "input_shape": [21, 3],
    "output_classes": 26,
    "training_data": {
      "dataset_size": 87000,
      "validation_accuracy": 0.9957,
      "training_date": "2024-01-01",
      "epochs": 100
    },
    "supported_gestures": [
      "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
      "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", 
      "U", "V", "W", "X", "Y", "Z"
    ],
    "performance": {
      "inference_time": "~45ms",
      "memory_usage": "~1.2GB",
      "accuracy": 0.9957
    }
  }
}
```

### Model Statistics

Get detailed model performance statistics.

**Endpoint**: `GET /api/model/stats`

**Response**:
```json
{
  "success": true,
  "data": {
    "accuracy_by_letter": {
      "A": {"accuracy": 0.998, "samples": 3450, "errors": 7},
      "B": {"accuracy": 0.995, "samples": 3420, "errors": 17},
      "C": {"accuracy": 0.992, "samples": 3380, "errors": 27}
    },
    "confusion_matrix": {
      "A": {"A": 3443, "S": 4, "T": 3},
      "B": {"B": 3403, "P": 12, "R": 5}
    },
    "performance_trends": {
      "daily_accuracy": [0.995, 0.996, 0.997, 0.996],
      "response_times": [45.2, 46.1, 44.8, 47.3]
    }
  }
}
```

## 🔧 Configuration API

### Get Configuration

Get current system configuration.

**Endpoint**: `GET /api/config`

**Response**:
```json
{
  "success": true,
  "data": {
    "model": {
      "confidence_threshold": 0.7,
      "max_alternatives": 3,
      "enable_caching": true,
      "cache_ttl": 300
    },
    "performance": {
      "max_batch_size": 10,
      "request_timeout": 30,
      "enable_gpu": true,
      "max_concurrent_requests": 100
    },
    "features": {
      "batch_prediction": true,
      "detailed_metrics": true,
      "model_stats": true
    }
  }
}
```

## 📊 Rate Limiting

### Rate Limits
- **Default**: 1000 requests per hour per IP
- **Burst**: 100 requests per minute per IP
- **Batch**: 10 requests per batch maximum

### Rate Limit Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
X-RateLimit-Retry-After: 3600
```

### Rate Limit Exceeded Response
```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Too many requests",
    "details": {
      "limit": 1000,
      "window": 3600,
      "retry_after": 1800
    }
  }
}
```

## 🚨 Error Codes

### Client Errors (4xx)
- `400 BAD_REQUEST`: Invalid request format or parameters
- `422 UNPROCESSABLE_ENTITY`: Valid format but unprocessable content
- `429 TOO_MANY_REQUESTS`: Rate limit exceeded

### Server Errors (5xx)
- `500 INTERNAL_SERVER_ERROR`: Unexpected server error
- `503 SERVICE_UNAVAILABLE`: Service temporarily unavailable
- `504 GATEWAY_TIMEOUT`: Request timeout

### Custom Error Codes
- `INVALID_LANDMARKS`: Landmark data format is incorrect
- `CONFIDENCE_TOO_LOW`: Prediction confidence below threshold
- `MODEL_ERROR`: ML model inference failed
- `GPU_ERROR`: GPU processing error
- `CACHE_ERROR`: Caching system error

## 📝 Request/Response Examples

### cURL Examples

**Basic Prediction**:
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "landmarks": [
      [0.5, 0.3, 0.0],
      [0.45, 0.35, -0.01],
      [0.4, 0.4, -0.02]
    ],
    "confidence_threshold": 0.7
  }'
```

**Health Check**:
```bash
curl -X GET "http://localhost:8000/api/health"
```

**Performance Metrics**:
```bash
curl -X GET "http://localhost:8000/api/metrics"
```

### JavaScript Examples

**Using Fetch API**:
```javascript
// Predict gesture
const predictGesture = async (landmarks) => {
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        landmarks,
        confidence_threshold: 0.7,
        max_alternatives: 3
      })
    })
    
    const result = await response.json()
    
    if (result.success) {
      return result.data
    } else {
      throw new Error(result.error.message)
    }
  } catch (error) {
    console.error('Prediction failed:', error)
    throw error
  }
}

// Check health
const checkHealth = async () => {
  const response = await fetch('/api/health')
  const health = await response.json()
  return health.data
}
```

**Using Axios**:
```javascript
import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000/api',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// Predict gesture
const predictGesture = async (landmarks) => {
  try {
    const response = await api.post('/predict', {
      landmarks,
      confidence_threshold: 0.7
    })
    return response.data.data
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.error.message)
    }
    throw error
  }
}
```

### Python Examples

**Using requests**:
```python
import requests
import json

API_BASE = "http://localhost:8000/api"

def predict_gesture(landmarks, confidence_threshold=0.7):
    """Predict ASL gesture from landmarks."""
    url = f"{API_BASE}/predict"
    payload = {
        "landmarks": landmarks,
        "confidence_threshold": confidence_threshold,
        "max_alternatives": 3
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        if result["success"]:
            return result["data"]
        else:
            raise Exception(result["error"]["message"])
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {e}")

def check_health():
    """Check API health status."""
    url = f"{API_BASE}/health"
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return response.json()["data"]

# Example usage
landmarks = [
    [0.5, 0.3, 0.0],
    [0.45, 0.35, -0.01],
    # ... more landmarks
]

try:
    prediction = predict_gesture(landmarks)
    print(f"Predicted: {prediction['prediction']} (confidence: {prediction['confidence']:.2f})")
except Exception as e:
    print(f"Error: {e}")
```

## 🔍 Testing the API

### Interactive API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Postman Collection
Import the SignEase API collection for easy testing:
```json
{
  "info": {
    "name": "SignEase MVP API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Predict Gesture",
      "request": {
        "method": "POST",
        "header": [
          {
            "key": "Content-Type",
            "value": "application/json"
          }
        ],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"landmarks\": [[0.5, 0.3, 0.0]],\n  \"confidence_threshold\": 0.7\n}"
        },
        "url": {
          "raw": "{{baseUrl}}/api/predict",
          "host": ["{{baseUrl}}"],
          "path": ["api", "predict"]
        }
      }
    }
  ],
  "variable": [
    {
      "key": "baseUrl",
      "value": "http://localhost:8000"
    }
  ]
}
```

## 📞 Support

### API Issues
- **Documentation**: Check this guide and inline API docs
- **GitHub Issues**: Create issue with API details
- **Discord**: Join our community for real-time help
- **Email**: api@signease.dev

### Rate Limit Increases
For higher rate limits, contact: ratelimits@signease.dev

---

**SignEase MVP API - Powering real-time ASL recognition** 🤖

*Making communication accessible through technology*
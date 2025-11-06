#!/usr/bin/env python3
"""
Performance Optimizer
====================

Advanced performance optimization utilities for the SignEase ASL Recognition API.
Includes batching, caching, memory profiling, and concurrent request handling.
"""

import time
import threading
import queue
import hashlib
import psutil
import gc
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from functools import lru_cache
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

class RequestCache:
    """LRU cache for repeated prediction requests"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.lock = threading.RLock()
    
    def _generate_key(self, landmarks: np.ndarray) -> str:
        """Generate cache key from landmarks"""
        # Round landmarks to reduce cache misses from minor variations
        rounded = np.round(landmarks, decimals=3)
        return hashlib.md5(rounded.tobytes()).hexdigest()
    
    def get(self, landmarks: np.ndarray) -> Optional[Dict]:
        """Get cached prediction result"""
        with self.lock:
            key = self._generate_key(landmarks)
            current_time = time.time()
            
            if key in self.cache:
                cached_time, result = self.cache[key]
                
                # Check if cache entry is still valid
                if current_time - cached_time < self.ttl_seconds:
                    self.access_times[key] = current_time
                    result['cached'] = True
                    return result
                else:
                    # Remove expired entry
                    del self.cache[key]
                    del self.access_times[key]
            
            return None
    
    def put(self, landmarks: np.ndarray, result: Dict):
        """Cache prediction result"""
        with self.lock:
            key = self._generate_key(landmarks)
            current_time = time.time()
            
            # Remove oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
            
            # Store result without 'cached' flag
            clean_result = {k: v for k, v in result.items() if k != 'cached'}
            self.cache[key] = (current_time, clean_result)
            self.access_times[key] = current_time
    
    def _evict_oldest(self):
        """Remove oldest cache entry"""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': getattr(self, '_hit_count', 0) / max(getattr(self, '_total_requests', 1), 1),
                'ttl_seconds': self.ttl_seconds
            }

class BatchProcessor:
    """Batch processing for improved GPU utilization"""
    
    def __init__(self, max_batch_size: int = 32, max_wait_time: float = 0.05):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = queue.Queue()
        self.processing = False
        self.lock = threading.Lock()
    
    def add_request(self, landmarks: np.ndarray, callback: callable) -> str:
        """Add request to batch queue"""
        request_id = f"req_{time.time()}_{id(landmarks)}"
        self.pending_requests.put({
            'id': request_id,
            'landmarks': landmarks,
            'callback': callback,
            'timestamp': time.time()
        })
        
        # Start processing if not already running
        with self.lock:
            if not self.processing:
                self.processing = True
                threading.Thread(target=self._process_batch, daemon=True).start()
        
        return request_id
    
    def _process_batch(self):
        """Process batched requests"""
        try:
            batch = []
            start_time = time.time()
            
            # Collect requests for batch
            while len(batch) < self.max_batch_size and (time.time() - start_time) < self.max_wait_time:
                try:
                    request = self.pending_requests.get(timeout=0.01)
                    batch.append(request)
                except queue.Empty:
                    if batch:  # Process partial batch if we have requests
                        break
                    continue
            
            if batch:
                self._execute_batch(batch)
        
        finally:
            with self.lock:
                self.processing = False
    
    def _execute_batch(self, batch: List[Dict]):
        """Execute batch of requests"""
        try:
            # Prepare batch data
            landmarks_batch = np.array([req['landmarks'] for req in batch])
            
            # Process batch (this would be called from the inference engine)
            # For now, we'll call individual callbacks
            for request in batch:
                try:
                    request['callback'](request['landmarks'])
                except Exception as e:
                    logger.error(f"Batch processing error for request {request['id']}: {e}")
        
        except Exception as e:
            logger.error(f"Batch execution error: {e}")

class MemoryProfiler:
    """GPU and system memory profiling"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.memory_history = deque(maxlen=100)
        self.peak_memory = 0
    
    def profile_memory(self) -> Dict[str, Any]:
        """Profile current memory usage"""
        profile = {
            'timestamp': time.time(),
            'system_memory': self._get_system_memory(),
            'gpu_memory': self._get_gpu_memory() if self.gpu_available else None
        }
        
        self.memory_history.append(profile)
        
        # Update peak memory
        if profile['gpu_memory']:
            current_gpu = profile['gpu_memory']['allocated_mb']
            self.peak_memory = max(self.peak_memory, current_gpu)
        
        return profile
    
    def _get_system_memory(self) -> Dict[str, float]:
        """Get system memory usage"""
        memory = psutil.virtual_memory()
        return {
            'total_mb': memory.total / 1024**2,
            'available_mb': memory.available / 1024**2,
            'used_mb': memory.used / 1024**2,
            'percent': memory.percent
        }
    
    def _get_gpu_memory(self) -> Dict[str, float]:
        """Get GPU memory usage"""
        if not self.gpu_available:
            return None
        
        try:
            return {
                'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'cached_mb': torch.cuda.memory_reserved() / 1024**2,
                'total_mb': torch.cuda.get_device_properties(0).total_memory / 1024**2
            }
        except Exception as e:
            logger.warning(f"Could not get GPU memory info: {e}")
            return None
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        if not self.memory_history:
            return {'error': 'No memory data available'}
        
        recent_profiles = list(self.memory_history)
        
        # Calculate averages
        avg_system = np.mean([p['system_memory']['percent'] for p in recent_profiles])
        
        stats = {
            'system_memory_avg_percent': avg_system,
            'peak_gpu_memory_mb': self.peak_memory,
            'memory_samples': len(recent_profiles)
        }
        
        if self.gpu_available and recent_profiles[-1]['gpu_memory']:
            current_gpu = recent_profiles[-1]['gpu_memory']
            stats.update({
                'current_gpu_allocated_mb': current_gpu['allocated_mb'],
                'current_gpu_cached_mb': current_gpu['cached_mb'],
                'gpu_utilization_percent': (current_gpu['allocated_mb'] / current_gpu['total_mb']) * 100
            })
        
        return stats
    
    def cleanup_memory(self):
        """Force memory cleanup"""
        gc.collect()
        if self.gpu_available:
            torch.cuda.empty_cache()
        logger.info("Memory cleanup performed")

class PerformanceMonitor:
    """Comprehensive performance monitoring"""
    
    def __init__(self):
        self.request_times = deque(maxlen=1000)
        self.error_count = 0
        self.total_requests = 0
        self.concurrent_requests = 0
        self.max_concurrent = 0
        self.lock = threading.Lock()
        
        # Performance thresholds
        self.target_response_time = 50  # ms
        self.max_concurrent_threshold = 10
    
    def start_request(self) -> str:
        """Start tracking a request"""
        with self.lock:
            self.concurrent_requests += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent_requests)
            self.total_requests += 1
        
        request_id = f"perf_{time.time()}_{self.total_requests}"
        return request_id
    
    def end_request(self, request_id: str, response_time_ms: float, success: bool = True):
        """End tracking a request"""
        with self.lock:
            self.concurrent_requests = max(0, self.concurrent_requests - 1)
            self.request_times.append(response_time_ms)
            
            if not success:
                self.error_count += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        with self.lock:
            if not self.request_times:
                return {'error': 'No performance data available'}
            
            times = list(self.request_times)
            
            stats = {
                'total_requests': self.total_requests,
                'error_count': self.error_count,
                'error_rate': (self.error_count / max(self.total_requests, 1)) * 100,
                'current_concurrent': self.concurrent_requests,
                'max_concurrent': self.max_concurrent,
                'avg_response_time_ms': np.mean(times),
                'median_response_time_ms': np.median(times),
                'p95_response_time_ms': np.percentile(times, 95),
                'p99_response_time_ms': np.percentile(times, 99),
                'min_response_time_ms': np.min(times),
                'max_response_time_ms': np.max(times),
                'requests_per_second': len(times) / max((times[-1] - times[0]) / 1000, 1) if len(times) > 1 else 0
            }
            
            # Performance health check
            stats['performance_health'] = self._assess_performance_health(stats)
            
            return stats
    
    def _assess_performance_health(self, stats: Dict) -> Dict[str, Any]:
        """Assess overall performance health"""
        health = {
            'status': 'healthy',
            'issues': []
        }
        
        # Check response time
        if stats['avg_response_time_ms'] > self.target_response_time:
            health['issues'].append(f"Average response time ({stats['avg_response_time_ms']:.1f}ms) exceeds target ({self.target_response_time}ms)")
        
        # Check error rate
        if stats['error_rate'] > 5:  # 5% error rate threshold
            health['issues'].append(f"High error rate: {stats['error_rate']:.1f}%")
        
        # Check concurrent load
        if stats['max_concurrent'] > self.max_concurrent_threshold:
            health['issues'].append(f"High concurrent load: {stats['max_concurrent']} (threshold: {self.max_concurrent_threshold})")
        
        if health['issues']:
            health['status'] = 'degraded' if len(health['issues']) < 3 else 'unhealthy'
        
        return health

class PerformanceOptimizer:
    """Main performance optimization coordinator"""
    
    def __init__(self, inference_engine):
        self.inference_engine = inference_engine
        self.cache = RequestCache()
        self.batch_processor = BatchProcessor()
        self.memory_profiler = MemoryProfiler()
        self.performance_monitor = PerformanceMonitor()
        
        # Start background monitoring
        self._start_background_monitoring()
        
        logger.info("Performance optimizer initialized")
    
    def _start_background_monitoring(self):
        """Start background performance monitoring"""
        def monitor_loop():
            while True:
                try:
                    self.memory_profiler.profile_memory()
                    time.sleep(10)  # Profile every 10 seconds
                except Exception as e:
                    logger.error(f"Background monitoring error: {e}")
                    time.sleep(30)  # Wait longer on error
        
        threading.Thread(target=monitor_loop, daemon=True).start()
    
    def optimized_predict(self, landmarks: np.ndarray, use_cache: bool = True) -> Dict[str, Any]:
        """Optimized prediction with caching and monitoring"""
        request_id = self.performance_monitor.start_request()
        start_time = time.time()
        
        try:
            # Check cache first
            if use_cache:
                cached_result = self.cache.get(landmarks)
                if cached_result:
                    response_time = (time.time() - start_time) * 1000
                    self.performance_monitor.end_request(request_id, response_time, True)
                    return cached_result
            
            # Make prediction
            result = self.inference_engine.predict_from_landmarks(landmarks)
            
            # Cache result
            if use_cache and result.get('preprocessing_success', True):
                self.cache.put(landmarks, result)
            
            response_time = (time.time() - start_time) * 1000
            self.performance_monitor.end_request(request_id, response_time, True)
            
            return result
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.performance_monitor.end_request(request_id, response_time, False)
            raise
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        return {
            'cache': self.cache.get_stats(),
            'memory': self.memory_profiler.get_memory_stats(),
            'performance': self.performance_monitor.get_performance_stats(),
            'timestamp': time.time()
        }
    
    def cleanup_resources(self):
        """Cleanup resources and optimize memory"""
        self.memory_profiler.cleanup_memory()
        logger.info("Resources cleaned up")
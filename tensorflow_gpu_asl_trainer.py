#!/usr/bin/env python3
"""
TENSORFLOW GPU ASL TRAINER
Optimized for RTX 5060 with Maximum GPU Utilization

Features:
- TensorFlow GPU acceleration with mixed precision
- MediaPipe hand landmark detection
- Advanced data augmentation and preprocessing
- Real-time GPU monitoring and optimization
- Production-ready model architecture
- Comprehensive logging and error handling

Author: Senior AI/ML Engineer
"""

import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
import json
import time
import threading
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import warnings
import traceback
from datetime import datetime
import gc
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

@dataclass
class TensorFlowGPUConfig:
    """Configuration for TensorFlow GPU training"""
    
    # GPU Configuration
    mixed_precision: bool = True
    batch_size: int = 512
    prefetch_buffer: int = -1  # Will be set to tf.data.AUTOTUNE later
    num_parallel_calls: int = -1  # Will be set to tf.data.AUTOTUNE later
    
    # Data paths
    dataset_path: str = "C:/Users/atulk/Desktop/Innotech/data/ASL_Alphabet_Dataset/asl_alphabet_train"
    output_path: str = "backend/processed_data_tensorflow"
    model_path: str = "backend/models/tensorflow"
    
    # MediaPipe settings
    mp_model_complexity: int = 1
    mp_min_detection_confidence: float = 0.7
    mp_min_tracking_confidence: float = 0.7
    mp_static_image_mode: bool = True
    mp_max_num_hands: int = 1
    
    # Processing settings
    thread_pool_size: int = 16
    batch_process_size: int = 300
    max_retries: int = 2
    
    # Training settings
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    
    # Model architecture
    input_size: int = 68
    hidden_sizes: List[int] = field(default_factory=lambda: [1024, 512, 256, 128])
    num_classes: int = 29
    dropout_rate: float = 0.3
    
    # Data augmentation
    augmentation_enabled: bool = True
    noise_factor: float = 0.02
    rotation_range: float = 0.1
    
    def __post_init__(self):
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        Path(self.model_path).mkdir(parents=True, exist_ok=True)
        # Set TensorFlow constants after import
        self.prefetch_buffer = tf.data.AUTOTUNE
        self.num_parallel_calls = tf.data.AUTOTUNE

class TensorFlowGPUMonitor:
    """GPU monitoring for TensorFlow"""
    
    def __init__(self):
        self._monitoring = False
        self._monitor_thread = None
        
    def start_monitoring(self):
        """Start GPU monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("🎮 TensorFlow GPU monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=3)
        logger.info("GPU monitoring stopped")
    
    def _monitor_loop(self):
        """Monitoring loop"""
        while self._monitoring:
            try:
                self.log_gpu_status()
                time.sleep(10)
            except Exception as e:
                logger.debug(f"GPU monitoring error: {e}")
    
    def log_gpu_status(self):
        """Log GPU status"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
                current_mb = gpu_memory['current'] / 1024 / 1024
                peak_mb = gpu_memory['peak'] / 1024 / 1024
                
                utilization = self.get_gpu_utilization()
                
                logger.info(f"GPU Status - Util: {utilization}% | Memory: {current_mb:.0f}MB (Peak: {peak_mb:.0f}MB)")
        except Exception as e:
            logger.debug(f"GPU status logging failed: {e}")
    
    def get_gpu_utilization(self) -> int:
        """Get GPU utilization"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception:
            pass
        return 0

class MediaPipeProcessor:
    """MediaPipe processor for hand landmark extraction"""
    
    def __init__(self, config: TensorFlowGPUConfig):
        self.config = config
        self.mp_hands = mp.solutions.hands
    
    def process_batch(self, image_batch: List[Tuple[Path, int]]) -> Tuple[List[np.ndarray], List[int], int]:
        """Process batch of images"""
        hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        batch_features = []
        batch_labels = []
        failed_count = 0
        
        try:
            for img_path, label in image_batch:
                try:
                    image = cv2.imread(str(img_path))
                    if image is None:
                        failed_count += 1
                        continue
                    
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb_image)
                    
                    if results.multi_hand_landmarks:
                        landmarks = results.multi_hand_landmarks[0]
                        feature_vector = self._extract_features(landmarks)
                        
                        if feature_vector is not None:
                            batch_features.append(feature_vector)
                            batch_labels.append(label)
                        else:
                            failed_count += 1
                    else:
                        failed_count += 1
                        
                except Exception:
                    failed_count += 1
                    continue
            
            return batch_features, batch_labels, failed_count
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return [], [], len(image_batch)
        
        finally:
            hands.close()
    
    def _extract_features(self, landmarks) -> Optional[np.ndarray]:
        """Extract normalized features from landmarks"""
        try:
            # Extract coordinates
            coords = []
            for landmark in landmarks.landmark:
                coords.extend([landmark.x, landmark.y, landmark.z])
            
            landmarks_np = np.array(coords, dtype=np.float32).reshape(21, 3)
            
            # Normalize relative to wrist
            wrist = landmarks_np[0]
            normalized = landmarks_np - wrist
            
            # Scale normalization
            middle_mcp = normalized[9]  # Middle finger MCP
            hand_size = np.linalg.norm(middle_mcp - normalized[0])
            
            if hand_size < 1e-6:
                return None
            
            normalized = normalized / hand_size
            
            # Distance features
            fingertips = [4, 8, 12, 16, 20]
            distances = []
            
            for tip_idx in fingertips:
                dist = np.linalg.norm(normalized[tip_idx] - normalized[0])
                distances.append(dist)
            
            # Combine features: 63 (21*3) + 5 distances = 68 features
            feature_vector = np.concatenate([
                normalized.flatten(),
                np.array(distances, dtype=np.float32)
            ])
            
            # Validation
            if len(feature_vector) != 68:
                return None
            
            if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
                return None
            
            return feature_vector
            
        except Exception:
            return None

class TensorFlowDataProcessor:
    """Data processor for TensorFlow training"""
    
    def __init__(self, config: TensorFlowGPUConfig, gpu_monitor: TensorFlowGPUMonitor):
        self.config = config
        self.gpu_monitor = gpu_monitor
        self.stats = {'total_images': 0, 'processed_images': 0, 'failed_images': 0}
    
    def discover_dataset(self) -> Tuple[List[Path], List[int], List[str]]:
        """Discover dataset structure"""
        logger.info("🔍 Discovering ASL dataset...")
        
        dataset_path = Path(self.config.dataset_path)
        class_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
        classes = [d.name for d in class_dirs]
        
        if not classes:
            raise ValueError(f"No classes found in {dataset_path}")
        
        all_images = []
        all_labels = []
        
        for i, class_dir in enumerate(class_dirs):
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            class_images = []
            
            for ext in image_extensions:
                class_images.extend(list(class_dir.glob(ext)))
            
            for img_path in class_images:
                if img_path.stat().st_size > 0:
                    all_images.append(img_path)
                    all_labels.append(i)
        
        self.stats['total_images'] = len(all_images)
        
        logger.info(f"Dataset discovered: {len(all_images)} images, {len(classes)} classes")
        return all_images, all_labels, classes
    
    def process_full_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Process complete dataset"""
        logger.info("🚀 Processing dataset with TensorFlow optimization")
        
        start_time = time.time()
        
        # Discover dataset
        all_images, all_labels, classes = self.discover_dataset()
        
        # Create batches
        image_label_pairs = list(zip(all_images, all_labels))
        batches = []
        
        for i in range(0, len(image_label_pairs), self.config.batch_process_size):
            batch = image_label_pairs[i:i + self.config.batch_process_size]
            batches.append(batch)
        
        logger.info(f"Processing {len(batches)} batches with {self.config.thread_pool_size} threads")
        
        # Initialize processor
        processor = MediaPipeProcessor(self.config)
        
        # Process batches
        all_features = []
        all_processed_labels = []
        total_failed = 0
        
        with ThreadPoolExecutor(max_workers=self.config.thread_pool_size) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(processor.process_batch, batch): i 
                for i, batch in enumerate(batches)
            }
            
            # Process with progress tracking
            with tqdm(total=len(batches), desc="🎮 TensorFlow Processing") as pbar:
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    
                    try:
                        batch_features, batch_labels, failed_count = future.result(timeout=120)
                        
                        all_features.extend(batch_features)
                        all_processed_labels.extend(batch_labels)
                        total_failed += failed_count
                        
                        pbar.update(1)
                        
                        # Periodic logging
                        if (batch_idx + 1) % 10 == 0:
                            elapsed = time.time() - start_time
                            processed = len(all_features)
                            speed = processed / elapsed if elapsed > 0 else 0
                            
                            logger.info(f"Batch {batch_idx + 1}/{len(batches)} - "
                                      f"Processed: {processed} - Speed: {speed:.1f} img/sec")
                    
                    except Exception as e:
                        logger.error(f"Batch {batch_idx} failed: {e}")
                        total_failed += len(batches[batch_idx])
        
        processing_time = time.time() - start_time
        
        self.stats.update({
            'processed_images': len(all_features),
            'failed_images': total_failed,
            'processing_time': processing_time
        })
        
        logger.info(f"Processing completed: {len(all_features)} features extracted in {processing_time/60:.1f} min")
        
        if not all_features:
            raise ValueError("No features extracted")
        
        features_array = np.array(all_features, dtype=np.float32)
        labels_array = np.array(all_processed_labels, dtype=np.int32)
        
        return features_array, labels_array, classes

def create_tensorflow_model(config: TensorFlowGPUConfig) -> tf.keras.Model:
    """Create TensorFlow model optimized for GPU"""
    
    # Input layer
    inputs = tf.keras.Input(shape=(config.input_size,), name='hand_landmarks')
    
    # Feature normalization
    x = tf.keras.layers.BatchNormalization(name='input_norm')(inputs)
    
    # Hidden layers with residual connections
    for i, hidden_size in enumerate(config.hidden_sizes):
        # Dense layer
        dense = tf.keras.layers.Dense(
            hidden_size,
            activation=None,
            kernel_regularizer=tf.keras.regularizers.l2(config.weight_decay),
            name=f'dense_{i+1}'
        )(x)
        
        # Batch normalization
        bn = tf.keras.layers.BatchNormalization(name=f'bn_{i+1}')(dense)
        
        # Activation
        activated = tf.keras.layers.Activation('gelu', name=f'gelu_{i+1}')(bn)
        
        # Dropout
        x = tf.keras.layers.Dropout(config.dropout_rate, name=f'dropout_{i+1}')(activated)
        
        # Residual connection (if dimensions match)
        if i > 0 and x.shape[-1] == hidden_size:
            x = tf.keras.layers.Add(name=f'residual_{i+1}')([x, activated])
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        config.num_classes,
        activation='softmax',
        name='predictions'
    )(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='ASL_Classifier')
    
    return model

def create_data_augmentation_layer(config: TensorFlowGPUConfig):
    """Create data augmentation layer"""
    if not config.augmentation_enabled:
        return None
    
    augmentation = tf.keras.Sequential([
        tf.keras.layers.GaussianNoise(config.noise_factor, name='gaussian_noise'),
    ], name='data_augmentation')
    
    return augmentation

class TensorFlowTrainer:
    """TensorFlow trainer with GPU optimization"""
    
    def __init__(self, config: TensorFlowGPUConfig, gpu_monitor: TensorFlowGPUMonitor):
        self.config = config
        self.gpu_monitor = gpu_monitor
        
        # Setup mixed precision
        if config.mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            logger.info("Mixed precision enabled")
        
        # Create model
        self.model = create_tensorflow_model(config)
        
        # Create augmentation
        self.augmentation = create_data_augmentation_layer(config)
        
        # Compile model
        self._compile_model()
        
        # Training state
        self.best_val_acc = 0.0
        self.training_history = []
        
        logger.info(f"Model created with {self.model.count_params():,} parameters")
    
    def _compile_model(self):
        """Compile model with optimized settings"""
        
        # Optimizer with mixed precision support
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        if self.config.mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        # Compile
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        logger.info("Model compiled with AdamW optimizer and mixed precision")
    
    def create_dataset(self, features: np.ndarray, labels: np.ndarray, training: bool = True) -> tf.data.Dataset:
        """Create optimized TensorFlow dataset"""
        
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        
        if training:
            # Shuffle
            dataset = dataset.shuffle(buffer_size=10000, seed=42)
            
            # Apply augmentation
            if self.augmentation:
                dataset = dataset.map(
                    lambda x, y: (self.augmentation(x, training=True), y),
                    num_parallel_calls=self.config.num_parallel_calls
                )
        
        # Batch and prefetch
        dataset = dataset.batch(self.config.batch_size)
        dataset = dataset.prefetch(self.config.prefetch_buffer)
        
        return dataset
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train the model"""
        
        logger.info("🔥 Starting TensorFlow GPU training")
        
        # Create datasets
        train_dataset = self.create_dataset(X_train, y_train, training=True)
        val_dataset = self.create_dataset(X_val, y_val, training=False)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(Path(self.config.model_path) / 'best_model.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                str(Path(self.config.model_path) / 'training_log.csv')
            )
        ]
        
        # Train
        start_time = time.time()
        
        history = self.model.fit(
            train_dataset,
            epochs=self.config.epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # Get best metrics
        best_val_acc = max(history.history['val_accuracy'])
        best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
        
        logger.info(f"Training completed in {training_time/60:.1f} minutes")
        logger.info(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
        
        return {
            'history': history.history,
            'best_val_accuracy': best_val_acc,
            'best_epoch': best_epoch,
            'training_time': training_time
        }
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test set"""
        
        test_dataset = self.create_dataset(X_test, y_test, training=False)
        
        results = self.model.evaluate(test_dataset, verbose=1)
        
        metrics = {
            'test_loss': results[0],
            'test_accuracy': results[1],
            'test_top3_accuracy': results[2]
        }
        
        logger.info(f"Test Results - Accuracy: {metrics['test_accuracy']:.4f}, "
                   f"Top-3 Accuracy: {metrics['test_top3_accuracy']:.4f}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

def setup_tensorflow_gpu():
    """Setup TensorFlow GPU configuration"""
    
    # Check GPU availability
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if not gpus:
        logger.warning("No GPU found! Training will use CPU")
        return False
    
    try:
        # Configure GPU memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Set GPU as visible
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        
        gpu_name = tf.config.experimental.get_device_details(gpus[0])['device_name']
        logger.info(f"GPU configured: {gpu_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"GPU setup failed: {e}")
        return False

def main():
    """Main training function"""
    
    logger.info("🚀 TENSORFLOW GPU ASL TRAINER")
    logger.info("=" * 60)
    
    try:
        # Setup GPU
        gpu_available = setup_tensorflow_gpu()
        if not gpu_available:
            logger.warning("Continuing with CPU training")
        
        # Initialize configuration
        config = TensorFlowGPUConfig()
        
        # Start GPU monitoring
        gpu_monitor = TensorFlowGPUMonitor()
        gpu_monitor.start_monitoring()
        
        try:
            # Process dataset
            logger.info("📊 DATASET PROCESSING")
            processor = TensorFlowDataProcessor(config, gpu_monitor)
            features, labels, classes = processor.process_full_dataset()
            
            logger.info(f"Dataset processed: {features.shape[0]} samples, {len(classes)} classes")
            
            # Create data splits
            logger.info("📊 DATA PREPARATION")
            X_train, X_temp, y_train, y_temp = train_test_split(
                features, labels, test_size=0.3, stratify=labels, random_state=42
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
            )
            
            logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
            # Initialize trainer
            logger.info("🔥 MODEL TRAINING")
            trainer = TensorFlowTrainer(config, gpu_monitor)
            
            # Train model
            training_results = trainer.train(X_train, y_train, X_val, y_val)
            
            # Evaluate on test set
            test_results = trainer.evaluate(X_test, y_test)
            
            # Save model
            model_filepath = str(Path(config.model_path) / 'final_asl_model.keras')
            trainer.save_model(model_filepath)
            
            # Save training report
            training_report = {
                'config': config.__dict__,
                'dataset_stats': processor.stats,
                'training_results': training_results,
                'test_results': test_results,
                'classes': classes,
                'timestamp': datetime.now().isoformat()
            }
            
            report_path = Path(config.model_path) / 'training_report.json'
            with open(report_path, 'w') as f:
                json.dump(training_report, f, indent=2, default=str)
            
            logger.info(f"Training report saved: {report_path}")
            
            print("\n" + "="*60)
            print("🎉 TENSORFLOW GPU TRAINING COMPLETED!")
            print(f"🎯 Best Validation Accuracy: {training_results['best_val_accuracy']:.4f}")
            print(f"🎯 Test Accuracy: {test_results['test_accuracy']:.4f}")
            print(f"📊 Images Processed: {processor.stats['processed_images']:,}")
            print(f"⏱️  Training Time: {training_results['training_time']/60:.1f} minutes")
            print("="*60)
            
            return True
            
        finally:
            gpu_monitor.stop_monitoring()
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
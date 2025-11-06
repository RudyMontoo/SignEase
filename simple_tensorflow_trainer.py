#!/usr/bin/env python3
"""
SIMPLE TENSORFLOW ASL TRAINER
Optimized for RTX 5060 GPU

Features:
- TensorFlow GPU acceleration
- MediaPipe hand landmark detection
- Simple but effective model architecture
- Real-time training progress

Author: Senior AI/ML Engineer
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
import json
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import traceback
from datetime import datetime

# Import TensorFlow components
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    # Import keras components
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    
    print("TensorFlow and Keras imported successfully")
except ImportError as e:
    print(f"TensorFlow import error: {e}")
    exit(1)

# Suppress warnings
warnings.filterwarnings("ignore")

class SimpleConfig:
    """Simple configuration"""
    
    def __init__(self):
        # Data paths
        self.dataset_path = "C:/Users/atulk/Desktop/Innotech/data/ASL_Alphabet_Dataset/asl_alphabet_train"
        self.model_path = "backend/models/simple_tensorflow"
        
        # Training settings
        self.batch_size = 256
        self.epochs = 30
        self.learning_rate = 0.001
        
        # Model settings
        self.input_size = 68
        self.num_classes = 29
        self.dropout_rate = 0.3
        
        # Processing
        self.thread_pool_size = 12
        self.batch_process_size = 200
        
        # Create directories
        Path(self.model_path).mkdir(parents=True, exist_ok=True)

def setup_gpu():
    """Setup GPU for TensorFlow"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"GPU found: {len(gpus)} device(s)")
            return True
        else:
            print("No GPU found, using CPU")
            return False
    except Exception as e:
        print(f"GPU setup error: {e}")
        return False

class MediaPipeProcessor:
    """Simple MediaPipe processor"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
    
    def process_batch(self, image_batch):
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
            
        finally:
            hands.close()
    
    def _extract_features(self, landmarks):
        """Extract features from landmarks"""
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
            middle_mcp = normalized[9]
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
            
            # Combine features
            feature_vector = np.concatenate([
                normalized.flatten(),
                np.array(distances, dtype=np.float32)
            ])
            
            if len(feature_vector) != 68:
                return None
            
            if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
                return None
            
            return feature_vector
            
        except Exception:
            return None

def discover_dataset(dataset_path):
    """Discover dataset structure"""
    print("🔍 Discovering dataset...")
    
    dataset_path = Path(dataset_path)
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
    
    print(f"Dataset discovered: {len(all_images)} images, {len(classes)} classes")
    return all_images, all_labels, classes

def process_dataset(config):
    """Process complete dataset"""
    print("🚀 Processing dataset...")
    
    start_time = time.time()
    
    # Discover dataset
    all_images, all_labels, classes = discover_dataset(config.dataset_path)
    
    # Create batches
    image_label_pairs = list(zip(all_images, all_labels))
    batches = []
    
    for i in range(0, len(image_label_pairs), config.batch_process_size):
        batch = image_label_pairs[i:i + config.batch_process_size]
        batches.append(batch)
    
    print(f"Processing {len(batches)} batches...")
    
    # Initialize processor
    processor = MediaPipeProcessor()
    
    # Process batches
    all_features = []
    all_processed_labels = []
    total_failed = 0
    
    with ThreadPoolExecutor(max_workers=config.thread_pool_size) as executor:
        # Submit all batches
        future_to_batch = {
            executor.submit(processor.process_batch, batch): i 
            for i, batch in enumerate(batches)
        }
        
        # Process with progress tracking
        with tqdm(total=len(batches), desc="Processing") as pbar:
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                
                try:
                    batch_features, batch_labels, failed_count = future.result(timeout=120)
                    
                    all_features.extend(batch_features)
                    all_processed_labels.extend(batch_labels)
                    total_failed += failed_count
                    
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"Batch {batch_idx} failed: {e}")
                    total_failed += len(batches[batch_idx])
    
    processing_time = time.time() - start_time
    
    print(f"Processing completed: {len(all_features)} features in {processing_time/60:.1f} min")
    
    if not all_features:
        raise ValueError("No features extracted")
    
    features_array = np.array(all_features, dtype=np.float32)
    labels_array = np.array(all_processed_labels, dtype=np.int32)
    
    return features_array, labels_array, classes

def create_model(config):
    """Create TensorFlow model"""
    
    model = Sequential([
        Input(shape=(config.input_size,)),
        BatchNormalization(),
        
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(config.dropout_rate),
        
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(config.dropout_rate),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(config.dropout_rate),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(config.dropout_rate),
        
        Dense(config.num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Model created with {model.count_params():,} parameters")
    return model

def main():
    """Main training function"""
    
    print("🚀 SIMPLE TENSORFLOW ASL TRAINER")
    print("=" * 50)
    
    try:
        # Setup GPU
        gpu_available = setup_gpu()
        
        # Initialize configuration
        config = SimpleConfig()
        
        # Process dataset
        print("📊 DATASET PROCESSING")
        features, labels, classes = process_dataset(config)
        
        print(f"Dataset processed: {features.shape[0]} samples, {len(classes)} classes")
        
        # Create data splits
        print("📊 DATA PREPARATION")
        X_train, X_temp, y_train, y_temp = train_test_split(
            features, labels, test_size=0.3, stratify=labels, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )
        
        print(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Create model
        print("🔥 MODEL TRAINING")
        model = create_model(config)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(Path(config.model_path) / 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            batch_size=config.batch_size,
            epochs=config.epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # Evaluate on test set
        print("🎯 FINAL EVALUATION")
        test_results = model.evaluate(X_test, y_test, verbose=1)
        test_loss, test_accuracy = test_results
        
        # Get best metrics
        best_val_acc = max(history.history['val_accuracy'])
        best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
        
        # Save model
        model_filepath = str(Path(config.model_path) / 'final_asl_model.h5')
        model.save(model_filepath)
        
        # Save training report
        training_report = {
            'config': {
                'batch_size': config.batch_size,
                'epochs': config.epochs,
                'learning_rate': config.learning_rate,
                'input_size': config.input_size,
                'num_classes': config.num_classes
            },
            'results': {
                'best_val_accuracy': float(best_val_acc),
                'best_epoch': int(best_epoch),
                'test_accuracy': float(test_accuracy),
                'test_loss': float(test_loss),
                'training_time_minutes': training_time / 60
            },
            'classes': classes,
            'timestamp': datetime.now().isoformat()
        }
        
        report_path = Path(config.model_path) / 'training_report.json'
        with open(report_path, 'w') as f:
            json.dump(training_report, f, indent=2)
        
        print("\n" + "="*50)
        print("🎉 TENSORFLOW TRAINING COMPLETED!")
        print(f"🎯 Best Validation Accuracy: {best_val_acc:.4f}")
        print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
        print(f"📊 Images Processed: {features.shape[0]:,}")
        print(f"⏱️  Training Time: {training_time/60:.1f} minutes")
        print(f"💾 Model saved: {model_filepath}")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"Training failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
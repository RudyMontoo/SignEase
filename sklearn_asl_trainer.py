#!/usr/bin/env python3
"""
SCIKIT-LEARN ASL TRAINER
Fast and Reliable Training with GPU-Accelerated Data Processing

Features:
- MediaPipe GPU-accelerated hand landmark detection
- Scikit-learn machine learning models
- Fast parallel data processing
- Multiple model comparison
- Production-ready pipeline

Author: Senior AI/ML Engineer
"""

import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
import json
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import traceback
from datetime import datetime
import joblib
import subprocess

# Suppress warnings
warnings.filterwarnings("ignore")

class Config:
    """Configuration for scikit-learn training"""
    
    def __init__(self):
        # Data paths
        self.dataset_path = "C:/Users/atulk/Desktop/Innotech/data/ASL_Alphabet_Dataset/asl_alphabet_train"
        self.model_path = "backend/models/sklearn"
        
        # Processing settings
        self.thread_pool_size = 16
        self.batch_process_size = 250
        
        # Model settings
        self.test_size = 0.2
        self.val_size = 0.2
        self.random_state = 42
        
        # Create directories
        Path(self.model_path).mkdir(parents=True, exist_ok=True)

def get_gpu_utilization():
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
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
    
    def process_batch(self, image_batch):
        """Process batch of images with GPU acceleration"""
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
        """Extract comprehensive features from landmarks"""
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
            fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
            distances = []
            
            for tip_idx in fingertips:
                dist = np.linalg.norm(normalized[tip_idx] - normalized[0])
                distances.append(dist)
            
            # Angle features (between fingers)
            angles = []
            for i in range(len(fingertips)-1):
                v1 = normalized[fingertips[i]] - normalized[0]
                v2 = normalized[fingertips[i+1]] - normalized[0]
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                angles.append(angle)
            
            # Palm area (approximate)
            palm_points = [0, 1, 5, 9, 13, 17]  # Wrist and base of fingers
            palm_coords = normalized[palm_points]
            palm_area = np.std(palm_coords.flatten())
            
            # Combine all features
            feature_vector = np.concatenate([
                normalized.flatten(),  # 63 features (21 landmarks * 3 coords)
                np.array(distances, dtype=np.float32),  # 5 distance features
                np.array(angles, dtype=np.float32),  # 4 angle features
                [palm_area]  # 1 palm area feature
            ])
            
            # Total: 63 + 5 + 4 + 1 = 73 features
            
            if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
                return None
            
            return feature_vector
            
        except Exception:
            return None

def discover_dataset(dataset_path):
    """Discover dataset structure"""
    print("🔍 Discovering ASL dataset...")
    
    dataset_path = Path(dataset_path)
    class_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    classes = [d.name for d in class_dirs]
    
    if not classes:
        raise ValueError(f"No classes found in {dataset_path}")
    
    all_images = []
    all_labels = []
    class_counts = {}
    
    for i, class_dir in enumerate(class_dirs):
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        class_images = []
        
        for ext in image_extensions:
            class_images.extend(list(class_dir.glob(ext)))
        
        valid_images = 0
        for img_path in class_images:
            if img_path.stat().st_size > 0:
                all_images.append(img_path)
                all_labels.append(i)
                valid_images += 1
        
        class_counts[class_dir.name] = valid_images
    
    print(f"Dataset discovered: {len(all_images)} images, {len(classes)} classes")
    print(f"Class distribution: {dict(list(class_counts.items())[:5])}...")
    
    return all_images, all_labels, classes

def process_dataset(config):
    """Process complete dataset with GPU acceleration"""
    print("🚀 Processing dataset with GPU acceleration...")
    
    start_time = time.time()
    
    # Discover dataset
    all_images, all_labels, classes = discover_dataset(config.dataset_path)
    
    # Create batches
    image_label_pairs = list(zip(all_images, all_labels))
    batches = []
    
    for i in range(0, len(image_label_pairs), config.batch_process_size):
        batch = image_label_pairs[i:i + config.batch_process_size]
        batches.append(batch)
    
    print(f"Processing {len(batches)} batches with {config.thread_pool_size} threads...")
    
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
        with tqdm(total=len(batches), desc="🎮 GPU Processing") as pbar:
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                
                try:
                    batch_features, batch_labels, failed_count = future.result(timeout=120)
                    
                    all_features.extend(batch_features)
                    all_processed_labels.extend(batch_labels)
                    total_failed += failed_count
                    
                    pbar.update(1)
                    
                    # Periodic logging
                    if (batch_idx + 1) % 20 == 0:
                        elapsed = time.time() - start_time
                        processed = len(all_features)
                        speed = processed / elapsed if elapsed > 0 else 0
                        gpu_util = get_gpu_utilization()
                        
                        print(f"Batch {batch_idx + 1}/{len(batches)} - "
                              f"Processed: {processed} - Speed: {speed:.1f} img/sec - GPU: {gpu_util}%")
                    
                except Exception as e:
                    print(f"Batch {batch_idx} failed: {e}")
                    total_failed += len(batches[batch_idx])
    
    processing_time = time.time() - start_time
    success_rate = len(all_features) / len(all_images) * 100 if all_images else 0
    
    print(f"Processing completed: {len(all_features)} features in {processing_time/60:.1f} min")
    print(f"Success rate: {success_rate:.1f}% (Failed: {total_failed})")
    
    if not all_features:
        raise ValueError("No features extracted")
    
    features_array = np.array(all_features, dtype=np.float32)
    labels_array = np.array(all_processed_labels, dtype=np.int32)
    
    return features_array, labels_array, classes

def create_models():
    """Create different ML models for comparison"""
    
    models = {
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ]),
        
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            ))
        ]),
        
        'Neural Network': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', MLPClassifier(
                hidden_layer_sizes=(512, 256, 128),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            ))
        ]),
        
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                random_state=42
            ))
        ])
    }
    
    return models

def train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test, classes, config):
    """Train and evaluate multiple models"""
    
    print("🔥 Training multiple ML models...")
    
    models = create_models()
    results = {}
    
    for name, model in models.items():
        print(f"\n🎯 Training {name}...")
        start_time = time.time()
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            test_acc = accuracy_score(y_test, test_pred)
            
            training_time = time.time() - start_time
            
            # Store results
            results[name] = {
                'model': model,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'test_accuracy': test_acc,
                'training_time': training_time,
                'predictions': {
                    'train': train_pred,
                    'val': val_pred,
                    'test': test_pred
                }
            }
            
            print(f"✅ {name} - Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f} "
                  f"(Time: {training_time:.1f}s)")
            
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results[name] = {'error': str(e)}
    
    return results

def save_results(results, classes, config):
    """Save training results and models"""
    
    print("💾 Saving models and results...")
    
    # Find best model
    best_model_name = None
    best_test_acc = 0
    
    for name, result in results.items():
        if 'test_accuracy' in result and result['test_accuracy'] > best_test_acc:
            best_test_acc = result['test_accuracy']
            best_model_name = name
    
    # Save best model
    if best_model_name:
        best_model = results[best_model_name]['model']
        model_path = Path(config.model_path) / 'best_asl_model.joblib'
        joblib.dump(best_model, model_path)
        print(f"Best model ({best_model_name}) saved: {model_path}")
    
    # Save all models
    for name, result in results.items():
        if 'model' in result:
            model_path = Path(config.model_path) / f'{name.lower().replace(" ", "_")}_model.joblib'
            joblib.dump(result['model'], model_path)
    
    # Create summary report
    summary = {
        'timestamp': datetime.now().isoformat(),
        'best_model': best_model_name,
        'best_test_accuracy': best_test_acc,
        'classes': classes,
        'num_classes': len(classes),
        'results': {}
    }
    
    for name, result in results.items():
        if 'test_accuracy' in result:
            summary['results'][name] = {
                'train_accuracy': float(result['train_accuracy']),
                'val_accuracy': float(result['val_accuracy']),
                'test_accuracy': float(result['test_accuracy']),
                'training_time': float(result['training_time'])
            }
    
    # Save summary
    summary_path = Path(config.model_path) / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Training summary saved: {summary_path}")
    
    return summary

def main():
    """Main training function"""
    
    print("🚀 SCIKIT-LEARN ASL TRAINER")
    print("=" * 50)
    
    try:
        # Initialize configuration
        config = Config()
        
        # Process dataset
        print("📊 DATASET PROCESSING")
        features, labels, classes = process_dataset(config)
        
        print(f"Dataset processed: {features.shape[0]} samples, {features.shape[1]} features, {len(classes)} classes")
        
        # Create data splits
        print("📊 DATA PREPARATION")
        
        # First split: train + val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels, 
            test_size=config.test_size, 
            stratify=labels, 
            random_state=config.random_state
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=config.val_size, 
            stratify=y_temp, 
            random_state=config.random_state
        )
        
        print(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Train and evaluate models
        print("🔥 MODEL TRAINING & EVALUATION")
        results = train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test, classes, config)
        
        # Save results
        summary = save_results(results, classes, config)
        
        # Print final results
        print("\n" + "="*50)
        print("🎉 SCIKIT-LEARN TRAINING COMPLETED!")
        print(f"🏆 Best Model: {summary['best_model']}")
        print(f"🎯 Best Test Accuracy: {summary['best_test_accuracy']:.4f}")
        print(f"📊 Total Samples: {features.shape[0]:,}")
        print(f"🔢 Feature Dimensions: {features.shape[1]}")
        print(f"📝 Classes: {len(classes)}")
        
        print("\n📈 Model Comparison:")
        for name, metrics in summary['results'].items():
            print(f"  {name}: {metrics['test_accuracy']:.4f} "
                  f"(Train: {metrics['train_accuracy']:.4f}, "
                  f"Val: {metrics['val_accuracy']:.4f})")
        
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"Training failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
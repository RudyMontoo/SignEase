#!/usr/bin/env python3
"""
IMPROVED SCIKIT-LEARN ASL TRAINER
Enhanced MediaPipe Processing for Higher Success Rate

Features:
- Adaptive detection thresholds
- Image preprocessing pipeline
- Multi-attempt processing strategies
- Detailed failure analysis
- Quality-based filtering

Author: Senior AI/ML Engineer
"""

import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
import json
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import traceback
from datetime import datetime
import joblib
import subprocess
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings("ignore")

class ImprovedConfig:
    """Enhanced configuration for better success rate"""
    
    def __init__(self):
        # Data paths
        self.dataset_path = "C:/Users/atulk/Desktop/Innotech/data/ASL_Alphabet_Dataset/asl_alphabet_train"
        self.model_path = "backend/models/improved_sklearn"
        
        # Processing settings
        self.thread_pool_size = 12  # Reduced for stability
        self.batch_process_size = 150  # Smaller batches for better error handling
        
        # Detection settings - Progressive thresholds
        self.detection_attempts = [
            {'detection': 0.7, 'tracking': 0.7, 'complexity': 1},
            {'detection': 0.5, 'tracking': 0.5, 'complexity': 1},
            {'detection': 0.3, 'tracking': 0.3, 'complexity': 0},
            {'detection': 0.1, 'tracking': 0.1, 'complexity': 0}
        ]
        
        # Image preprocessing
        self.target_size = (640, 480)
        self.enhance_contrast = True
        self.histogram_equalization = True
        
        # Model settings
        self.test_size = 0.2
        self.val_size = 0.2
        self.random_state = 42
        
        # Create directories
        Path(self.model_path).mkdir(parents=True, exist_ok=True)

class ImagePreprocessor:
    """Advanced image preprocessing for better detection"""
    
    def __init__(self, config):
        self.config = config
    
    def enhance_image(self, image):
        """Apply multiple enhancement techniques"""
        try:
            # Resize to optimal size
            if image.shape[:2] != self.config.target_size[::-1]:
                image = cv2.resize(image, self.config.target_size)
            
            # Convert to RGB for MediaPipe
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume BGR input from cv2.imread
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Enhance contrast
            if self.config.enhance_contrast:
                lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                enhanced = cv2.merge([l, a, b])
                rgb_image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            # Histogram equalization
            if self.config.histogram_equalization:
                # Convert to YUV for better histogram equalization
                yuv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV)
                yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                rgb_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
            
            return rgb_image
            
        except Exception as e:
            # Return original image if enhancement fails
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
    
    def assess_quality(self, image):
        """Assess image quality for filtering"""
        try:
            # Check if image is too small
            if image.shape[0] < 50 or image.shape[1] < 50:
                return False, "too_small"
            
            # Check if image is too dark or too bright
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            mean_brightness = np.mean(gray)
            
            if mean_brightness < 20:
                return False, "too_dark"
            if mean_brightness > 235:
                return False, "too_bright"
            
            # Check for blur (Laplacian variance)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < 50:
                return False, "too_blurry"
            
            return True, "good_quality"
            
        except Exception:
            return False, "quality_check_failed"

class EnhancedMediaPipeProcessor:
    """Enhanced MediaPipe processor with multiple strategies"""
    
    def __init__(self, config):
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.preprocessor = ImagePreprocessor(config)
        self.failure_stats = defaultdict(int)
    
    def process_batch(self, image_batch):
        """Process batch with enhanced strategies"""
        batch_features = []
        batch_labels = []
        failed_count = 0
        failure_reasons = defaultdict(int)
        
        for img_path, label in image_batch:
            try:
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    failed_count += 1
                    failure_reasons["load_failed"] += 1
                    continue
                
                # Assess quality
                quality_ok, quality_reason = self.preprocessor.assess_quality(image)
                if not quality_ok:
                    failed_count += 1
                    failure_reasons[f"quality_{quality_reason}"] += 1
                    continue
                
                # Enhance image
                enhanced_image = self.preprocessor.enhance_image(image)
                
                # Try multiple detection strategies
                feature_vector = self._try_multiple_detections(enhanced_image)
                
                if feature_vector is not None:
                    batch_features.append(feature_vector)
                    batch_labels.append(label)
                else:
                    failed_count += 1
                    failure_reasons["no_hands_detected"] += 1
                    
            except Exception as e:
                failed_count += 1
                failure_reasons["processing_error"] += 1
                continue
        
        # Update global failure stats
        for reason, count in failure_reasons.items():
            self.failure_stats[reason] += count
        
        return batch_features, batch_labels, failed_count, dict(failure_reasons)
    
    def _try_multiple_detections(self, image):
        """Try multiple MediaPipe configurations"""
        
        for attempt_config in self.config.detection_attempts:
            try:
                hands = self.mp_hands.Hands(
                    static_image_mode=True,
                    max_num_hands=1,
                    model_complexity=attempt_config['complexity'],
                    min_detection_confidence=attempt_config['detection'],
                    min_tracking_confidence=attempt_config['tracking']
                )
                
                results = hands.process(image)
                hands.close()
                
                if results.multi_hand_landmarks:
                    landmarks = results.multi_hand_landmarks[0]
                    feature_vector = self._extract_enhanced_features(landmarks)
                    
                    if feature_vector is not None:
                        return feature_vector
                        
            except Exception:
                continue
        
        return None
    
    def _extract_enhanced_features(self, landmarks):
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
            
            # Distance features (fingertips to wrist)
            fingertips = [4, 8, 12, 16, 20]
            distances = []
            for tip_idx in fingertips:
                dist = np.linalg.norm(normalized[tip_idx] - normalized[0])
                distances.append(dist)
            
            # Angle features (between consecutive fingers)
            angles = []
            for i in range(len(fingertips)-1):
                v1 = normalized[fingertips[i]] - normalized[0]
                v2 = normalized[fingertips[i+1]] - normalized[0]
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                angles.append(angle)
            
            # Inter-finger distances
            inter_distances = []
            for i in range(len(fingertips)):
                for j in range(i+1, len(fingertips)):
                    dist = np.linalg.norm(normalized[fingertips[i]] - normalized[fingertips[j]])
                    inter_distances.append(dist)
            
            # Palm features
            palm_points = [0, 1, 5, 9, 13, 17]  # Wrist and finger bases
            palm_coords = normalized[palm_points]
            palm_center = np.mean(palm_coords, axis=0)
            palm_spread = np.std(palm_coords.flatten())
            
            # Finger curl features (distance from tip to base)
            finger_curls = []
            finger_bases = [1, 5, 9, 13, 17]  # Base of each finger
            for i, tip_idx in enumerate(fingertips):
                base_idx = finger_bases[i]
                curl = np.linalg.norm(normalized[tip_idx] - normalized[base_idx])
                finger_curls.append(curl)
            
            # Combine all features
            feature_vector = np.concatenate([
                normalized.flatten(),  # 63 features (21 landmarks * 3 coords)
                np.array(distances, dtype=np.float32),  # 5 distance features
                np.array(angles, dtype=np.float32),  # 4 angle features
                np.array(inter_distances, dtype=np.float32),  # 10 inter-finger distances
                palm_center,  # 3 palm center coordinates
                [palm_spread],  # 1 palm spread feature
                np.array(finger_curls, dtype=np.float32)  # 5 finger curl features
            ])
            
            # Total: 63 + 5 + 4 + 10 + 3 + 1 + 5 = 91 features
            
            if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
                return None
            
            return feature_vector
            
        except Exception:
            return None
    
    def get_failure_summary(self):
        """Get summary of failure reasons"""
        return dict(self.failure_stats)

def discover_dataset(dataset_path):
    """Discover dataset with quality assessment"""
    print("🔍 Discovering ASL dataset with quality assessment...")
    
    dataset_path = Path(dataset_path)
    class_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    classes = [d.name for d in class_dirs]
    
    if not classes:
        raise ValueError(f"No classes found in {dataset_path}")
    
    all_images = []
    all_labels = []
    class_counts = {}
    total_files = 0
    
    for i, class_dir in enumerate(class_dirs):
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        class_images = []
        
        for ext in image_extensions:
            class_images.extend(list(class_dir.glob(ext)))
        
        total_files += len(class_images)
        valid_images = 0
        
        for img_path in class_images:
            if img_path.stat().st_size > 100:  # At least 100 bytes
                all_images.append(img_path)
                all_labels.append(i)
                valid_images += 1
        
        class_counts[class_dir.name] = valid_images
    
    print(f"Dataset discovered: {len(all_images)} valid images out of {total_files} total files")
    print(f"Classes: {len(classes)}")
    print(f"Sample class distribution: {dict(list(class_counts.items())[:5])}...")
    
    return all_images, all_labels, classes

def process_dataset_enhanced(config):
    """Process dataset with enhanced strategies"""
    print("🚀 Processing dataset with enhanced MediaPipe strategies...")
    
    start_time = time.time()
    
    # Discover dataset
    all_images, all_labels, classes = discover_dataset(config.dataset_path)
    
    # Create batches
    image_label_pairs = list(zip(all_images, all_labels))
    batches = []
    
    for i in range(0, len(image_label_pairs), config.batch_process_size):
        batch = image_label_pairs[i:i + config.batch_process_size]
        batches.append(batch)
    
    print(f"Processing {len(batches)} batches with enhanced detection...")
    
    # Initialize processor
    processor = EnhancedMediaPipeProcessor(config)
    
    # Process batches
    all_features = []
    all_processed_labels = []
    total_failed = 0
    global_failure_reasons = defaultdict(int)
    
    with ThreadPoolExecutor(max_workers=config.thread_pool_size) as executor:
        # Submit all batches
        future_to_batch = {
            executor.submit(processor.process_batch, batch): i 
            for i, batch in enumerate(batches)
        }
        
        # Process with progress tracking
        with tqdm(total=len(batches), desc="🎮 Enhanced Processing") as pbar:
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                
                try:
                    batch_features, batch_labels, failed_count, failure_reasons = future.result(timeout=180)
                    
                    all_features.extend(batch_features)
                    all_processed_labels.extend(batch_labels)
                    total_failed += failed_count
                    
                    # Aggregate failure reasons
                    for reason, count in failure_reasons.items():
                        global_failure_reasons[reason] += count
                    
                    pbar.update(1)
                    
                    # Periodic detailed logging
                    if (batch_idx + 1) % 15 == 0:
                        elapsed = time.time() - start_time
                        processed = len(all_features)
                        speed = processed / elapsed if elapsed > 0 else 0
                        success_rate = processed / (processed + total_failed) * 100 if (processed + total_failed) > 0 else 0
                        
                        print(f"Batch {batch_idx + 1}/{len(batches)} - "
                              f"Processed: {processed} - Success Rate: {success_rate:.1f}% - "
                              f"Speed: {speed:.1f} img/sec")
                    
                except Exception as e:
                    print(f"Batch {batch_idx} failed: {e}")
                    total_failed += len(batches[batch_idx])
    
    processing_time = time.time() - start_time
    success_rate = len(all_features) / len(all_images) * 100 if all_images else 0
    
    print(f"\n📊 PROCESSING RESULTS:")
    print(f"Total images: {len(all_images):,}")
    print(f"Successfully processed: {len(all_features):,}")
    print(f"Failed: {total_failed:,}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Processing time: {processing_time/60:.1f} minutes")
    
    # Print failure analysis
    print(f"\n🔍 FAILURE ANALYSIS:")
    total_failures = sum(global_failure_reasons.values())
    for reason, count in sorted(global_failure_reasons.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_failures * 100 if total_failures > 0 else 0
        print(f"  {reason}: {count:,} ({percentage:.1f}%)")
    
    if not all_features:
        raise ValueError("No features extracted")
    
    features_array = np.array(all_features, dtype=np.float32)
    labels_array = np.array(all_processed_labels, dtype=np.int32)
    
    # Save processing report
    processing_report = {
        'total_images': len(all_images),
        'successful_extractions': len(all_features),
        'failed_extractions': total_failed,
        'success_rate': success_rate,
        'processing_time_minutes': processing_time / 60,
        'failure_reasons': dict(global_failure_reasons),
        'feature_dimensions': features_array.shape[1],
        'timestamp': datetime.now().isoformat()
    }
    
    report_path = Path(config.model_path) / 'processing_report.json'
    with open(report_path, 'w') as f:
        json.dump(processing_report, f, indent=2)
    
    print(f"Processing report saved: {report_path}")
    
    return features_array, labels_array, classes, processing_report

def create_optimized_models():
    """Create optimized ML models"""
    
    models = {
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=300,
                max_depth=25,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ))
        ]),
        
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=10,
                subsample=0.8,
                random_state=42
            ))
        ]),
        
        'Neural Network': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', MLPClassifier(
                hidden_layer_sizes=(1024, 512, 256, 128),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42
            ))
        ])
    }
    
    return models

def main():
    """Main training function with enhanced processing"""
    
    print("🚀 IMPROVED SCIKIT-LEARN ASL TRAINER")
    print("=" * 60)
    
    try:
        # Initialize configuration
        config = ImprovedConfig()
        
        print("🔧 Configuration:")
        print(f"  Detection attempts: {len(config.detection_attempts)}")
        print(f"  Thread pool size: {config.thread_pool_size}")
        print(f"  Batch size: {config.batch_process_size}")
        print(f"  Target image size: {config.target_size}")
        
        # Process dataset with enhanced strategies
        print("\n📊 ENHANCED DATASET PROCESSING")
        features, labels, classes, processing_report = process_dataset_enhanced(config)
        
        print(f"\n✅ Dataset processed successfully!")
        print(f"Features shape: {features.shape}")
        print(f"Classes: {len(classes)}")
        
        # Create data splits
        print("\n📊 DATA PREPARATION")
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels, 
            test_size=config.test_size, 
            stratify=labels, 
            random_state=config.random_state
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=config.val_size, 
            stratify=y_temp, 
            random_state=config.random_state
        )
        
        print(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Train models
        print("\n🔥 MODEL TRAINING")
        models = create_optimized_models()
        results = {}
        
        for name, model in models.items():
            print(f"\n🎯 Training {name}...")
            start_time = time.time()
            
            try:
                model.fit(X_train, y_train)
                
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                test_pred = model.predict(X_test)
                
                train_acc = accuracy_score(y_train, train_pred)
                val_acc = accuracy_score(y_val, val_pred)
                test_acc = accuracy_score(y_test, test_pred)
                
                training_time = time.time() - start_time
                
                results[name] = {
                    'model': model,
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'test_accuracy': test_acc,
                    'training_time': training_time
                }
                
                print(f"✅ {name} - Train: {train_acc:.4f}, Val: {val_acc:.4f}, "
                      f"Test: {test_acc:.4f} (Time: {training_time:.1f}s)")
                
            except Exception as e:
                print(f"❌ {name} failed: {e}")
        
        # Find and save best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
        best_model = results[best_model_name]['model']
        best_test_acc = results[best_model_name]['test_accuracy']
        
        # Save best model
        model_path = Path(config.model_path) / 'best_enhanced_model.joblib'
        joblib.dump(best_model, model_path)
        
        # Create comprehensive summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'processing_report': processing_report,
            'best_model': best_model_name,
            'best_test_accuracy': float(best_test_acc),
            'classes': classes,
            'feature_dimensions': int(features.shape[1]),
            'model_results': {
                name: {
                    'train_accuracy': float(result['train_accuracy']),
                    'val_accuracy': float(result['val_accuracy']),
                    'test_accuracy': float(result['test_accuracy']),
                    'training_time': float(result['training_time'])
                }
                for name, result in results.items()
            }
        }
        
        summary_path = Path(config.model_path) / 'enhanced_training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Final results
        print("\n" + "="*60)
        print("🎉 ENHANCED TRAINING COMPLETED!")
        print(f"🏆 Best Model: {best_model_name}")
        print(f"🎯 Best Test Accuracy: {best_test_acc:.4f}")
        print(f"📊 Success Rate: {processing_report['success_rate']:.1f}%")
        print(f"🔢 Feature Dimensions: {features.shape[1]}")
        print(f"💾 Model saved: {model_path}")
        
        print("\n📈 Model Comparison:")
        for name, result in results.items():
            print(f"  {name}: {result['test_accuracy']:.4f}")
        
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"Training failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
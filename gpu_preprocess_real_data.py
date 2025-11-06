#!/usr/bin/env python3
"""
GPU-accelerated preprocessing of REAL ASL dataset for RTX 5060
"""

import torch
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import json
from tqdm import tqdm
import time

def main():
    print("🚀 GPU-Accelerated ASL Data Preprocessing")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # Paths
    dataset_dir = Path("data/ASL_Alphabet_Dataset/asl_alphabet_train")
    output_dir = Path("backend/processed_data")
    output_dir.mkdir(exist_ok=True)
    
    # Check dataset exists
    if not dataset_dir.exists():
        print(f"❌ Dataset not found: {dataset_dir}")
        return False
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    # Discover classes
    classes = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    print(f"📁 Found {len(classes)} classes: {classes}")
    
    # Collect all images
    all_images = []
    all_labels = []
    
    for class_name, class_idx in class_to_idx.items():
        class_dir = dataset_dir / class_name
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG"))
        
        print(f"  {class_name}: {len(images)} images")
        
        for img_path in images:
            all_images.append(img_path)
            all_labels.append(class_idx)
    
    print(f"📊 Total images: {len(all_images)}")
    
    # Process images with GPU acceleration
    features = []
    labels = []
    failed = 0
    
    start_time = time.time()
    
    for i, (img_path, label) in enumerate(tqdm(zip(all_images, all_labels), 
                                              desc="Processing images", 
                                              total=len(all_images))):
        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                failed += 1
                continue
            
            # Convert to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract landmarks
            results = hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                # Get landmarks
                landmarks = results.multi_hand_landmarks[0]
                
                # Extract coordinates
                coords = []
                for landmark in landmarks.landmark:
                    coords.extend([landmark.x, landmark.y, landmark.z])
                
                # Convert to tensor and move to GPU for processing
                landmark_tensor = torch.tensor(coords, dtype=torch.float32, device='cuda')
                
                # Normalize on GPU
                # Wrist normalization
                wrist = landmark_tensor[:3]
                normalized = landmark_tensor.view(-1, 3) - wrist
                
                # Size normalization
                middle_mcp = normalized[9]  # Middle finger MCP
                hand_size = torch.norm(middle_mcp - normalized[0])
                if hand_size > 1e-6:
                    normalized = normalized / hand_size
                
                # Additional features on GPU
                # Distances from wrist to fingertips
                fingertips = [4, 8, 12, 16, 20]
                distances = []
                for tip_idx in fingertips:
                    dist = torch.norm(normalized[tip_idx] - normalized[0])
                    distances.append(dist)
                
                # Combine features
                feature_vector = torch.cat([
                    normalized.flatten(),  # 63 features
                    torch.tensor(distances, device='cuda')  # 5 distance features
                ])
                
                # Move back to CPU for storage
                features.append(feature_vector.cpu().numpy())
                labels.append(label)
            else:
                failed += 1
                
        except Exception as e:
            failed += 1
            continue
    
    processing_time = time.time() - start_time
    
    print(f"✅ Processing complete!")
    print(f"  Processed: {len(features)} images")
    print(f"  Failed: {failed} images")
    print(f"  Success rate: {len(features)/(len(features)+failed)*100:.1f}%")
    print(f"  Processing time: {processing_time:.1f}s")
    print(f"  Speed: {len(all_images)/processing_time:.1f} images/sec")
    
    # Convert to numpy arrays
    features_array = np.array(features, dtype=np.float32)
    labels_array = np.array(labels, dtype=np.int64)
    
    print(f"📊 Final dataset:")
    print(f"  Features shape: {features_array.shape}")
    print(f"  Labels shape: {labels_array.shape}")
    
    # Split data (80% train, 10% val, 10% test)
    n_samples = len(features_array)
    indices = np.random.permutation(n_samples)
    
    train_end = int(0.8 * n_samples)
    val_end = int(0.9 * n_samples)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    # Save splits
    np.save(output_dir / 'train_features.npy', features_array[train_idx])
    np.save(output_dir / 'train_labels.npy', labels_array[train_idx])
    np.save(output_dir / 'val_features.npy', features_array[val_idx])
    np.save(output_dir / 'val_labels.npy', labels_array[val_idx])
    np.save(output_dir / 'test_features.npy', features_array[test_idx])
    np.save(output_dir / 'test_labels.npy', labels_array[test_idx])
    
    # Save metadata
    metadata = {
        'feature_shape': features_array.shape,
        'num_classes': len(classes),
        'classes': classes,
        'class_to_idx': class_to_idx,
        'feature_size': features_array.shape[1],
        'splits': {
            'train': len(train_idx),
            'val': len(val_idx),
            'test': len(test_idx)
        },
        'processing_stats': {
            'total_images': len(all_images),
            'successful': len(features),
            'failed': failed,
            'processing_time': processing_time
        }
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Data saved to {output_dir}")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val: {len(val_idx)} samples") 
    print(f"  Test: {len(test_idx)} samples")
    
    # Cleanup
    hands.close()
    
    print("🎉 GPU preprocessing complete! Ready for GPU training.")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
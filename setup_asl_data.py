#!/usr/bin/env python3
"""
ASL Dataset Setup
================

Simple script to create a mock ASL dataset for testing the training pipeline.
In production, you would replace this with actual ASL dataset loading.
"""

import os
import numpy as np
import cv2
from pathlib import Path

def create_mock_asl_dataset():
    """Create a small mock ASL dataset for testing"""
    print("📁 Creating mock ASL dataset...")
    
    # ASL alphabet classes
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
               'del', 'nothing', 'space']
    
    data_dir = Path("asl_alphabet")
    data_dir.mkdir(exist_ok=True)
    
    # Create directories and sample images
    for class_name in classes:
        class_dir = data_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Create 10 sample images per class
        for i in range(10):
            # Create a simple colored image (different color per class)
            img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            
            # Add some pattern based on class
            color = (hash(class_name) % 255, (hash(class_name) * 2) % 255, (hash(class_name) * 3) % 255)
            cv2.rectangle(img, (50, 50), (150, 150), color, -1)
            
            # Add text
            cv2.putText(img, class_name, (75, 105), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Save image
            img_path = class_dir / f"{class_name}_{i:03d}.jpg"
            cv2.imwrite(str(img_path), img)
        
        print(f"   📂 {class_name}: 10 images created")
    
    print(f"✅ Mock dataset created with {len(classes)} classes, 10 images each")
    print(f"📊 Total: {len(classes) * 10} images")
    print(f"📁 Location: {data_dir.absolute()}")

def main():
    """Main function"""
    print("🎯 ASL Dataset Setup")
    print("=" * 30)
    
    # Check if dataset already exists
    data_dir = Path("asl_alphabet")
    if data_dir.exists() and any(data_dir.iterdir()):
        print("📁 ASL dataset already exists!")
        
        # Count existing samples
        total_samples = 0
        for class_dir in data_dir.iterdir():
            if class_dir.is_dir():
                samples = len(list(class_dir.glob("*.jpg")))
                total_samples += samples
                print(f"   📂 {class_dir.name}: {samples} images")
        
        print(f"📊 Total: {total_samples} images")
        
        response = input("\nReplace with mock data? (y/N): ").lower()
        if response != 'y':
            print("Keeping existing dataset.")
            return
    
    create_mock_asl_dataset()
    print("\n🎉 Dataset setup complete!")
    print("\nNext steps:")
    print("1. Run: python quick_test.py")
    print("2. Run: python asl_training_master.py")

if __name__ == "__main__":
    main()
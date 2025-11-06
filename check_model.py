#!/usr/bin/env python3
"""Check model parameters and info"""

import torch
import sys
from pathlib import Path

def check_model(model_path):
    """Check model parameters and info"""
    try:
        print(f"Checking model: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
            # Count parameters
            total_params = 0
            for key, tensor in state_dict.items():
                params = tensor.numel()
                total_params += params
                print(f"  {key}: {tensor.shape} -> {params:,} parameters")
            
            print(f"\nTotal parameters: {total_params:,}")
            
            # Check other info
            if 'training_info' in checkpoint:
                print(f"Training info: {checkpoint['training_info']}")
            
            if 'model_info' in checkpoint:
                print(f"Model info: {checkpoint['model_info']}")
                
            if 'val_metrics' in checkpoint:
                print(f"Validation metrics: {checkpoint['val_metrics']}")
                
        return total_params
        
    except Exception as e:
        print(f"Error checking {model_path}: {e}")
        return 0

if __name__ == "__main__":
    models_to_check = [
        'backend/models/rtx5060_full/rtx5060_best_model.pth',
        'backend/models/asl_model_best.pth',
        'backend/models/improved_asl_model_best.pth',
        'models/asl_model_best_20251104_210836.pth'
    ]
    
    print("=== Model Parameter Analysis ===\n")
    
    for model_path in models_to_check:
        if Path(model_path).exists():
            params = check_model(model_path)
            print(f"{'='*60}\n")
        else:
            print(f"Model not found: {model_path}\n")
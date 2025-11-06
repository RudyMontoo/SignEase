#!/usr/bin/env python3
"""
ASL Classification Training Script
Complete training pipeline for ASL hand gesture recognition
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time

# Import the workaround configuration
from rtx5060_pytorch_workaround import configure_rtx5060_workaround, UniversalASLClassifier

def create_mock_asl_dataset(num_samples=2000):
    """Create mock ASL dataset for training"""
    
    print("Creating mock ASL dataset...")
    
    # Generate synthetic image data
    images = torch.randn(num_samples, 3, 64, 64)
    
    # Generate labels (A-Z = 0-25)
    labels = torch.randint(0, 26, (num_samples,))
    
    # Add some pattern to make it learnable
    for i in range(26):
        mask = labels == i
        if mask.sum() > 0:
            # Add class-specific patterns
            images[mask] += torch.randn(1, 3, 64, 64) * 0.5
    
    return images, labels

def train_asl_model():
    """Train ASL classification model"""
    
    print("=== ASL CLASSIFICATION TRAINING ===\n")
    
    # Configure device
    device = configure_rtx5060_workaround()
    
    # Create dataset
    images, labels = create_mock_asl_dataset(2000)
    
    # Split into train/validation
    train_size = int(0.8 * len(images))
    train_images, val_images = images[:train_size], images[train_size:]
    train_labels, val_labels = labels[:train_size], labels[train_size:]
    
    # Create data loaders
    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = UniversalASLClassifier(num_classes=26).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    num_epochs = 20
    best_val_acc = 0.0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1:2d}/{num_epochs}: "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_asl_model.pth')
        
        scheduler.step()
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    return model

if __name__ == "__main__":
    model = train_asl_model()

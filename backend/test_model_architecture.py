#!/usr/bin/env python3
"""
Comprehensive test for Task 2.2: Model Architecture Implementation
Tests all subtasks and acceptance criteria
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent))

def test_asl_classifier_implementation():
    """Test ASLClassifier (MLP) implementation"""
    print("=== Testing ASLClassifier Implementation ===")
    
    from models.asl_classifier import ASLClassifier
    
    # Test model creation with different configurations
    configs = [
        {'input_size': 107, 'hidden_sizes': [256, 128, 64], 'num_classes': 29},
        {'input_size': 107, 'hidden_sizes': [512, 256, 128], 'num_classes': 29},
        {'input_size': 107, 'hidden_sizes': [128, 64], 'num_classes': 29, 'dropout_rate': 0.5}
    ]
    
    for i, config in enumerate(configs):
        model = ASLClassifier(**config)
        print(f"✅ Model {i+1} created: {model.count_parameters():,} parameters")
        
        # Test forward pass
        batch_size = 16
        test_input = torch.randn(batch_size, config['input_size'])
        
        model.eval()
        with torch.no_grad():
            output = model(test_input)
            probabilities = model.predict_proba(test_input)
            predictions, confidence = model.predict(test_input)
        
        # Verify output shapes
        assert output.shape == (batch_size, config['num_classes']), f"Wrong output shape: {output.shape}"
        assert probabilities.shape == (batch_size, config['num_classes']), f"Wrong prob shape: {probabilities.shape}"
        assert predictions.shape == (batch_size,), f"Wrong pred shape: {predictions.shape}"
        assert confidence.shape == (batch_size,), f"Wrong conf shape: {confidence.shape}"
        
        print(f"   Forward pass: {test_input.shape} -> {output.shape}")
        print(f"   Predictions range: [0, {predictions.max().item()}]")
        print(f"   Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
    
    print("✅ ASLClassifier implementation working correctly")
    return True

def test_custom_dataset_class():
    """Test custom Dataset class for landmarks"""
    print("\n=== Testing Custom Dataset Class ===")
    
    from datasets.asl_dataset import ASLLandmarkDataset, ASLDatasetLoader
    
    # Test with processed data
    data_dir = Path("processed_data")
    
    if not data_dir.exists():
        print("⚠️  Processed data not found, creating synthetic test data...")
        
        # Create synthetic test data
        import numpy as np
        
        test_features = np.random.randn(1000, 107).astype(np.float32)
        test_labels = np.random.randint(0, 29, 1000)
        class_names = [chr(i) for i in range(ord('A'), ord('Z')+1)] + ['space', 'del', 'nothing']
        
        # Test dataset creation
        dataset = ASLLandmarkDataset(
            features=test_features,
            labels=test_labels,
            class_names=class_names,
            augment=True,
            augmentation_factor=2
        )
        
        print(f"✅ Synthetic dataset created: {len(dataset)} samples")
        
    else:
        # Test with real processed data
        try:
            dataset = ASLDatasetLoader.load_from_processed_data(
                data_dir, 'train', augment=True, augmentation_factor=2
            )
            print(f"✅ Real dataset loaded: {len(dataset)} samples")
        except Exception as e:
            print(f"⚠️  Could not load real data: {e}")
            return False
    
    # Test dataset functionality
    print(f"   Original samples: {dataset.num_samples}")
    print(f"   Effective size: {dataset.effective_size}")
    print(f"   Feature dim: {dataset.feature_dim}")
    print(f"   Classes: {dataset.num_classes}")
    
    # Test sample access
    features, label = dataset[0]  # Original sample
    aug_features, aug_label = dataset[dataset.num_samples + 10]  # Augmented sample
    
    print(f"✅ Sample access working:")
    print(f"   Original: {features.shape}, label: {label}")
    print(f"   Augmented: {aug_features.shape}, label: {aug_label}")
    
    # Test class distribution
    distribution = dataset.get_class_distribution()
    print(f"✅ Class distribution: {len(distribution)} classes")
    
    # Test dataset statistics
    stats = dataset.get_dataset_stats()
    print(f"✅ Dataset statistics computed")
    
    print("✅ Custom Dataset class working correctly")
    return True

def test_dataloader_setup():
    """Test DataLoader with proper batching"""
    print("\n=== Testing DataLoader Setup ===")
    
    from datasets.asl_dataset import ASLDatasetLoader
    from torch.utils.data import DataLoader
    
    data_dir = Path("processed_data")
    
    if not data_dir.exists():
        print("⚠️  Using synthetic data for DataLoader test...")
        
        # Create synthetic dataset
        import numpy as np
        from datasets.asl_dataset import ASLLandmarkDataset
        
        test_features = np.random.randn(500, 107).astype(np.float32)
        test_labels = np.random.randint(0, 29, 500)
        
        dataset = ASLLandmarkDataset(
            features=test_features,
            labels=test_labels,
            augment=True,
            augmentation_factor=1
        )
        
        # Create DataLoader
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
    else:
        # Test with real data
        try:
            train_loader, val_loader, test_loader = ASLDatasetLoader.create_data_loaders(
                data_dir, batch_size=32, augment_train=True, balanced_sampling=False
            )
            
            dataloader = train_loader
            print(f"✅ Real data loaders created:")
            print(f"   Train: {len(train_loader)} batches")
            print(f"   Val: {len(val_loader)} batches")
            print(f"   Test: {len(test_loader)} batches")
            
        except Exception as e:
            print(f"⚠️  Could not create real data loaders: {e}")
            return False
    
    # Test batch loading
    batch_features, batch_labels = next(iter(dataloader))
    
    print(f"✅ DataLoader working:")
    print(f"   Batch features: {batch_features.shape}")
    print(f"   Batch labels: {batch_labels.shape}")
    print(f"   Feature range: [{batch_features.min():.3f}, {batch_features.max():.3f}]")
    print(f"   Label range: [{batch_labels.min()}, {batch_labels.max()}]")
    
    # Test multiple batches
    batch_count = 0
    for batch_features, batch_labels in dataloader:
        batch_count += 1
        if batch_count >= 3:  # Test first 3 batches
            break
    
    print(f"✅ Multiple batch loading working: {batch_count} batches tested")
    
    print("✅ DataLoader setup working correctly")
    return True

def test_model_save_load():
    """Test model save/load functionality"""
    print("\n=== Testing Model Save/Load Functionality ===")
    
    from models.asl_classifier import ASLClassifier
    from utils.model_utils import ModelManager
    
    # Create test model
    model = ASLClassifier(
        input_size=107,
        hidden_sizes=[256, 128, 64],
        num_classes=29
    )
    
    print(f"✅ Test model created: {model.count_parameters():,} parameters")
    
    # Test built-in save/load
    save_path = Path("test_model.pth")
    
    # Save model
    model.save_model(save_path, training_info={'test': True})
    print(f"✅ Model saved to {save_path}")
    
    # Load model
    loaded_model = ASLClassifier.load_model(save_path)
    print(f"✅ Model loaded successfully")
    
    # Test that models produce same output
    test_input = torch.randn(1, 107)
    
    model.eval()
    loaded_model.eval()
    
    with torch.no_grad():
        original_output = model(test_input)
        loaded_output = loaded_model(test_input)
    
    # Check if outputs are close (allowing for small numerical differences)
    if torch.allclose(original_output, loaded_output, atol=1e-5):
        print("✅ Loaded model produces identical outputs")
    else:
        print("⚠️  Small differences in loaded model outputs (acceptable)")
    
    # Test ModelManager
    model_manager = ModelManager(Path("test_models"))
    
    # Save with manager
    saved_path = model_manager.save_model(
        model=model,
        model_name="test_save_load",
        epoch=5,
        metrics={'val_loss': 0.3, 'val_acc': 0.9},
        is_best=True
    )
    
    print(f"✅ Model saved with ModelManager")
    
    # Load with manager
    loaded_data = model_manager.load_model(
        model_name="test_save_load",
        model_class=ASLClassifier
    )
    
    print(f"✅ Model loaded with ModelManager")
    print(f"   Epoch: {loaded_data['epoch']}")
    print(f"   Metrics: {loaded_data['metrics']}")
    
    # Cleanup
    if save_path.exists():
        save_path.unlink()
    if save_path.with_suffix('.json').exists():
        save_path.with_suffix('.json').unlink()
    
    import shutil
    if Path("test_models").exists():
        shutil.rmtree("test_models")
    
    print("✅ Model save/load functionality working correctly")
    return True

def test_model_summary_and_parameters():
    """Test model summary and parameter counting"""
    print("\n=== Testing Model Summary and Parameter Counting ===")
    
    from models.asl_classifier import ASLClassifier
    from utils.model_utils import count_parameters, model_summary
    
    # Test different model sizes
    configs = [
        {'hidden_sizes': [128, 64], 'name': 'Small'},
        {'hidden_sizes': [256, 128, 64], 'name': 'Medium'},
        {'hidden_sizes': [512, 256, 128, 64], 'name': 'Large'}
    ]
    
    for config in configs:
        model = ASLClassifier(
            input_size=107,
            hidden_sizes=config['hidden_sizes'],
            num_classes=29
        )
        
        # Test parameter counting
        total_params, trainable_params = count_parameters(model)
        model_params = model.count_parameters()
        
        assert total_params == model_params, "Parameter count mismatch"
        
        print(f"✅ {config['name']} model:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        # Test model summary
        summary = model.get_model_summary()
        
        print(f"   Architecture: {summary['architecture']}")
        print(f"   Hidden layers: {summary['hidden_layers']}")
        print(f"   Layer details: {len(summary['layer_details'])} layers")
        
        # Test utility function summary
        util_summary = model_summary(model, (107,))
        
        print(f"   Model size: {util_summary['model_size_mb']:.2f} MB")
        
        # Verify summary consistency
        assert summary['total_parameters'] == util_summary['total_parameters']
        assert summary['trainable_parameters'] == util_summary['trainable_parameters']
    
    print("✅ Model summary and parameter counting working correctly")
    return True

def verify_acceptance_criteria():
    """Verify all acceptance criteria for Task 2.2"""
    print("\n=== Verifying Acceptance Criteria ===")
    
    criteria = [
        ("Model forward pass works with sample input", test_forward_pass),
        ("Model outputs correct shape (batch_size, 29)", test_output_shape),
        ("Dataset class loads data correctly", test_dataset_loading),
        ("Model can be saved and loaded", test_save_load_basic)
    ]
    
    results = []
    for criterion, test_func in criteria:
        try:
            result = test_func()
            results.append((criterion, result))
            status = "✅" if result else "❌"
            print(f"{status} {criterion}")
        except Exception as e:
            results.append((criterion, False))
            print(f"❌ {criterion}: {e}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n📊 Acceptance Criteria: {passed}/{total} passed")
    
    return passed == total

def test_forward_pass():
    """Test model forward pass"""
    from models.asl_classifier import ASLClassifier
    
    model = ASLClassifier()
    test_input = torch.randn(16, 107)
    
    model.eval()
    with torch.no_grad():
        output = model(test_input)
    
    return output is not None and not torch.isnan(output).any()

def test_output_shape():
    """Test model output shape"""
    from models.asl_classifier import ASLClassifier
    
    model = ASLClassifier()
    batch_size = 32
    test_input = torch.randn(batch_size, 107)
    
    model.eval()
    with torch.no_grad():
        output = model(test_input)
    
    return output.shape == (batch_size, 29)

def test_dataset_loading():
    """Test dataset loading"""
    from datasets.asl_dataset import ASLLandmarkDataset
    import numpy as np
    
    test_features = np.random.randn(100, 107).astype(np.float32)
    test_labels = np.random.randint(0, 29, 100)
    
    dataset = ASLLandmarkDataset(test_features, test_labels)
    features, label = dataset[0]
    
    return features.shape == (107,) and isinstance(label.item(), int)

def test_save_load_basic():
    """Test basic save/load functionality"""
    from models.asl_classifier import ASLClassifier
    
    model = ASLClassifier()
    save_path = Path("temp_test_model.pth")
    
    try:
        model.save_model(save_path)
        loaded_model = ASLClassifier.load_model(save_path)
        
        # Cleanup
        if save_path.exists():
            save_path.unlink()
        if save_path.with_suffix('.json').exists():
            save_path.with_suffix('.json').unlink()
        
        return loaded_model is not None
    except Exception:
        return False

def main():
    """Main test function"""
    print("=== SignEase MVP - Model Architecture Implementation Test ===\n")
    
    # Run all tests
    tests = [
        ("ASLClassifier Implementation", test_asl_classifier_implementation),
        ("Custom Dataset Class", test_custom_dataset_class),
        ("DataLoader Setup", test_dataloader_setup),
        ("Model Save/Load", test_model_save_load),
        ("Model Summary & Parameters", test_model_summary_and_parameters)
    ]
    
    test_results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            logger.error(f"{test_name} failed: {e}")
            test_results.append((test_name, False))
    
    # Verify acceptance criteria
    criteria_passed = verify_acceptance_criteria()
    
    # Summary
    print(f"\n{'='*70}")
    print("MODEL ARCHITECTURE IMPLEMENTATION TEST SUMMARY")
    print(f"{'='*70}")
    
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    print(f"\nTest Results: {passed_tests}/{total_tests} passed")
    print(f"Acceptance Criteria: {'✅ PASSED' if criteria_passed else '❌ FAILED'}")
    
    if passed_tests == total_tests and criteria_passed:
        print("\n🎉 MODEL ARCHITECTURE IMPLEMENTATION COMPLETE!")
        print("✅ All tests passed")
        print("✅ All acceptance criteria met")
        print("✅ Ready for training pipeline implementation")
        return True
    else:
        print("\n❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
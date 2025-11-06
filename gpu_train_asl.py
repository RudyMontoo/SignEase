import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
import cv2
import numpy as np
from pathlib import Path
import time

print("🚀 GPU ASL Training Started!")

class FastASLDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = Path(data_path)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        classes = sorted([d.name for d in self.data_path.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        for class_dir in self.data_path.iterdir():
            if class_dir.is_dir():
                class_idx = self.class_to_idx[class_dir.name]
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((str(img_path), class_idx))
        
        print(f"Loaded {len(self.samples)} samples, {len(classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

class ASLModel(nn.Module):
    def __init__(self, num_classes=29):
        super().__init__()
        self.backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.backbone.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def main():
    print("=" * 60)
    
    # GPU check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if device.type != 'cuda':
        print("❌ No GPU available")
        return
    
    print(f"✅ GPU: {torch.cuda.get_device_name()}")
    
    # Data
    data_path = "data/ASL_Alphabet_Dataset/asl_alphabet_train"
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Loading dataset...")
    dataset = FastASLDataset(data_path, transform)
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Model
    model = ASLModel(num_classes=29).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    epochs = 10
    best_acc = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 500 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        train_acc = 100. * train_correct / train_total
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'asl_model_{val_acc:.2f}.pth')
            print(f"💾 Best model saved: {val_acc:.2f}%")
    
    print(f"\n🎉 Training completed! Best accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
 
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from sklearn.metrics import classification_report, accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to dataset
dataset_dir = r'C:\Users\panka\OneDrive\Desktop\testing\Dataset'

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = datasets.ImageFolder(dataset_dir, transform=transform)
num_classes = len(dataset.classes)

# Split dataset into training and validation
val_ratio = 0.2
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load pre-trained ViT model and modify head
weights = ViT_B_16_Weights.DEFAULT
model = vit_b_16(weights=weights)
model.heads = nn.Sequential(nn.Linear(model.heads[0].in_features, num_classes))
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Learning rate scheduler


# Evaluation function
def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return acc, all_labels, all_preds

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    # Validation
    val_acc, y_true, y_pred = evaluate(model, val_loader, device)
    # Log progress
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f} - LR: {current_lr:.6f}")
    # ...no learning rate scheduler...

# Final evaluation and saving
val_acc, y_true, y_pred = evaluate(model, val_loader, device)
print(f"\nFinal Validation Accuracy: {val_acc:.4f}")
print(classification_report(y_true, y_pred, target_names=dataset.classes))

# Save the trained model
model_save_path = "vit_pet_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")


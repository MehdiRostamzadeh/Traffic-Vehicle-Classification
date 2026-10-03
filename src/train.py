from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms , models

# Dataset paths
DATASET_DIR = Path("dataset/final")

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

# Image transformations

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create datasets

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

# DataLoaders

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


print("Classes:", train_dataset.classes)
print("Train images:", len(train_dataset))
print("Test images:", len(test_dataset))

#_______________________________________________________________________
print("-"*70)

# Create ResNet model
model = models.resnet18(weights="DEFAULT")

# Change the last layer for our 8 classes
num_classes = len(train_dataset.classes)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    num_classes
)

print("Number of classes:", num_classes)
print("Model is ready!")

# Loss function
criterion = torch.nn.CrossEntropyLoss()
# پیش بینی تو چقد با جواب واقعی فاصله داره ؟

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001)
# وظیفه داره وزن های مدل را بر اساس خطا به روز رسانی کنه adam

print("Loss function and optimizer are ready!")

# Training settings

epochs = 10


# Training loop

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()

    train_accuracy = correct / total
    train_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {train_loss:.4f} "
        f"Accuracy: {train_accuracy:.4f}"
    )

# Save trained model
# فقط وزن های یاد گرفته شده مدل را ذخیره میکنیم ، نه کل ابجکت مدل state_dict
torch.save(model.state_dict(), "models/resnet18_vehicle.pth")

print("Model saved successfully!")
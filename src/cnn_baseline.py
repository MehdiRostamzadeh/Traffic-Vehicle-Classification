from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Dataset paths
DATASET_DIR = Path("dataset/final")

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "test"


# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# چرا فعلاً Augmentation نداریم؟
# چون این مدل Baseline است.


# Create datasets
train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=transform
)


# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

print("Classes:", train_dataset.classes)
print("Train images:", len(train_dataset))
print("Validation images:", len(val_dataset))
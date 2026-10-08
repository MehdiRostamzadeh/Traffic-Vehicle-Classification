from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ==========================================
# Configuration
# ==========================================

DATASET_DIR = Path("dataset/final")
TRAIN_DIR = DATASET_DIR / "train"

BATCH_SIZE = 32


# ==========================================
# Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ==========================================
# Transform
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ==========================================
# Dataset
# ==========================================

dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

num_classes = len(dataset.classes)


# ==========================================
# Simple model
# ==========================================

class SimpleCNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                128 * 28 * 28,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        return self.classifier(x)


# ==========================================
# Create model
# ==========================================

model = SimpleCNN(num_classes).to(device)


# ==========================================
# Loss functions
# ==========================================

cross_entropy = nn.CrossEntropyLoss()

bce_loss = nn.BCEWithLogitsLoss()


# ==========================================
# Show difference
# ==========================================

images, labels = next(iter(loader))

images = images.to(device)
labels = labels.to(device)

outputs = model(images)


# CrossEntropy
ce_value = cross_entropy(
    outputs,
    labels
)


# One-hot labels for BCE
one_hot_labels = torch.zeros(
    labels.size(0),
    num_classes,
    device=device
)

one_hot_labels.scatter_(
    1,
    labels.unsqueeze(1),
    1
)


# BCE
bce_value = bce_loss(
    outputs,
    one_hot_labels
)


print("\nLoss comparison")
print("-------------------------")
print("CrossEntropyLoss:", ce_value.item())
print("BCEWithLogitsLoss:", bce_value.item())

print("\nLabel shape:", labels.shape)
print("One-hot shape:", one_hot_labels.shape)
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms


# ==========================================
# Configuration
# ==========================================

DATASET_DIR = Path("dataset/final")
TRAIN_DIR = DATASET_DIR / "train"

BATCH_SIZE = 32
SEED = 42


# ==========================================
# Reproducibility
# ==========================================

random.seed(SEED)
torch.manual_seed(SEED)


# ==========================================
# Transform
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ==========================================
# Load original dataset
# ==========================================

dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=transform
)

print("Classes:")
print(dataset.classes)

print("\nOriginal dataset size:")
print(len(dataset))


# ==========================================
# Get indices for each class
# ==========================================

class_indices = {}

for class_id, class_name in enumerate(dataset.classes):

    indices = [
        i
        for i, label in enumerate(dataset.targets)
        if label == class_id
    ]

    class_indices[class_name] = indices

    print(f"{class_name}: {len(indices)} images")


# ==========================================
# Simulate class imbalance
# ==========================================

samples_per_class = {
    "ambulance": 150,
    "autobus": 150,
    "kamyun": 100,
    "kamyunet": 100,
    "minibus": 75,
    "savari": 75,
    "taxi": 50,
    "vanet": 25,
}


# ==========================================
# Create imbalanced subset
# ==========================================

imbalanced_indices = []

for class_name, num_samples in samples_per_class.items():

    indices = class_indices[class_name]

    selected_indices = random.sample(
        indices,
        num_samples
    )

    imbalanced_indices.extend(selected_indices)


imbalanced_dataset = Subset(
    dataset,
    imbalanced_indices
)

print("\nImbalanced dataset size:")
print(len(imbalanced_dataset))


# ==========================================
# Standard DataLoader
# ==========================================

standard_loader = DataLoader(
    imbalanced_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ==========================================
# Create sample weights
# ==========================================

class_counts = {}

for class_name, count in samples_per_class.items():
    class_counts[class_name] = count


weights = []

for index in imbalanced_indices:

    original_label = dataset.targets[index]

    class_name = dataset.classes[original_label]

    weight = 1.0 / class_counts[class_name]

    weights.append(weight)


weights = torch.DoubleTensor(weights)


# ==========================================
# Balanced Sampler
# ==========================================

balanced_sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(weights),
    replacement=True
)


balanced_loader = DataLoader(
    imbalanced_dataset,
    batch_size=BATCH_SIZE,
    sampler=balanced_sampler
)


# ==========================================
# Compare class distribution
# ==========================================

def get_distribution(loader, name):

    counts = {
        class_name: 0
        for class_name in dataset.classes
    }

    for _, labels in loader:

        for label in labels:

            class_name = dataset.classes[label.item()]

            counts[class_name] += 1

    print(f"\n{name} distribution:")

    for class_name, count in counts.items():

        print(
            f"{class_name}: {count}"
        )


# Standard sampling
get_distribution(
    standard_loader,
    "Standard Sampling"
)


# Balanced sampling
get_distribution(
    balanced_loader,
    "Balanced Sampling"
)
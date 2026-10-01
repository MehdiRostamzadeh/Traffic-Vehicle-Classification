from pathlib import Path

import torch
from torch.utils.data import dataloader
from torchvision import datasets,transforms

from model import create_model



# Dataset paths
DATASET_DIR = Path("dataset/final")

TEST_DIR = DATASET_DIR / "test"

MODEL_PATH = Path("models/resnet18_vehicle.pth")


# Image transformations

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
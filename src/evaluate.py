from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from model import create_model

from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt


# Dataset paths
DATASET_DIR = Path("dataset/final")

TEST_DIR = DATASET_DIR / "test"

MODEL_PATH = Path("models/resnet18_vehicle.pth")


# Image transformations
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Create test dataset
test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)


# Create test DataLoader
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print("Test images:", len(test_dataset))
print("Classes:", test_dataset.classes)
#__________________________________________________________________
print("-" * 50)

#کردن مدل ذخیره شده load
# Create model
num_classes = len(test_dataset.classes)

model = create_model(num_classes)


# Load trained weights

model.load_state_dict(
    torch.load(MODEL_PATH, weights_only=True)
)

print("Trained model loaded successfully!")



# Evaluation
# model.eval()

# correct = 0
# total = 0

# with torch.no_grad():

#     for images, labels in test_loader:

#         outputs = model(images)

#         _, predicted = torch.max(outputs, 1)

#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()


# test_accuracy = correct / total

# print(f"Test Accuracy: {test_accuracy:.4f}")

# new predict label 
model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predicted.numpy())

# Calculate accuracy
correct = sum(
    prediction == label
    for prediction, label in zip(all_predictions, all_labels)
)

total = len(all_labels)

test_accuracy = correct / total

print(f"Test Accuracy: {test_accuracy:.4f}")

#_____________________________________________________________________

# Confusion Matrix
cm = confusion_matrix(
    all_labels,
    all_predictions
)
print("\nConfusion Matrix:")
print(cm)

report = classification_report(all_labels,all_predictions,target_names=test_dataset.classes)
print("\nClassification Report:")
print(report)


# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    range(len(test_dataset.classes)),
    test_dataset.classes,
    rotation=45
)

plt.yticks(
    range(len(test_dataset.classes)),
    test_dataset.classes
)

# Add numbers to the matrix
for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()

plt.savefig(
    "reports/confusion_matrix.png",
    dpi=300
)

plt.show()
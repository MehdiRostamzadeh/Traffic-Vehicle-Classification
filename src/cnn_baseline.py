from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


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


# CNN Baseline Model
class CNNBaseline(torch.nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.features = torch.nn.Sequential(

            # Block 1
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            # Block 2
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            # Block 3
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.classifier = torch.nn.Sequential(

            torch.nn.Flatten(),

            torch.nn.Linear(
                128 * 28 * 28,
                128
            ),

            torch.nn.ReLU(),

            torch.nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x
    

# Create model
num_classes = len(train_dataset.classes)

model = CNNBaseline(num_classes)
print(model)
#______________________________________________

# Loss functio
criterion = torch.nn.CrossEntropyLoss()


# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

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

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {train_loss:.4f} "
        f"Accuracy: {train_accuracy:.4f}"
    )


# =========================
# Validation
# =========================

model.eval()

correct = 0
total = 0
validation_loss = 0.0

with torch.no_grad():

    for images, labels in val_loader:

        outputs = model(images)

        loss = criterion(outputs, labels)

        validation_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()


validation_loss = validation_loss / len(val_loader)
validation_accuracy = correct / total


print("\nValidation Results")
print("-------------------------")
print(f"Validation Loss: {validation_loss:.4f}")
print(f"Validation Accuracy: {validation_accuracy:.4f}")


#________________________________
print("Train classes:")
print(train_dataset.classes)

print("\nValidation classes:")
print(val_dataset.classes)


from collections import Counter

print("\nPredicted classes:")

all_predictions = []

model.eval()

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(predicted.tolist())

prediction_counts = Counter(all_predictions)

for class_index, count in sorted(prediction_counts.items()):
    print(
        train_dataset.classes[class_index],
        ":", count
    )
    


# =========================
# Confusion Matrix
# =========================

# all_labels = []
# all_predictions = []

# model.eval()

# with torch.no_grad():

#     for images, labels in val_loader:

#         outputs = model(images)

#         _, predicted = torch.max(outputs, 1)

#         all_labels.extend(labels.tolist())
#         all_predictions.extend(predicted.tolist())


# cm = confusion_matrix(
#     all_labels,
#     all_predictions
# )

# print("\nConfusion Matrix:")
# print(cm)

# plt.figure(figsize=(8, 6))

# plt.imshow(cm)

# plt.title("CNN Baseline - Confusion Matrix")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")

# plt.xticks(
#     range(len(val_dataset.classes)),
#     val_dataset.classes,
#     rotation=45
# )

# plt.yticks(
#     range(len(val_dataset.classes)),
#     val_dataset.classes
# )

# for i in range(len(cm)):
#     for j in range(len(cm)):
#         plt.text(
#             j,
#             i,
#             cm[i, j],
#             ha="center",
#             va="center"
#         )

# plt.tight_layout()

# plt.savefig(
#     "reports/cnn_baseline_confusion_matrix.png",
#     dpi=300
# )

# plt.show()
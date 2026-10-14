
import time

start_time = time.perf_counter()
#______________________________________________

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from model import create_model


# مسیرهای پروژه
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_ROOT = PROJECT_ROOT / "dataset" / "final" / "TestingData"
MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_vehicle.pth"

BATCH_SIZE = 32

CLASS_NAMES = [
    "ambulance",
    "autobus",
    "kamyun",
    "kamyunet",
    "minibus",
    "savari",
    "taxi",
    "vanet",
]

# تبدیل تصاویر؛ باید با تبدیل زمان ارزیابی مدل قبلی یکسان باشد
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = create_model(num_classes=len(CLASS_NAMES))
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
model.to(device)
model.eval()


for test_folder in sorted(TEST_ROOT.iterdir()):
    if not test_folder.is_dir():
        continue

    dataset = datasets.ImageFolder(
        root=str(test_folder),
        transform=transform,
    )

    # جلوگیری از ارزیابی اشتباه در صورت تفاوت نام یا ترتیب کلاس‌ها
    if dataset.classes != CLASS_NAMES:
        raise ValueError(
            f"Class mismatch in {test_folder.name}\n"
            f"Expected: {CLASS_NAMES}\n"
            f"Found:    {dataset.classes}"
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu()

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())

    accuracy = accuracy_score(y_true, y_pred)

    print("\n" + "=" * 55)
    print(f"Results: {test_folder.name}")
    print("=" * 55)
    print(f"Total images: {len(dataset)}")
    print(f"Correct: {sum(a == b for a, b in zip(y_true, y_pred))}")
    print(f"Incorrect: {sum(a != b for a, b in zip(y_true, y_pred))}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            zero_division=0,
            digits=4,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
    ))

end_time = time.perf_counter()

print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
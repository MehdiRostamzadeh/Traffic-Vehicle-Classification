from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil


# =========================
# Configuration
# =========================

DATASET_DIR = Path("../dataset/final")

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

VAL_SIZE = 0.2
RANDOM_STATE = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================
# Create Validation Split
# =========================

def create_validation_split():

    print("=" * 50)
    print("TRAIN / VALIDATION SPLIT")
    print("=" * 50)

    # Create validation directory
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    total_train = 0
    total_val = 0

    # Get classes
    classes = [folder for folder in TRAIN_DIR.iterdir() if folder.is_dir()]

    for class_dir in classes:

        class_name = class_dir.name

        # Get images
        images = [
            file
            for file in class_dir.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not images:
            continue

        # Split images
        train_images, val_images = train_test_split(
            images,
            test_size=VAL_SIZE,
            random_state=RANDOM_STATE
        )

        # Create validation class directory
        val_class_dir = VAL_DIR / class_name
        val_class_dir.mkdir(parents=True, exist_ok=True)

        # Move validation images
        for image in val_images:
            shutil.move(str(image), str(val_class_dir / image.name))

        total_train += len(train_images)
        total_val += len(val_images)

        print(
            f"{class_name:12} | "
            f"Train: {len(train_images):4} | "
            f"Val: {len(val_images):4}"
        )

    print("-" * 50)
    print(f"Total Train: {total_train}")
    print(f"Total Val  : {total_val}")
    print("=" * 50)


if __name__ == "__main__":
    create_validation_split()
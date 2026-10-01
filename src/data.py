from pathlib import Path


# =========================
# Configuration
# =========================

DATASET_DIR = Path("dataset/final")

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================
# Dataset Summary
# =========================

def count_images(directory):
    """Count images inside each class folder."""

    total = 0
    class_counts = {}

    for class_dir in sorted(directory.iterdir()):

        if not class_dir.is_dir():
            continue

        count = sum(
            1
            for file in class_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        class_counts[class_dir.name] = count
        total += count

    return class_counts, total


# =========================
# Main
# =========================

def main():

    print("=" * 60)
    print("TRAFFIC VEHICLE CLASSIFICATION - DATASET SUMMARY")
    print("=" * 60)

    train_counts, train_total = count_images(TRAIN_DIR)
    test_counts, test_total = count_images(TEST_DIR)

    print("\nTRAIN DATASET")
    print("-" * 60)

    for class_name, count in train_counts.items():
        print(f"{class_name:12} : {count}")

    print(f"\nTotal Train : {train_total}")

    print("\nTEST DATASET")
    print("-" * 60)

    for class_name, count in test_counts.items():
        print(f"{class_name:12} : {count}")

    print(f"\nTotal Test  : {test_total}")

    print("\n" + "=" * 60)
    print("DATASET SUMMARY COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
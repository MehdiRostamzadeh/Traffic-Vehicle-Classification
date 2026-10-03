from pathlib import Path
from PIL import Image


# _______________ Configuration _______________

DATASET_DIR = Path("dataset")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# _______________ Count Images _______________

def count_images(directory):
    return sum(
        1
        for file in directory.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )


# _______________ Analyze Classes _______________

def analyze_classes(directory):

    print(f"\n📁 {directory}")

    total = count_images(directory)
    print(f"Total images: {total}")

    for class_dir in sorted(directory.iterdir()):

        if class_dir.is_dir():

            count = count_images(class_dir)

            print(f"  {class_dir.name}: {count}")


# _______________ Dataset Structure Analysis _______________

splits = [
    "old/train",
    "old/test",
    "old/unclean",
    "new/train",
    "new/unclean",
]

for split in splits:

    path = DATASET_DIR / split

    if path.exists():
        analyze_classes(path)
    else:
        print(f"\n❌ {split}: NOT FOUND")


# _______________ Data Quality Check _______________

def check_image_quality(directory):

    total = 0
    valid = 0
    corrupted = 0
    small_images = 0

    for file in directory.rglob("*"):

        if not file.is_file():
            continue

        if file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        total += 1

        try:
            with Image.open(file) as image:

                image.verify()

            with Image.open(file) as image:

                width, height = image.size

                if width < 100 or height < 100:
                    small_images += 1

            valid += 1

        except Exception:
            corrupted += 1

    return total, valid, corrupted, small_images


# _______________ Run Quality Check _______________

print("\n\n" + "=" * 50)
print("DATA QUALITY CHECK")
print("=" * 50)

for split in splits:

    path = DATASET_DIR / split

    if not path.exists():
        continue

    total, valid, corrupted, small = check_image_quality(path)

    print(
        f"{split:<15} | "
        f"Total: {total:<4} | "
        f"Valid: {valid:<4} | "
        f"Corrupted: {corrupted:<3} | "
        f"Small: {small}"
    )
    
# _______________ Duplicate Detection _______________

import hashlib


def get_hash(file_path):

    hash_md5 = hashlib.md5()

    with open(file_path, "rb") as file:

        for chunk in iter(lambda: file.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def remove_duplicates(train_path, test_path, unclean_path):

    # Hashes from train and test
    reference_hashes = set()

    for folder in [train_path, test_path]:

        if not folder.exists():
            continue

        for file in folder.rglob("*"):

            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                reference_hashes.add(get_hash(file))

    # Find and remove duplicates from unclean
    removed = 0

    if unclean_path.exists():

        for file in unclean_path.rglob("*"):

            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:

                if get_hash(file) in reference_hashes:

                    file.unlink()
                    removed += 1

    return removed


# _______________ Run Duplicate Cleaning _______________

print("\n" + "=" * 50)
print("DUPLICATE CLEANING")
print("=" * 50)


old_removed = remove_duplicates(
    DATASET_DIR / "old/train",
    DATASET_DIR / "old/test",
    DATASET_DIR / "old/unclean"
)


new_removed = remove_duplicates(
    DATASET_DIR / "new/train",
    DATASET_DIR / "new/test",
    DATASET_DIR / "new/unclean"
)


print(f"Old duplicates removed: {old_removed}")
print(f"New duplicates removed: {new_removed}")

# _______________ Merge Final Dataset _______________

import shutil


sources_train = [
    DATASET_DIR / "old/train",
    DATASET_DIR / "new/train",
    DATASET_DIR / "old/unclean",
    DATASET_DIR / "new/unclean",
]

source_test = DATASET_DIR / "old/test"

final_train = DATASET_DIR / "final/train"
final_test = DATASET_DIR / "final/test"


classes = [
    "ambulance",
    "autobus",
    "kamyun",
    "kamyunet",
    "minibus",
    "savari",
    "taxi",
    "vanet",
]


# _______________ Create Final Folders _______________

for class_name in classes:

    (final_train / class_name).mkdir(
        parents=True,
        exist_ok=True
    )

    (final_test / class_name).mkdir(
        parents=True,
        exist_ok=True
    )


# _______________ Merge Train Data _______________

copied_train = 0

for source in sources_train:

    for class_name in classes:

        source_class = source / class_name

        if not source_class.exists():
            continue

        for file in source_class.iterdir():

            if not file.is_file():
                continue

            destination = final_train / class_name / file.name

            # Prevent overwriting
            if destination.exists():

                stem = file.stem
                suffix = file.suffix

                counter = 1

                while destination.exists():

                    new_name = f"{stem}_{counter}{suffix}"
                    destination = final_train / class_name / new_name

                    counter += 1

            shutil.copy2(file, destination)

            copied_train += 1


# _______________ Copy Test Data _______________

copied_test = 0

for class_name in classes:

    source_class = source_test / class_name

    if not source_class.exists():
        continue

    for file in source_class.iterdir():

        if not file.is_file():
            continue

        destination = final_test / class_name / file.name

        shutil.copy2(file, destination)

        copied_test += 1


# _______________ Merge Summary _______________

print("\n" + "=" * 50)
print("FINAL DATASET MERGE")
print("=" * 50)

print(f"Train images copied: {copied_train}")
print(f"Test images copied : {copied_test}")
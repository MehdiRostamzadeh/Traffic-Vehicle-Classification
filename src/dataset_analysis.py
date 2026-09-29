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


def calculate_hash(file_path):

    hash_md5 = hashlib.md5()

    with open(file_path, "rb") as file:

        for chunk in iter(lambda: file.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def find_duplicates(directory):

    hashes = {}
    duplicates = []

    for file in directory.rglob("*"):

        if not file.is_file():
            continue

        if file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        #Hash فقط موقتاً در RAM برنامه ساخته شد و بعد از تمام شدن برنامه از بین رفت.
        file_hash = calculate_hash(file)

        if file_hash in hashes:
            duplicates.append((file, hashes[file_hash]))
        else:
            hashes[file_hash] = file

    return duplicates


# _______________ Run Duplicate Detection _______________

print("\n\n" + "=" * 50)
print("DUPLICATE DETECTION")
print("=" * 50)

for split in splits:

    path = DATASET_DIR / split

    if not path.exists():
        continue

    duplicates = find_duplicates(path)

    print(f"{split:<15} | Duplicates: {len(duplicates)}")

    for duplicate, original in duplicates[:5]:

        print(f"  {duplicate.name} == {original.name}")
        
# _______________ Cross-Split Duplicate Detection _______________

def find_cross_duplicates(directory_1, directory_2):

    hashes_1 = {}

    for file in directory_1.rglob("*"):

        if not file.is_file():
            continue

        if file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        file_hash = calculate_hash(file)
        hashes_1[file_hash] = file

    duplicates = []

    for file in directory_2.rglob("*"):

        if not file.is_file():
            continue

        if file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        file_hash = calculate_hash(file)

        if file_hash in hashes_1:
            duplicates.append(
                (file, hashes_1[file_hash])
            )

    return duplicates


# _______________ Run Cross-Split Duplicate Detection _______________

print("\n\n" + "=" * 50)
print("CROSS-SPLIT DUPLICATE DETECTION")
print("=" * 50)


comparisons = [
    ("new/train", "new/unclean"),
    ("old/train", "old/test"),
    ("old/train", "old/unclean"),
    ("old/test", "old/unclean"),
]


for split_1, split_2 in comparisons:

    path_1 = DATASET_DIR / split_1
    path_2 = DATASET_DIR / split_2

    duplicates = find_cross_duplicates(path_1, path_2)

    print(
        f"{split_1} ↔ {split_2}"
        f" | Duplicates: {len(duplicates)}"
    )

    for duplicate, original in duplicates[:5]:

        print(f"  {duplicate} == {original}")
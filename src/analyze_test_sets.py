from pathlib import Path
from collections import Counter, defaultdict
import hashlib

# مسیر اصلی پروژه
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_ROOT = PROJECT_ROOT / "dataset" / "final" / "TestingData"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def calculate_hash(file_path):
    """محاسبه هش محتوای فایل برای شناسایی تصاویر کاملاً یکسان."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def main():
    if not TEST_ROOT.exists():
        print(f"Test directory not found: {TEST_ROOT}")
        return

    all_hashes = defaultdict(list)
    total_images = 0

    print("=" * 65)
    print("TEST DATASETS QUALITY CHECK")
    print("=" * 65)

    test_folders = sorted(
        folder for folder in TEST_ROOT.iterdir()
        if folder.is_dir()
    )

    for test_folder in test_folders:
        class_counts = Counter()
        folder_images = []

        for class_folder in sorted(test_folder.iterdir()):
            if not class_folder.is_dir():
                continue

            images = [
                path for path in class_folder.rglob("*")
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            ]

            class_counts[class_folder.name] = len(images)
            folder_images.extend(
                (class_folder.name, image_path)
                for image_path in images
            )

        print(f"\nDataset: {test_folder.name}")
        print("-" * 40)

        for class_name, count in sorted(class_counts.items()):
            print(f"{class_name:<15} {count:>6}")

        folder_hashes = Counter()

        for class_name, image_path in folder_images:
            try:
                file_hash = calculate_hash(image_path)
            except OSError as error:
                print(f"Could not read {image_path}: {error}")
                continue

            all_hashes[file_hash].append(
                (test_folder.name, class_name, image_path)
            )
            folder_hashes[file_hash] += 1

        duplicates_inside = sum(
            count - 1
            for count in folder_hashes.values()
            if count > 1
        )

        total_images += len(folder_images)

        print(f"Total images: {len(folder_images)}")
        print(f"Exact duplicate copies within this set: {duplicates_inside}")

    print("\n" + "=" * 65)
    print("DUPLICATES ACROSS TEST FOLDERS")
    print("=" * 65)

    cross_folder_duplicates = 0

    for file_hash, entries in all_hashes.items():
        folder_names = {entry[0] for entry in entries}

        if len(folder_names) > 1:
            cross_folder_duplicates += 1

            print(f"\nIdentical image found in {len(folder_names)} sets:")

            for folder_name, class_name, image_path in entries:
                print(f"  {folder_name} | {class_name} | {image_path.name}")

    print("\n" + "=" * 65)
    print(f"Total images scanned: {total_images}")
    print(f"Unique image contents: {len(all_hashes)}")
    print(f"Exact duplicate groups across sets: {cross_folder_duplicates}")
    print("=" * 65)


if __name__ == "__main__":
    main()

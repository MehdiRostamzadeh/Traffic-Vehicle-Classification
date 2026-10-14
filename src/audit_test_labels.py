
from pathlib import Path
from collections import defaultdict, Counter
import hashlib

ROOT = (
    Path(__file__).resolve().parent.parent
    / "dataset" / "final" / "TestingData"
)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def file_hash(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


# هر هش را به تمام محل‌های وجود تصویر مرتبط می‌کنیم
images = defaultdict(list)

for test_folder in sorted(ROOT.iterdir()):
    if not test_folder.is_dir():
        continue

    for class_folder in sorted(test_folder.iterdir()):
        if not class_folder.is_dir():
            continue

        for path in class_folder.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue

            try:
                digest = file_hash(path)
            except OSError:
                print(f"Cannot read: {path}")
                continue

            images[digest].append({
                "test": test_folder.name,
                "label": class_folder.name,
                "file": path.name,
            })


conflicts = []

for digest, entries in images.items():
    labels = {entry["label"] for entry in entries}

    if len(labels) > 1:
        conflicts.append(entries)

print("=" * 60)
print("TEST LABEL AUDIT")
print("=" * 60)
print(f"Unique image contents: {len(images)}")
print(f"Images with conflicting labels: {len(conflicts)}")

# فقط 30 مورد اول را نمایش می‌دهیم تا خروجی بیش از حد طولانی نشود
for i, entries in enumerate(conflicts[:30], start=1):
    print(f"\nConflict {i}: {entries[0]['file']}")

    for entry in entries:
        print(f"  {entry['test']:<10} -> {entry['label']}")

if len(conflicts) > 30:
    print(f"\n... {len(conflicts) - 30} more conflicts not displayed.")

print("\nLabel frequency across all image occurrences:")
label_counts = Counter(
    entry["label"]
    for entries in images.values()
    for entry in entries
)

for label, count in sorted(label_counts.items()):
    print(f"{label:<15} {count:>6}")

from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "detector_dataset"

SOURCE_IMAGES = DATASET / "source_images"
SOURCE_LABELS = DATASET / "source_labels"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
random.seed(42)


def clear_folder(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)

    for item in folder.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


images = sorted(
    path
    for path in SOURCE_IMAGES.iterdir()
    if path.suffix.lower() in IMAGE_EXTENSIONS
)

pairs = []

for image in images:
    label = SOURCE_LABELS / f"{image.stem}.txt"

    if not label.exists():
        raise FileNotFoundError(
            f"No matching label found for image: {image.name}"
        )

    pairs.append((image, label))

random.shuffle(pairs)

total = len(pairs)
train_end = int(total * 0.80)
val_end = train_end + int(total * 0.10)

splits = {
    "train": pairs[:train_end],
    "val": pairs[train_end:val_end],
    "test": pairs[val_end:],
}

for split_name, split_pairs in splits.items():
    image_dir = DATASET / "images" / split_name
    label_dir = DATASET / "labels" / split_name

    clear_folder(image_dir)
    clear_folder(label_dir)

    for image, label in split_pairs:
        shutil.copy2(image, image_dir / image.name)
        shutil.copy2(label, label_dir / label.name)

    print(f"{split_name}: {len(split_pairs)}")

print(f"Total: {total}")
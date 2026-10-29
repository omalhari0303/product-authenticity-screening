
"""Prepare GoldNet images for authenticity model training.

Source:
    data/goldnet/real
    data/goldnet/fake

Output:
    data/authenticity/
        train/
        val/
        test/
            genuine/
            counterfeit/
"""

import random
import shutil
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Dataset paths
SOURCE_DIR = ROOT_DIR / "data" / "goldnet"
OUTPUT_DIR = ROOT_DIR / "data" / "authenticity"

# Dataset split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Reproducibility
SEED = 42

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_images(folder: Path):
    """Return all supported images from a folder."""
    return sorted(
        [
            file
            for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def split_images(images):
    """Split images into train, validation, and test sets."""

    images = images.copy()
    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    return train_images, val_images, test_images


def copy_images(images, destination: Path):
    """Copy images into the destination folder."""

    destination.mkdir(parents=True, exist_ok=True)

    for image in images:
        shutil.copy2(
            image,
            destination / image.name
        )


def prepare_dataset():
    """Create the GoldNet dataset split."""

    random.seed(SEED)

    # Validate source directories
    genuine_dir = SOURCE_DIR / "real"
    counterfeit_dir = SOURCE_DIR / "fake"

    if not genuine_dir.exists():
        raise FileNotFoundError(
            f"Genuine images not found: {genuine_dir}"
        )

    if not counterfeit_dir.exists():
        raise FileNotFoundError(
            f"Counterfeit images not found: {counterfeit_dir}"
        )

    # Load images
    genuine_images = get_images(genuine_dir)
    counterfeit_images = get_images(counterfeit_dir)

    if not genuine_images or not counterfeit_images:
        raise ValueError(
            "One or both source folders contain no images."
        )

    print("\nGoldNet Dataset")
    print("-" * 40)

    print(f"Genuine images: {len(genuine_images)}")
    print(f"Counterfeit images: {len(counterfeit_images)}")

    # Split each class independently
    genuine_train, genuine_val, genuine_test = split_images(
        genuine_images
    )

    counterfeit_train, counterfeit_val, counterfeit_test = split_images(
        counterfeit_images
    )

    # Define output directories
    splits = {
        "train": {
            "genuine": genuine_train,
            "counterfeit": counterfeit_train,
        },
        "val": {
            "genuine": genuine_val,
            "counterfeit": counterfeit_val,
        },
        "test": {
            "genuine": genuine_test,
            "counterfeit": counterfeit_test,
        },
    }

    # Copy images
    for split_name, classes in splits.items():

        for class_name, images in classes.items():

            destination = (
                OUTPUT_DIR
                / split_name
                / class_name
            )

            copy_images(images, destination)

            print(
                f"{split_name:5} | "
                f"{class_name:12} | "
                f"{len(images):4} images"
            )

    print("\nDataset preparation completed!")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    print("Script started!")

    prepare_dataset()

    print("Script finished!")
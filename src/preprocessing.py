"""Image input + preprocessing (Component 1).

Deliberately free of TensorFlow so it is cheap to import and easy to unit-test.
Normalisation (rescaling pixel values) happens *inside* the Keras models, so
this module only produces raw 0-255 float arrays. That guarantees training and
inference use identical scaling.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image, ImageOps

from .config import IMG_SIZE


def load_image(source) -> Image.Image:
    """Open an image (path, file-like or PIL image), fix EXIF rotation, convert to RGB."""
    img = source if isinstance(source, Image.Image) else Image.open(source)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def validate_image(img: Image.Image, min_side: int = 64) -> None:
    """Raise ValueError if the image is too small to carry useful detail."""
    w, h = img.size
    if min(w, h) < min_side:
        raise ValueError(
            f"Image is {w}x{h}px; at least {min_side}px on the short side is "
            "required (fine details like stitching vanish in tiny images)."
        )


def resize_with_padding(img: Image.Image, size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Resize preserving aspect ratio, padding with black. Returns uint8 (H, W, 3).

    Mirrors `pad_to_aspect_ratio=True` used by tf.keras image_dataset_from_directory
    (which pads with zeros), so training and inference see the same geometry.
    """
    h, w = size
    padded = ImageOps.pad(img, (w, h), method=Image.BILINEAR, color=(0, 0, 0))
    return np.asarray(padded, dtype=np.uint8)


def preprocess_image(source, size: Tuple[int, int] = IMG_SIZE, min_side: int = 64):
    """Full pipeline for one image.

    Returns:
        display: uint8 array (H, W, 3) for showing / overlaying Grad-CAM.
        batch:   float32 array (1, H, W, 3) with raw 0-255 values for the model.
    """
    img = load_image(source)
    validate_image(img, min_side=min_side)
    display = resize_with_padding(img, size)
    batch = display.astype(np.float32)[None, ...]
    return display, batch

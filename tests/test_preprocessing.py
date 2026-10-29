import numpy as np
import pytest
from PIL import Image

from src.evaluate import compute_metrics
from src.preprocessing import preprocess_image


def test_output_shapes_and_dtype():
    img = Image.new("RGB", (400, 200), (10, 120, 200))
    display, batch = preprocess_image(img, size=(224, 224))
    assert display.shape == (224, 224, 3) and display.dtype == np.uint8
    assert batch.shape == (1, 224, 224, 3) and batch.dtype == np.float32
    assert batch.max() <= 255.0


def test_padding_preserves_aspect_ratio():
    img = Image.new("RGB", (400, 200), (255, 255, 255))
    display, _ = preprocess_image(img, size=(224, 224))
    assert display[0, 112].sum() == 0 and display[223, 112].sum() == 0  # black bars top/bottom
    assert display[112, 112].sum() == 255 * 3


def test_grayscale_and_rgba_become_rgb():
    for mode in ("L", "RGBA"):
        _, batch = preprocess_image(Image.new(mode, (128, 128)))
        assert batch.shape[-1] == 3


def test_tiny_image_rejected():
    with pytest.raises(ValueError):
        preprocess_image(Image.new("RGB", (30, 30)))


def test_metrics_binary():
    names = ["counterfeit", "genuine"]
    m = compute_metrics([0, 0, 1, 1], [0, 1, 1, 1], names, "counterfeit")
    assert m["accuracy"] == 0.75
    assert m["counterfeit_precision"] == 1.0 and m["counterfeit_recall"] == 0.5

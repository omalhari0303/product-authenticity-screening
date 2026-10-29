"""Dataset loading for both project phases."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from .config import IMG_SIZE, SEED


# ---------- Phase 1: Fashion-MNIST (learning the foundations only) ----------
def load_fashion_mnist():
    """Returns ((x_train, y_train), (x_test, y_test)); x is uint8 (N, 28, 28, 1)."""
    from tensorflow import keras

    (xtr, ytr), (xte, yte) = keras.datasets.fashion_mnist.load_data()
    return (xtr[..., None], ytr), (xte[..., None], yte)


# ---------- Phase 2: authenticity-labelled images ----------
# Expected layout (class folder names become labels):
#   data/authenticity/
#       train/{genuine,counterfeit}/*.jpg
#       val/{genuine,counterfeit}/*.jpg
#       test/{genuine,counterfeit}/*.jpg
# IMPORTANT: split by individual item / seller / source, not by photo, otherwise
# multiple photos of the same item leak across splits and inflate every metric.
def load_split(data_dir, split: str, img_size=IMG_SIZE, batch_size=32, shuffle=False):
    import tensorflow as tf

    ds = tf.keras.utils.image_dataset_from_directory(
        Path(data_dir) / split,
        label_mode="int",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=SEED,
        pad_to_aspect_ratio=True,
    )
    class_names: List[str] = ds.class_names
    return ds.prefetch(tf.data.AUTOTUNE), class_names


def load_authenticity_splits(data_dir, img_size=IMG_SIZE, batch_size=32):
    train, names = load_split(data_dir, "train", img_size, batch_size, shuffle=True)
    val, names_v = load_split(data_dir, "val", img_size, batch_size)
    test, names_t = load_split(data_dir, "test", img_size, batch_size)
    assert names == names_v == names_t, "Class folders differ between splits"
    return train, val, test, names


def class_counts(data_dir, split: str, class_names: List[str]) -> List[int]:
    return [len([p for p in (Path(data_dir) / split / c).iterdir() if p.is_file()]) for c in class_names]


def balanced_class_weights(counts: List[int]) -> dict:
    total, k = sum(counts), len(counts)
    return {i: total / (k * c) for i, c in enumerate(counts) if c > 0}


def dataset_to_numpy(ds) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for x, y in ds:
        xs.append(x.numpy())
        ys.append(y.numpy())
    return np.concatenate(xs), np.concatenate(ys)

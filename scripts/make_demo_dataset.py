"""Generate a SYNTHETIC genuine/counterfeit dataset to smoke-test the pipeline.

"Genuine" = evenly spaced, uniform stitch dashes on textured leather colour.
"Counterfeit" = jittered spacing/length and slightly off colour.
This has NO relation to real luxury goods. It only proves the code runs
end-to-end. Replace with real, ethically sourced, labelled data.

    python scripts/make_demo_dataset.py --out data/demo_synthetic
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def make_image(rng, genuine: bool, size=256):
    base = np.array([60, 90, 140] if genuine else [70, 95, 130], dtype=np.float32) + rng.normal(0, 4, 3)
    img = np.ones((size, size, 3), np.float32) * base
    img += rng.normal(0, 8, (size, size, 1))  # leather-like grain
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    for y in (size // 4, 3 * size // 4):
        x, step, length = 15, 22, 12
        while x < size - 30:
            j = 0 if genuine else rng.integers(-5, 6)
            l = length if genuine else int(length + rng.integers(-5, 6))
            yy = y + (0 if genuine else int(rng.integers(-3, 4)))
            cv2.line(img, (x + j, yy), (x + j + l, yy), (225, 225, 205), 2)
            x += step if genuine else int(step + rng.integers(-6, 7))
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo_synthetic")
    ap.add_argument("--per-class", type=int, default=150)
    args = ap.parse_args()
    rng = np.random.default_rng(42)
    splits = {"train": 0.7, "val": 0.15, "test": 0.15}
    for cls, genuine in (("genuine", True), ("counterfeit", False)):
        i = 0
        for split, frac in splits.items():
            d = Path(args.out) / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for _ in range(int(args.per_class * frac)):
                cv2.imwrite(str(d / f"{cls}_{i:04d}.png"), make_image(rng, genuine))
                i += 1
    print("Synthetic dataset written to", args.out)


if __name__ == "__main__":
    main()

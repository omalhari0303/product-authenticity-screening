"""Phase 1 - foundations on Fashion-MNIST: Logistic Regression vs CNN.

This teaches the pipeline (train / evaluate / compare). It is NOT an
authenticity model: Fashion-MNIST has no genuine/counterfeit labels.

    python -m src.train_fashion_mnist --epochs 5
"""
import argparse

import pandas as pd

from .config import FASHION_CLASSES, MODELS_DIR, REPORTS_DIR, SEED
from .data import load_fashion_mnist
from .evaluate import compute_metrics, plot_confusion_matrix, save_metrics
from .models import build_cnn, build_logistic_regression


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr-samples", type=int, default=10000, help="train subset size for logistic regression (speed)")
    args = ap.parse_args()

    out = REPORTS_DIR / "fashion_mnist"
    (xtr, ytr), (xte, yte) = load_fashion_mnist()
    rows = []

    # --- Logistic regression on flattened pixels ---
    lr = build_logistic_regression()
    n = args.lr_samples
    lr.fit(xtr[:n].reshape(n, -1) / 255.0, ytr[:n])
    pred = lr.predict(xte.reshape(len(xte), -1) / 255.0)
    m = compute_metrics(yte, pred, FASHION_CLASSES)
    save_metrics(m, out / "logreg_metrics.json")
    plot_confusion_matrix(yte, pred, FASHION_CLASSES, out / "logreg_cm.png", "Logistic Regression")
    rows.append({"model": "Logistic Regression", **{k: m[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}})

    # --- CNN from scratch ---
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    cnn = build_cnn(input_shape=(28, 28, 1), num_classes=10)
    cnn.fit(xtr, ytr, validation_split=0.1, epochs=args.epochs, batch_size=128, verbose=2)
    pred = cnn.predict(xte, batch_size=256, verbose=0).argmax(axis=1)
    m = compute_metrics(yte, pred, FASHION_CLASSES)
    save_metrics(m, out / "cnn_metrics.json")
    plot_confusion_matrix(yte, pred, FASHION_CLASSES, out / "cnn_cm.png", "CNN")
    rows.append({"model": "CNN (scratch)", **{k: m[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}})
    MODELS_DIR.mkdir(exist_ok=True)
    cnn.save(MODELS_DIR / "fashion_cnn.keras")

    df = pd.DataFrame(rows).round(4)
    df.to_csv(out / "comparison.csv", index=False)
    print("\n", df.to_string(index=False))


if __name__ == "__main__":
    main()

"""Evaluation (Component 3): accuracy, precision, recall, F1, confusion matrix."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(y_true, y_pred, class_names: List[str], positive_class: Optional[str] = None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(class_names)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f),
        "n_samples": int(len(y_true)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "report": classification_report(
            y_true, y_pred, labels=labels, target_names=class_names, zero_division=0, output_dict=True
        ),
    }
    if positive_class and positive_class in class_names:
        idx = class_names.index(positive_class)
        pp, rr, ff, _ = precision_recall_fscore_support(
            y_true == idx, y_pred == idx, average="binary", zero_division=0
        )
        out[f"{positive_class}_precision"] = float(pp)
        out[f"{positive_class}_recall"] = float(rr)
        out[f"{positive_class}_f1"] = float(ff)
    return out


def plot_confusion_matrix(y_true, y_pred, class_names: List[str], path, title: str = "Confusion matrix"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    side = max(4, len(class_names) * 0.7)
    fig, ax = plt.subplots(figsize=(side, side))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_metrics(metrics: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))


def predict_keras(model, ds):
    """Single pass over an (unshuffled) tf.data dataset -> y_true, y_pred, y_prob."""
    ys, probs = [], []
    for x, y in ds:
        probs.append(model.predict_on_batch(x))
        ys.append(y.numpy())
    y_prob = np.concatenate(probs)
    return np.concatenate(ys), y_prob.argmax(axis=1), y_prob

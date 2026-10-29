"""Explainable AI (Component 4): Grad-CAM for Keras CNNs."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def find_last_feature_layer(model) -> str:
    """Name of the last layer that outputs a 4-D (batch, H, W, C) feature map."""
    for layer in reversed(model.layers):
        shape = getattr(getattr(layer, "output", None), "shape", None)
        if shape is not None and len(shape) == 4:
            return layer.name
    raise ValueError("No 4-D feature-map layer found; Grad-CAM needs a convolutional model.")


def make_gradcam_heatmap(model, img_batch: np.ndarray, class_index: Optional[int] = None,
                         layer_name: Optional[str] = None):
    """Returns (heatmap in [0,1] of shape (h, w), class_index used)."""
    import tensorflow as tf

    layer_name = layer_name or find_last_feature_layer(model)
    grad_model = tf.keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    x = tf.convert_to_tensor(img_batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        score = preds[:, class_index]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))                 # channel importance
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)             # weighted feature map
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy(), class_index


def overlay_heatmap(img_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a [0,1] heatmap onto an RGB uint8 image; returns RGB uint8."""
    h, w = img_uint8.shape[:2]
    hm = cv2.resize(heatmap.astype(np.float32), (w, h))
    hm = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    return np.clip((1 - alpha) * img_uint8 + alpha * hm, 0, 255).astype(np.uint8)

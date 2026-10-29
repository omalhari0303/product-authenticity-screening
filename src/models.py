"""Model zoo (Component 2): Logistic Regression, CNN from scratch, MobileNetV2.

All Keras models take RAW 0-255 pixels; rescaling is the first layer inside the
model. Augmentation avoids horizontal flips on purpose: mirroring a logo or
monogram produces an image that looks "counterfeit" for the wrong reason.
"""
from __future__ import annotations

from typing import Tuple

HEAD_LAYER_NAMES = {"gap", "head_dropout", "predictions"}


def build_logistic_regression():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, class_weight="balanced"),
    )


def _augmentation():
    from tensorflow import keras
    from tensorflow.keras import layers

    return keras.Sequential(
        [
            layers.RandomRotation(0.03),
            layers.RandomTranslation(0.05, 0.05),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def compile_model(model, lr: float = 1e-3):
    from tensorflow import keras

    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn(input_shape: Tuple[int, int, int] = (224, 224, 3), num_classes: int = 2, augment: bool = False):
    """Small CNN trained from scratch. Last 4-D layer is `relu4` (used by Grad-CAM)."""
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=input_shape)
    x = _augmentation()(inputs) if augment else inputs
    x = layers.Rescaling(1.0 / 255)(x)
    filters = [32, 64, 128, 128]
    for i, f in enumerate(filters, start=1):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(momentum=0.9, name=f"bn{i}")(x)
        x = layers.ReLU(name=f"relu{i}")(x)
        if i < len(filters):
            x = layers.MaxPooling2D(name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.4, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return compile_model(keras.Model(inputs, outputs, name="cnn_scratch"))


def build_mobilenet_v2(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    weights="imagenet",
    augment: bool = True,
):
    """Transfer learning. Backbone starts frozen; call `unfreeze_backbone` to fine-tune.

    The backbone is built *inline* (input_tensor=...) rather than nested, so
    every layer, including the final `out_relu` feature map used by Grad-CAM,
    is directly reachable via model.get_layer().
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=input_shape)
    x = _augmentation()(inputs) if augment else inputs
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_rescale")(x)
    base = keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights=weights, input_tensor=x
    )
    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dropout(0.3, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs, outputs, name="mobilenetv2_transfer")
    freeze_backbone(model)
    return compile_model(model, lr=1e-3)


def freeze_backbone(model) -> None:
    for layer in model.layers:
        if layer.name not in HEAD_LAYER_NAMES:
            layer.trainable = False


def unfreeze_backbone(model, last_n: int = 30, lr: float = 1e-5):
    """Fine-tune the last `last_n` backbone layers (BatchNorm stays frozen)."""
    from tensorflow import keras

    backbone = [l for l in model.layers if l.name not in HEAD_LAYER_NAMES]
    for layer in backbone[-last_n:]:
        if not isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = True
    return compile_model(model, lr=lr)  # must recompile after changing trainable flags

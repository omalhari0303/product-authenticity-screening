"""Streamlit app (Component 5): upload -> predict -> explain.

    streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st

from src.config import CLASS_NAMES_PATH, DEFAULT_MODEL_PATH, POSITIVE_CLASS
from src.gradcam import make_gradcam_heatmap, overlay_heatmap
from src.preprocessing import preprocess_image

st.set_page_config(page_title="Product Authenticity Screening", page_icon="🔍", layout="wide")


@st.cache_resource
def load_model(path: str):
    import tensorflow as tf

    return tf.keras.models.load_model(path)


st.title("🔍 Product Authenticity Screening")
st.caption("Educational prototype. Output is an AI screening signal, not an authentication certificate.")

model_path = Path(st.sidebar.text_input("Model path", str(DEFAULT_MODEL_PATH)))
threshold = st.sidebar.slider(
    "Flag threshold", 0.05, 0.95, 0.50, 0.05,
    help="Images whose model score for the flagged class is at or above this are flagged. "
         "Tune it on validation data, not by eye.",
)
alpha = st.sidebar.slider("Heatmap opacity", 0.1, 0.9, 0.4, 0.05)

if not model_path.exists() or not CLASS_NAMES_PATH.exists():
    st.warning(
        "No trained model found. Train one first, e.g.\n\n"
        "```\npython scripts/make_demo_dataset.py\n"
        "python -m src.train_authenticity --data-dir data/demo_synthetic --model cnn --epochs 10\n```"
    )
    st.stop()

model = load_model(str(model_path))
class_names = json.loads(CLASS_NAMES_PATH.read_text())

file = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png", "webp"])
if file is None:
    st.info("Upload a clear, well-lit photo. Close-ups of stitching, hardware and logos work best.")
    st.stop()

try:
    display, batch = preprocess_image(file)
except Exception as e:  # unreadable or too-small image
    st.error(f"Could not process image: {e}")
    st.stop()

probs = model.predict(batch, verbose=0)[0]
flag_idx = class_names.index(POSITIVE_CLASS) if POSITIVE_CLASS in class_names else int(np.argmax(probs))
score = float(probs[flag_idx])
flagged = score >= threshold
label = f"Potentially {class_names[flag_idx]}" if flagged else "Consistent with genuine examples"

heatmap, _ = make_gradcam_heatmap(model, batch, class_index=flag_idx)
overlay = overlay_heatmap(display, heatmap, alpha)

left, mid, right = st.columns(3)
left.image(display, caption="Preprocessed input", use_container_width=True)
mid.image(overlay, caption=f"Grad-CAM for '{class_names[flag_idx]}' class", use_container_width=True)
with right:
    (st.error if flagged else st.success)(f"**{label}**")
    st.metric(f"Model score ({class_names[flag_idx]})", f"{score:.2f}")
    st.progress(min(max(score, 0.0), 1.0))
    st.caption("Uncalibrated softmax score, not a true probability.")
    st.dataframe({"class": class_names, "score": [round(float(p), 4) for p in probs]}, hide_index=True)

st.divider()
st.markdown(
    "**How to read this:** warm regions are where the network looked when producing the score for the "
    "selected class. A heatmap showing the background, a watermark or the photo border means the model is "
    "likely using shortcuts rather than product features. **This is not a definitive authentication.**"
)

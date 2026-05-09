import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Accident Detection AI",
    page_icon="🚨",
    layout="centered",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 720px; }
    div[data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #444;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH  = "accident_detection_model_new.h5"
IMG_SIZE    = (224, 224)
THRESHOLD   = 0.5

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"❌ Model file `{MODEL_PATH}` not found.\n\n"
            "Make sure `accident_detection_model_new.h5` is pushed to the root of your GitHub repo."
        )
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ── Prediction helper ─────────────────────────────────────────────────────────
def predict(img: Image.Image):
    """
    Training used flow_from_directory → class indices are alphabetical:
        Accident      → 0
        Non Accident  → 1
    Sigmoid output ~1.0 = Non Accident, ~0.0 = Accident
    """
    arr  = np.array(img.convert("RGB").resize(IMG_SIZE), dtype=np.float32) / 255.0
    prob = float(model.predict(np.expand_dims(arr, 0), verbose=0)[0][0])

    if prob >= THRESHOLD:
        label      = "Non Accident"
        confidence = prob
    else:
        label      = "Accident"
        confidence = 1.0 - prob

    return label, confidence, prob

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🚨 Accident Detection from CCTV")
st.caption(
    "Upload a CCTV frame or road image — the model classifies it as "
    "**Accident** or **Non Accident**."
)
st.divider()

uploaded = st.file_uploader(
    "Upload an image (JPG / PNG / WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded:
    img = Image.open(uploaded)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(img, caption="Uploaded image", use_container_width=True)

    with col2:
        with st.spinner("Analysing…"):
            label, confidence, raw = predict(img)

        pct = confidence * 100

        if label == "Accident":
            st.error(f"### ⚠️ {label}")
            st.metric("Confidence", f"{pct:.1f}%")
            st.progress(confidence)
            st.markdown("> **Action recommended:** Alert emergency services or traffic control.")
        else:
            st.success(f"### ✅ {label}")
            st.metric("Confidence", f"{pct:.1f}%")
            st.progress(confidence)
            st.markdown("> No accident detected in this frame.")

        with st.expander("🔬 Raw model output"):
            st.write(f"Sigmoid output : `{raw:.6f}`")
            st.write(f"Threshold      : `{THRESHOLD}`")
            st.write(f"Prediction     : `{'Non Accident' if raw >= THRESHOLD else 'Accident'}`")

else:
    st.info("👆 Upload a CCTV / road image above to get started.")

st.divider()
st.caption("Model: MobileNetV2 fine-tuned on Kaggle — Accident Detection from CCTV Footage dataset.")

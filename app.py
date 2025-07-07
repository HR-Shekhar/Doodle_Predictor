# app.py

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import joblib

# ✅ Load model saved in `.keras` format (TF ≥ 2.13 compatible)
# Make sure the model was saved WITHOUT using keras.Input() directly — use shape in first layer
model = load_model("doodle_model.keras", compile=False, safe_mode=False)
data_scaler = joblib.load("data_scaler.pkl")
scaler = data_scaler["scaler"]


# ✅ Classes must match your training labels
class_names = ["apple", "bat", "circle", "clock", "cloud",
               "crown", "diamond", "donut", "fish",
               "hot_dog", "lightning", "mountain", "skull",
               "smiley_face", "square", "star", "sun", "t-shirt", "tree"]

st.set_page_config(page_title="Doodle Classifier", page_icon="🎨")
st.title("🎨 Doodle Classifier with AI")
st.markdown("Draw an object in the canvas and let the AI predict what it is!")

# --- Sidebar canvas settings
st.sidebar.header("✏️ Canvas Settings")
stroke_width = st.sidebar.slider("Stroke Width:", 1, 25, 10)
stroke_color = st.sidebar.color_picker("Stroke Color:", "#000000")
bg_color = st.sidebar.color_picker("Background Color:", "#FFFFFF")

# --- Drawing canvas
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",  # Transparent fill
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=True,
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# --- Prediction
if st.button("🧠 Predict"):
    if canvas_result.image_data is not None:
        # Convert canvas RGBA to grayscale
        img = cv2.cvtColor(np.array(canvas_result.image_data, dtype=np.uint8), cv2.COLOR_RGBA2GRAY)

        # Invert image if background is white
        if bg_color == "#FFFFFF":
            img = 255 - img

        # Resize and normalize
        img = cv2.resize(img, (28, 28))
        img = img.reshape(1, 28 * 28)  # Flatten
        img = scaler.transform(img.astype("float32"))
        

        # Predict
        logits = model.predict(img)
        preds = tf.nn.softmax(logits, axis=1).numpy()
        pred_class = np.argmax(preds, axis=1)
        confidence = np.max(preds)

        # --- Show predictions
        st.markdown("### 🎯 Prediction")
        st.success(f"**{[pred_class]}** with **{confidence * 100:.2f}%** confidence.")

        st.markdown("### 👁️ What the model saw")
        st.image(img.reshape(28, 28), width=150, clamp=True, channels='L')

    else:
        st.warning("🖌️ Please draw something before predicting.")

st.markdown("---")
st.markdown("Built with ❤️ using **Streamlit + Keras**")

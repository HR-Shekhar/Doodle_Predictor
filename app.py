# app.py

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# ✅ Load updated model (works because model was built with Input())
model = load_model("doodle_model.keras")

# Define correct class names
class_names = ["circle", "crown", "skull", "smiley_face", "square", "star"]  # ✅ Match your training classes

# UI
st.title("🎨 Doodle Classifier with AI")
st.markdown("Draw something below and let AI predict!")

# Sidebar settings
st.sidebar.header("Canvas Settings")
stroke_width = st.sidebar.slider("🖊️ Stroke Width", 1, 25, 10)
stroke_color = st.sidebar.color_picker("🎨 Stroke Color", "#000000")
bg_color = st.sidebar.color_picker("🧻 Background Color", "#FFFFFF")

# Drawing canvas
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=True,
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Prediction logic
if st.button("🧠 Predict"):
    if canvas_result.image_data is not None:
        img = np.array(canvas_result.image_data, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

        # Invert for white bg drawings
        if bg_color == "#FFFFFF":
            img = 255 - img

        img = cv2.resize(img, (28, 28))
        img = img.astype("float32") / 255.0
        img = img.reshape(1, 784)  # ✅ Flatten to match (784,) input shape

        # Predict
        preds = model.predict(img)
        pred_class = np.argmax(preds)
        confidence = np.max(preds)

        st.markdown("### 🎯 Prediction:")
        st.success(f"**{class_names[pred_class]}** with **{confidence * 100:.2f}%** confidence.")
        st.markdown("### 🔍 What the model saw:")
        st.image(img.reshape(28, 28), width=150, clamp=True, channels='L')
    else:
        st.warning("🖌️ Please draw something first!")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Keras.")
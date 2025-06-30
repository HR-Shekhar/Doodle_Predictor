# app.py

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# ✅ FIXED: Load .h5 model (older format to avoid 'batch_shape' issue)
model = load_model("doodle_model.h5")

# ⚠️ Make sure class names match training data
class_names = ["circle", "crown", "skull", "smiley_face", "square", "star"]

st.title("🎨 Doodle Classifier with AI")
st.markdown("Draw an object in the canvas below and let the AI guess what it is!")

st.sidebar.header("Canvas Settings")
stroke_width = st.sidebar.slider("🖊️ Stroke Width: ", 1, 25, 10)
stroke_color = st.sidebar.color_picker("🎨 Stroke Color: ", "#000000")
bg_color = st.sidebar.color_picker("🧻 Background Color: ", "#FFFFFF")

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

if st.button("🧠 Predict"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data
        img = cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_RGBA2GRAY)

        if bg_color == "#FFFFFF":
            img = 255 - img

        img = cv2.resize(img, (28, 28))
        img = img.astype("float32") / 255.0
        img = img.reshape(1, 784)  # ✅ FIXED: match model input shape

        preds = model.predict(img)
        pred_class = np.argmax(preds)
        confidence = np.max(preds)

        st.markdown("### 🎯 Prediction:")
        st.success(f"**{class_names[pred_class]}** with **{confidence * 100:.2f}%** confidence.")

        st.markdown("### 🔍 What the model saw:")
        st.image(img.reshape(28, 28), width=150, clamp=True, channels='L')
    else:
        st.warning("🖌️ Please draw something first!")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Keras.")

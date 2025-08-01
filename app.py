# app.py

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from keras.layers import Dropout
from keras.regularizers import L2
import numpy as np
import cv2
import os

# ✅ Load model
os.environ["TF_KERAS_RESET_NAME_SCOPES"] = "1"  # Critical flag!

from tensorflow.keras.models import load_model
model = load_model("doodle_model.keras", compile=False, safe_mode=False, custom_objects={"Dropout": Dropout, "L2": L2})



class_names = ["apple", "bat", "circle", "clock", "cloud",
               "crown", "diamond", "donut", "fish",
               "hot_dog", "lightning", "mountain", "skull",
               "smiley_face", "square", "star", "sun", "t-shirt", "tree"]

# Streamlit app setup
st.set_page_config(page_title="Doodle Classifier", page_icon="🎨")
st.title("🎨 Doodle Classifier")
st.markdown("Draw an object in the canvas and let the DNN predict what it is!")
st.markdown("Categories it can predict: "
"apple, bat, circle, clock, cloud, crown, diamond, donut, fish, hot_dog, "
"lightning, mountain, skull, smiley_face, square, star, sun, t-shirt, tree")

# --- Sidebar canvas settings
st.sidebar.header("✏️ Canvas Settings")
st.sidebar.subheader("Don't modify these for the sake of model's performance")
stroke_width = st.sidebar.slider("Stroke Width:", 19, 25, 25)
stroke_color = st.sidebar.color_picker("Stroke Color:", "#000000")
bg_color = st.sidebar.color_picker("Background Color:", "#FFFFFF")

# --- Drawing canvas
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",  # Transparent fill
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=True,
    height=550,
    width=550,
    drawing_mode="freedraw",
    key="canvas"
)

# --- Prediction
if st.button("🧠 Predict"):
    if canvas_result.image_data is not None:
        # Convert RGBA canvas to grayscale
        img = cv2.cvtColor(np.array(canvas_result.image_data, dtype=np.uint8), cv2.COLOR_RGBA2GRAY)

        # Invert if background is white (like training)
        if bg_color.upper() == "#FFFFFF":
            img = 255 - img

        # Resize to 28x28
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

        # Normalize (same as training)
        img = img.astype("float32") / 255.0

        # Flatten for dense model
        img = img.reshape(1, 28 * 28)

        # Predict
        logits = model.predict(img)
        probs = tf.nn.softmax(logits, axis=1).numpy()
        pred_class = np.argmax(probs, axis=1).item()
        confidence = np.max(probs)

        # --- Results
        st.markdown("### 🎯 Prediction")
        st.success(f"**{class_names[pred_class]}** with **{confidence * 100:.2f}%** confidence.")

        st.markdown("### 👁️ What the model saw")
        st.image(img.reshape(28, 28), width=150, clamp=True, channels='L')

    else:
        st.warning("🖌️ Please draw something before predicting.")

st.markdown("---")
st.markdown("Built with ❤️ using **Tensorflow** by **Himanshu Shekhar**")

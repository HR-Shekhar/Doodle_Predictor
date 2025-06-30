import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Load the trained model
model = load_model("doodle_model.keras")

# Define class names in order
class_names = ['apple', 'star', 'triangle', 'fish', 'house']  # update as per your model

# Title
st.title("🎨 Doodle Classifier with AI")
st.markdown("Draw an object in the canvas below and let the AI guess what it is!")

# Sidebar
st.sidebar.header("Canvas Settings")
stroke_width = st.sidebar.slider("🖊️ Stroke Width: ", 1, 25, 10)
stroke_color = st.sidebar.color_picker("🎨 Stroke Color: ", "#000000")
bg_color = st.sidebar.color_picker("🧻 Background Color: ", "#FFFFFF")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",  # Transparent
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=True,
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Prediction section
if st.button("🧠 Predict"):
    if canvas_result.image_data is not None:
        # Convert canvas image to grayscale numpy
        img = canvas_result.image_data

        # Convert to 0-255, grayscale
        img = cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_RGBA2GRAY)

        # Invert colors if needed (black bg, white strokes)
        if bg_color == "#FFFFFF":
            img = 255 - img

        # Resize to match model input
        img = cv2.resize(img, (28, 28))

        # Normalize to 0-1
        img = img.astype("float32") / 255.0

        # Reshape for model input (1, 28, 28, 1)
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)

        # Predict
        preds = model.predict(img)
        pred_class = np.argmax(preds)
        confidence = np.max(preds)

        # Display
        st.markdown("### 🎯 Prediction:")
        st.success(f"**{class_names[pred_class]}** with **{confidence * 100:.2f}%** confidence.")

        # Optional: Show resized input
        st.markdown("### 🔍 What the model saw:")
        st.image(img.reshape(28, 28), width=150, clamp=True, channels='L')
    else:
        st.warning("🖌️ Please draw something first!")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Keras.")

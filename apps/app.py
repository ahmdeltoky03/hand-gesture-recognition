# Set the page title and icon
import streamlit as st
st.set_page_config(page_title="Hand Gesture Recognition", page_icon="🖐️")

import tensorflow as tf
from PIL import Image
import numpy as np
from utils import preprocess_image, load_gesture_model
from constants import CLASS_INDICES, INDEX_TO_CLASS

# Load the model
model = load_gesture_model()

# --- App Layout ---
st.title("🖐️ Hand Gesture Recognition System")
st.markdown("""
Upload a hand gesture image, or use your camera to capture one. The model will predict what gesture it is.

**Recognized gesture classes:**
- fist
- five
- ok
- peace
- rad
- straight
- thumbs

If your image does not match any of these gestures, the system will return **'none'**.
""")
input_image = None

option = st.radio("Select input method:", ("Upload Image", "Use Camera"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        input_image = Image.open(uploaded_file).convert("RGB")
elif option == "Use Camera":
    camera_file = st.camera_input("Take a picture")
    if camera_file:
        input_image = Image.open(camera_file).convert("RGB")

if input_image is not None:
    st.image(input_image, caption="Input Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        processed_img = preprocess_image(input_image)
        prediction = model.predict(processed_img)[0]
        predicted_index = np.argmax(prediction)
        predicted_label = INDEX_TO_CLASS.get(predicted_index, "Unknown")
        confidence = float(tf.nn.softmax(prediction)[predicted_index])

        # Show prediction vector with class names
        softmax_pred = tf.nn.softmax(prediction).numpy()
        pred_with_labels = [
            f"{INDEX_TO_CLASS.get(i, 'Unknown')}: {prob:.4f}" 
            for i, prob in enumerate(softmax_pred)
        ]
        st.write("Prediction vector:", pred_with_labels)

    st.markdown(f"<h3 style='text-align: center;'>🧠 Prediction: {predicted_label}</h3>", unsafe_allow_html=True)
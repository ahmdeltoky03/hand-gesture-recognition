import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from utils import preprocess_image, load_gesture_model
from constants import CLASS_INDICES, INDEX_TO_CLASS

# Set the page title and icon
st.set_page_config(page_title="Hand Gesture Recognition", page_icon="🖐️")

# Load the model
model = load_gesture_model()


# --- App Layout ---
st.title("🖐️ Hand Gesture Recognition System")
st.markdown("Upload a hand gesture image, and the model will predict what gesture it is.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        processed_img = preprocess_image(image)
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

    # st.success(f"🧠 **Prediction:** `{predicted_label}`")
    st.markdown(f"<h3 style='text-align: center;'>🧠 Prediction: {predicted_label}</h3>", unsafe_allow_html=True)
    # st.markdown(f"**Confidence Score:** `{confidence:.2f}`")

    # st.progress(confidence)

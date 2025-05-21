from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from constants import target_size
import streamlit as st

def preprocess_image(img, target_size= target_size):
    img = img.resize(target_size).convert('L')  # Convert to grayscale
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@st.cache_resource
def load_gesture_model():
    return load_model("../models/gesture_model.h5")


from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from constants import target_size
import streamlit as st
from PIL import Image
import cv2

def preprocess_image(img, target_size=target_size):
    # Convert PIL Image to numpy array
    img = img.resize(target_size).convert('RGB')
    img_np = np.array(img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    # Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu's thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert if background is white
    if np.sum(binary == 255) > np.sum(binary == 0):
        binary = cv2.bitwise_not(binary)
    
    # Normalize and expand dims for model
    img_array = binary.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
    return img_array

@st.cache_resource
def load_gesture_model():
    # return load_model("../models/gesture_model.h5")
    return load_model("models/gesture_model.h5")
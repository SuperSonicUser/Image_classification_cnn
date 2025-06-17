import streamlit as st
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
import cv2
from PIL import Image

# Load your saved model
model = load_model('cnn_model.h5')

# CIFAR-10 class names (same as your training labels)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Image preprocessing function
def preprocess_image(image):
    # Resize to 32x32 (CIFAR-10 size)
    image = image.resize((32, 32))
    image = np.array(image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit UI
st.title("Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.write(f"Predicted Class: **{predicted_class}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

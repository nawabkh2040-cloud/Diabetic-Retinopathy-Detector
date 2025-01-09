import os
import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
from io import BytesIO

# Hide warnings and info logs
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force TensorFlow to run on CPU if necessary

# Load the pre-trained model using TensorFlow
model = None

try:
    model = tf.keras.models.load_model('my_model.keras')  # Replace with your model path (e.g., .h5 or .keras)
    st.write("Model loaded successfully.")
except Exception as e:
    st.write(f"Error loading model: {e}")

# Class labels (same as in your model training)
class_names = {
    0: 'no_DR',
    1: 'mild_DR',
    2: 'moderate_DR',
    3: 'severe_DR',
    4: 'proliferative_DR'
}

# Image preprocessing function
def preprocess_image(uploaded_file):
    try:
        # Read the image as bytes
        image_bytes = BytesIO(uploaded_file.read())
        
        # Use OpenCV to decode the image
        img = cv2.imdecode(np.frombuffer(image_bytes.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Invalid image file.")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (224, 224))  # Resize to the required size for the model
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = img / 255.0  # Normalize the image
        
        return img
    except Exception as e:
        st.write(f"Error preprocessing image: {e}")
        return None

# Streamlit app UI
st.title("Diabetic Retinopathy Prediction")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Predict button
if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    if st.button("Predict"):
        if model is not None:
            img = preprocess_image(uploaded_file)
            if img is not None:
                try:
                    # Make prediction
                    prediction = model.predict(img)
                    predicted_class = np.argmax(prediction, axis=1)[0]
                    predicted_label = class_names[predicted_class]
                    confidence = prediction[0][predicted_class] * 100

                    # Show prediction results
                    st.write(f"Prediction: {predicted_label}")
                    st.write(f"Confidence: {confidence:.2f}%")
                except Exception as e:
                    st.write(f"Error during prediction: {e}")
        else:
            st.write("Model is not loaded.")

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pickle

# Load the trained model and class indices
model = load_model(r"models/plant_disease_model.h5.h5")

with open(r"models/class_indices.pkl", 'rb') as f:
    class_indices = pickle.load(f)

# Reverse the class indices dictionary
class_indices = {v: k for k, v in class_indices.items()}

# Function to predict the class of an image
def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 100.0

    prediction = model.predict(img_array)
    predicted_class = class_indices[np.argmax(prediction)]

    return predicted_class

st.title("Plant Disease Classifier")
st.image(r"images/img4.jpg",width=710)

st.write("""
Upload an image of a plant leaf, and the model will predict the type of disease.
""")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    st.write("")
    st.write("Classifying...")

    # Predict the disease
    predicted_class = predict_disease("temp.jpg")

    st.write(f"Predicted Disease: {predicted_class}")
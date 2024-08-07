# Importing required libraries
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image


# Defining class names
CLASS_NAMES = []
f = open('breeds.txt', 'r')

for breed in f.readlines():
    CLASS_NAMES.append(breed)

f.close()

CLASS_NAMES.sort()

# Configuring Streamlit app
st.title("Dog Breed Classifier :dog:")
st.markdown("Welcome to my Dog Breed Classifier! This application uses a deep learning model to predict the breed of a dog from an image. Please upload an image file below, and the app will predict the breed of the dog.")
st.markdown("Note that the model can only recognize 120 breeds. Hoping to add more in the future :)")

# Upload button for dog image
dog_image = st.file_uploader("Please upload an image file of the dog:", type=["jpg", "jpeg", "png"])

# Predict button
submit = st.button('Predict')

# Functionality for Predict button
if submit:
    if dog_image is not None:

        # Load the model
        model = tf.keras.models.load_model('./final_model.keras')

        img = image.load_img(dog_image, target_size=(350, 350))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize to [0, 1]
        predictions = model.predict(img_array)

        # Display the predicted dog breed
        prediction = CLASS_NAMES[int(np.argmax(predictions, axis=1))].replace('_', ' ')
        st.title(f"The dog breed is most likely a {prediction}.")
        st.image(dog_image)
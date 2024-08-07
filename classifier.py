import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = tf.keras.models.load_model('./final_model.keras')

def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0, 1] if the model expects this
    return img_array

CLASS_NAMES = []
f = open('breeds.txt', 'r')

for breed in f.readlines():
    CLASS_NAMES.append(breed)

f.close()

CLASS_NAMES.sort()

# Example usage
img_path = './path/to/img' # hard-coded path
preprocessed_img = preprocess_image(img_path, target_size=(350, 350))

# Make predictions
predictions = model.predict(preprocessed_img)
predicted_class = CLASS_NAMES[int(np.argmax(predictions, axis=1))]

print(f'Predicted class: {predicted_class}')



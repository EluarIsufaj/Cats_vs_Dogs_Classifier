import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load your saved model
model = tf.keras.models.load_model('cats_vs_dogs_model.h5')

# Load the image from file path or URL (make sure you have it saved locally)
img_path = 'image_path_or_url'
img = image.load_img(img_path, target_size=(150, 150))

# Convert to array and normalize
img_array = image.img_to_array(img) / 255.0

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

if prediction[0] > 0.5:
    print("It's a dog!")
else:
    print("It's a cat!")

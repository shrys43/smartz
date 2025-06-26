import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

# Load model
model = load_model("fabric_pattern_classifier.h5")

# Class labels
class_names = list(train_gen.class_indices.keys())

# Load and preprocess new image
img_path = 'test_image.jpg'
img = image.load_img(img_path, target_size=image_size)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]
print(f"Predicted Pattern: {predicted_class}")

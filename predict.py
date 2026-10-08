
import sys
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np

# Get the folder where predict.py is located
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "dog_cat_mobilenetv2.keras"
)

def predict_image(image_path):
    model = load_model(MODEL_PATH)

    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    probability = model.predict(image_array, verbose=0)[0][0]

    if probability >= 0.5:
        label = "Dog"
        confidence = probability
    else:
        label = "Cat"
        confidence = 1 - probability

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    predict_image(sys.argv[1])

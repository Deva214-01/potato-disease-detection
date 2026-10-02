import os

os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("potato_disease_model.keras")

# Class names
class_names = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            image = Image.open(file).convert("RGB")
            image = image.resize((224, 224))

            image_array = np.array(image)
            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(image_array, verbose=0)

            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class]) * 100

            prediction = class_names[predicted_class]

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
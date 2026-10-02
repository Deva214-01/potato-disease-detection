from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="potato_disease_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
        file = request.files.get("image")

        if file:
            image = Image.open(file).convert("RGB")
            image = image.resize((224, 224))

            image_array = np.array(image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            # MobileNetV2 preprocessing
            

            interpreter.set_tensor(
                input_details[0]["index"],
                image_array
            )

            interpreter.invoke()

            predictions = interpreter.get_tensor(
                output_details[0]["index"]
            )

            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class]) * 100

            prediction = class_names[predicted_class]

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import base64
import re

model = tf.keras.models.load_model("digit_model.h5")
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["image"]
    image_data = re.sub('^data:image/.+;base64,', '', data)
    image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('L')
    image = image.resize((28, 28))
    image = np.array(image)
    image = 255 - image  # invert black/white
    image = image / 255.0
    image = image.reshape(1, 28, 28, 1)
    pred = model.predict(image)
    return jsonify({"prediction": int(np.argmax(pred))})

if __name__ == "__main__":
    app.run(debug=True)

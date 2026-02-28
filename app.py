
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

model = tf.keras.models.load_model("diabetic-retinopathy.h5")

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.reshape(img, (1,224,224,3))
    prediction = model.predict(img)
    return "Diabetic Retinopathy Detected" if prediction[0][0] > 0.5 else "No Diabetic Retinopathy"

@app.route("/", methods=["GET","POST"])
def index():
    result = ""
    if request.method == "POST":
        file = request.files["file"]
        filepath = os.path.join("static", file.filename)
        file.save(filepath)
        result = predict_image(filepath)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)

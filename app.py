import os
import numpy as np
from flask import Flask, render_template, request
import sklearn
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import uuid

app = Flask(__name__)

# Load model once at startup
model = keras.models.load_model("model/brain_tumor_model_final.keras")

# IMPORTANT: use same class order as training
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_idx = np.argmax(pred)
    confidence = float(np.max(pred))

    return class_names[class_idx], confidence


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], uuid.uuid4().hex+file.filename)
            file.save(filepath)

            pred_class, conf = predict_image(filepath)

            prediction = pred_class
            confidence = round(conf * 100, 2)
            image_path = filepath

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )


if __name__ == "__main__":
    app.run(debug=False)
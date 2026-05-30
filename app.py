# app.py

import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CLASS_LABELS = ['Normal', 'Pneumonia', 'COVID', 'Tuberculosis']

# ---------------- INIT APP ----------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
MODEL_PATH = 'models/respiratory_disease_model.h5'
model = load_model(MODEL_PATH)

# ---------------- HELPER FUNCTIONS ----------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    confidence = round(preds[0][class_idx] * 100, 2)
    label = CLASS_LABELS[class_idx]
    return label, confidence

# ---------------- ROUTES ----------------
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    confidence = None
    image_path = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part"
        file = request.files['image']
        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = file.filename
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            # Make prediction
            prediction, confidence = predict_disease(save_path)
            image_path = save_path

    return render_template('index.html', prediction=prediction, confidence=confidence, image_path=image_path)

# ---------------- RUN APP ----------------
if __name__ == '__main__':
    app.run(debug=True)

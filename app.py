from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('fruit_model.h5', compile=False)

# Set class labels in the same order as training
class_labels = [
    "Apple_Blotch", "Apple_Healthy", "Apple_Rot", "Apple_Scab",
    "Guava_Anthracnose", "Guava_Fruitfly", "Guava_Healthy",
    "Mango_Alternaria", "Mango_Anthracnose", "Mango_Black Mould Rot (Aspergillus)", 
    "Mango_Healthy", "Mango_Stem and Rot (Lasiodiplodia)",
    "Pomegranate_Alternaria", "Pomegranate_Anthracnose", "Pomegranate_Bacterial_Blight",
    "Pomegranate_Cercospora", "Pomegranate_Healthy"
]

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image uploaded.'

    file = request.files['image']
    if file.filename == '':
        return 'No file selected.'

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_label = class_labels[predicted_index]
    confidence = float(np.max(prediction)) * 100

    # Pass predictions as JSON serializable data
    labels = class_labels
    probabilities = [round(float(p) * 100, 2) for p in prediction]

    return render_template(
        'result.html',
        label=predicted_label,
        confidence=round(confidence, 2),
        image_path=filepath,
        labels=labels,
        probabilities=probabilities
    )


if __name__ == '__main__':
    app.run(debug=True)

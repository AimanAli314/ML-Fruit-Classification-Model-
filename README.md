# ML Fruit Classification Model

This repository contains a simple web application and a training script for classifying fruit leaf images into various disease categories using a MobileNetV2 based convolutional neural network.

## Directory overview
- **app.py** – Flask application that loads a pre-trained model (`fruit_model.h5`) and exposes a web form for uploading images. Predictions are displayed with a bar chart showing the probability for each class.
- **train_model.py** – Script to train a model using images located under `dataset/train` (not included in this repository). The script performs data augmentation and fine-tunes MobileNetV2.
- **templates/** – HTML templates (`index.html`, `result.html`) for the web interface.
- **static/uploads/** – Folder where uploaded images are stored during prediction.
- **dataset/test/** – Example dataset of test images organised into class folders. Useful for validating the model.
- **fruit_model.h5** – Pretrained model weights (~9 MB).

## Requirements
- Python 3.9 or later
- Flask
- TensorFlow (tested with TensorFlow 2)
- Additional packages from `tensorflow.keras` (installed automatically with TensorFlow)

You can install the necessary dependencies with:

```bash
pip install flask tensorflow
```

## Training a new model
1. Place your training images in `dataset/train` using one subdirectory per class. The names of these folders must match the class labels defined in `app.py`.
2. Optionally place validation images in `dataset/test` following the same structure.
3. Run the training script:

```bash
python train_model.py
```

The script will create `fruit_model.h5` containing the trained weights.

## Running the web application
1. Ensure `fruit_model.h5` is present in the project root (use the provided file or train your own).
2. Launch the Flask app:

```bash
python app.py
```

3. Open your browser to `http://localhost:5000` and upload an image to see the predicted class and confidence score.

## Dataset
The provided `dataset/test` directory contains 1312 sample images spread across multiple fruit disease categories such as `Apple_Healthy`, `Mango_Alternaria`, `Pomegranate_Bacterial_Blight`, and others. These images can be used for quick evaluation of the model.


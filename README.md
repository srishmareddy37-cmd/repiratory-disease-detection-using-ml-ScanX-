# Respiratory Disease Detection Using CNN

## Overview

Respiratory Disease Detection is a deep learning-based web application that predicts respiratory diseases from chest X-ray images. The system uses a Convolutional Neural Network (CNN) model trained on medical imaging data to classify images into one of four categories: Normal, Pneumonia, COVID-19, and Tuberculosis. The application provides a simple web interface where users can upload chest X-ray images and receive instant predictions with confidence scores.

## Features

* Upload chest X-ray images in JPG, JPEG, or PNG format.
* Automated image preprocessing and normalization.
* CNN-based disease classification.
* Prediction confidence score display.
* User-friendly Flask web interface.
* Support for multiple respiratory disease categories.

## Technologies Used

* Python
* Flask
* TensorFlow/Keras
* NumPy
* HTML/CSS
* CNN (Convolutional Neural Network)

## Project Structure

```
project/
│
├── app.py
├── models/
│   └── respiratory_disease_model.h5
├── static/
│   └── uploads/
├── templates/
│   └── index.html
└── README.md
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd respiratory-disease-detection
```

2. Install required packages:

```bash
pip install flask tensorflow numpy pillow
```

3. Place the trained model file in the models folder:

```text
models/respiratory_disease_model.h5
```

4. Run the application:

```bash
python app.py
```

5. Open your browser and navigate to:

```text
http://127.0.0.1:5000
```

## Working

The user uploads a chest X-ray image through the web interface. The image is resized to 224×224 pixels, normalized, and passed to the trained CNN model. The model analyzes image features and predicts the most likely respiratory disease category. The predicted class and confidence percentage are then displayed on the screen.

## Disease Classes

* Normal
* Pneumonia
* COVID-19
* Tuberculosis

## Future Enhancements

* Support for additional respiratory diseases.
* Integration with cloud deployment platforms.
* Improved model accuracy using larger datasets.
* Patient report generation and storage.
* Real-time medical assistance features.

## Conclusion

This project demonstrates the application of deep learning in healthcare by providing an efficient and accurate system for respiratory disease detection. The CNN model assists in early diagnosis, reduces manual effort, and supports healthcare professionals in clinical decision-making.

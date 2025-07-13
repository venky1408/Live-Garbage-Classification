# Garbage Classification using Deep Learning

## Overview
In this project we have two models that perform two different operations.
    -> One model is used to classify the type of garbage into 12 categories (paper, metal, cardboard etc)
    -> The other model is used to classify the type of garbage into 2 categories (Organic, Recyclable)
This project aims to automatically classify garbage images into categories using a Convolutional Neural Network (CNN) and Transfer Learning. The model is trained on a labeled image dataset to assist in smart waste management and recycling efforts.

## Files
-> ModelTest.ipynb : This is the main file where we test our models for classifying orgainc and recyclable waste
-> FinalProject(individual).ipynb : This file is used to train the model for classifying 12, but I just made the model using metal and paper as sample dataset

## Project Structure
```
garbage-classification/
├── data/
│   ├── train/
│   └── test/
├── data(12 categories)/
│   └── Sample Dataset(Metal&paper)/
├── Scripts(12 Categories)/
│   ├── Final Project (Individual classification).py
├── Scripts(Organic and Recyclable)/
│   ├── ModelTest.ipynb
|   └── Training.ipynb
├── Model(Organic and Recyclable)/
│   └── garbage_classifier.h5
├── results/
│   └── Results(classification(metal/paper)).png
└── README.md
```

## Features
- Image preprocessing and data augmentation
- Model training using CNN and/or Transfer Learning (e.g., MobileNetV2, ResNet50)
- Performance evaluation using metrics like accuracy, precision, recall, F1-score
- Confusion matrix visualization
- Real-time prediction via web app or webcam (optional)

## Model Summary
- **Input:** RGB image of size 224x224
- **Output:** Garbage class label (e.g., Metal, Paper)
- **Architecture:** MobileNetV2 (pretrained) + Custom Dense layers
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam

## Model Summary
- **Input:** RGB image of size 224x224
- **Output:** Garbage class label -  Organic, Recyclable
- **Architecture:** MobileNetV2 (pretrained) + Custom Dense layers
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam

##  Dataset
- Dataset contains organic and recyclable waste images used for this project can be found in this repository.
- Sample dataset used for classifying the 12 categories is there in the kaggle dataset.
- Use the below code to load the dataset. Note : - You should set up the API key properly before running the below code.
    CODE : - 
        import kagglehub
        # Download latest version
        path = kagglehub.dataset_download("mostafaabla/garbage-classification")
        print("Path to dataset files:", path)

## Installation
```bash
git clone https://github.com/venky1408/garbage-classification.git
cd garbage-classification
```

## Results (Categories(12) - model)
| Metric              | Value      |
|---------------------|------------|
| Test Accuracy       | 97.87%     |

## Results (Categories(Organic & Recyclable) - model)
| Metric              | Value      |
|---------------------|------------|
| Test Accuracy       | 93.12%     |


> Check `results/` for output images..

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn
- Streamlit (for Web Interface)

## Future Work
- Expand dataset to include more classes
- Improve real-time webcam inference accuracy
- Integrate with smart dustbins for IoT-based deployment
- Explore transfer learning with other architectures (e.g., ResNet50, InceptionV3)




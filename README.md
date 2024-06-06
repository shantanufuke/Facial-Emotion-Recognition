# Facial-Emotion-Recognition

## Overview

This project implements a Convolutional Neural Network (CNN) for recognizing various traffic signs from images. The dataset consists of over 50,000 images of traffic signs, organized into 43 different classes for classification purposes. The dataset is divided into training and testing sets.

## Features

Traffic Sign Dataset: Contains images of 43 different classes of traffic signs.
Custom CNN Model: A deep learning model built using TensorFlow and Keras for classifying traffic signs.
Data Processing and Augmentation: Utilizes the PIL library for image processing and augmentation techniques to enhance model performance.
Model Training and Validation: Includes scripts for training and validating the model, ensuring high accuracy and robustness.
Model Testing: Provides functionality for testing the trained model on a separate test dataset.
User Interface: Implements a user-friendly GUI using Gradio for easy interaction and visualization of classification results.

## Dataset

The image dataset consists of more than 50,000 pictures of various traffic signs (speed limit, crossing, traffic signals, etc.). The dataset is organized into two folders, train and test, where the train folder contains subfolders representing different classes, and each subfolder contains multiple images.

You can download the Kaggle dataset for this project from the below link:
[Kaggle Traffic Sign Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

## Workflow

## Dataset Exploration
Explore the dataset containing traffic sign images organized into subfolders representing different classes.
Utilize the os module to iterate through the images and their labels, and the PIL library to open and process image data.
Store image data and labels into lists and convert them into NumPy arrays for model training.
CNN Model Building

## Implement a custom CNN architecture using TensorFlow's Keras API.
Define convolutional layers with ReLU activation, max-pooling, and dropout layers to prevent overfitting.
Configure convolutional layers with varying filter sizes and dropout rates to capture hierarchical features from input images.
Add dense layers with ReLU activation and a final output layer with softmax activation to yield class probabilities.
Construct the model to process input images of specified shape, facilitating effective image classification.

## Model Training and Validation
Train the model using the model.fit() method, specifying batch size and epochs to optimize performance.
Evaluate model performance on training and validation sets, monitoring accuracy and stability to ensure effective learning.
Model Testing
Utilize a separate test dataset containing image paths and corresponding class labels.
Resize test images to match the model input dimensions and generate a NumPy array with image data.
Predict class labels using the trained model and evaluate predictions using accuracy_score from sklearn.metrics.

## Conclusion

We successfully developed a CNN model to classify traffic signs with 95% accuracy, leveraging a large dataset and implementing a custom architecture. The intuitive GUI built with Gradio enhances user interaction and understanding of the classification process, providing a seamless experience for traffic sign identification.

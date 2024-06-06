# Human Face Emotion Recognition Using CNN

## Overview

This project implements a Convolutional Neural Network (CNN) for recognizing human emotions from facial images. The dataset consists of images categorized into several emotion classes. The dataset is divided into training and testing sets for robust model evaluation.

## Features

Emotion Dataset: Contains images labeled with different human emotions.
Custom CNN Model: A deep learning model built using TensorFlow and Keras for classifying human emotions.
Data Processing and Augmentation: Utilizes the PIL library for image processing and augmentation techniques to enhance model performance.
Model Training and Validation: Includes scripts for training and validating the model, ensuring high accuracy and robustness.
Model Testing: Provides functionality for testing the trained model on a separate test dataset.
User Interface: Implements a user-friendly GUI using Gradio for easy interaction and visualization of classification results.

## Dataset
The image dataset consists of various pictures of human faces expressing different emotions. The dataset is organized into two folders, train and test, where the train folder contains subfolders representing different emotion classes, and each subfolder contains multiple images.

You can download the Kaggle dataset for this project from the below link:
[Kaggle Emotion Dataset](https://www.kaggle.com/datasets/msambare/fer2013/data)

## Workflow

## Dataset Exploration

Explore the dataset containing facial images organized into subfolders representing different emotion classes.
Utilize the os module to iterate through the images and their labels, and the PIL library to open and process image data.
Store image data and labels into lists and convert them into NumPy arrays for model training.

## CNN Model Building

Implement a custom CNN architecture using TensorFlow's Keras API.
Define convolutional layers with ReLU activation, max-pooling, and dropout layers to prevent overfitting.
Configure convolutional layers with varying filter sizes and dropout rates to capture hierarchical features from input images.
Add dense layers with ReLU activation and a final output layer with softmax activation to yield class probabilities.
Construct the model to process input images of specified shape, facilitating effective emotion classification.

## Model Training and Validation
Train the model using the model.fit() method, specifying batch size and epochs to optimize performance.
Evaluate model performance on training and validation sets, monitoring accuracy and stability to ensure effective learning.
Model Testing
Utilize a separate test dataset containing image paths and corresponding class labels.
Resize test images to match the model input dimensions and generate a NumPy array with image data.
Predict class labels using the trained model and evaluate predictions using accuracy_score from sklearn.metrics.

## Conclusion
We successfully developed a CNN model to classify human emotions from facial images with high accuracy, leveraging a large dataset and implementing a custom architecture. The intuitive GUI built with Gradio enhances user interaction and understanding of the classification process, providing a seamless experience for emotion identification.

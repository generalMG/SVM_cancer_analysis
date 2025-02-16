# Breast Cancer SVM Classification Visualizer

A Python script that demonstrates Support Vector Machine (SVM) classification on the breast cancer dataset using scikit-learn. The script visualizes the decision boundary of the SVM classifier using the first two features of the dataset.

## Requirements

- Python 3.10 or above
- scikit-learn
- matplotlib

## Features

- Loads the breast cancer dataset from scikit-learn
- Implements SVM classification with RBF kernel
- Visualizes the decision boundary and data points
- Uses only the first two features for 2D visualization

## Usage

Simply run the script to see the visualization. The plot will show:
- Data points colored by their class (malignant/benign)
- Decision boundary colored using a spectral colormap
- Feature names on the x and y axes

## Parameters

- SVM kernel: RBF (Radial Basis Function)
- Gamma: 0.5 (kernel coefficient)
- C: 1.0 (regularization parameter)

## Output

The script generates a plot showing how the SVM classifier separates the two classes of breast cancer data (malignant and benign) based on the first two features. The decision boundary is displayed with a spectral colormap, and data points are scattered with black edges for better visibility.

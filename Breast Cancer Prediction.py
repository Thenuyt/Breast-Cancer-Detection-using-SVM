"""
Breast Cancer Detection using SVM (Support Vector Machine)

This script demonstrates how to use an SVM classifier to predict breast cancer
using the Breast Cancer Wisconsin Diagnostic dataset from scikit-learn.

Author: Thenmozhi M
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load breast cancer dataset
data = load_breast_cancer()

# Features
X = data.data
# Labels (0: malignant, 1: benign)
y = data.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Support Vector Machine (SVM) classifier
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

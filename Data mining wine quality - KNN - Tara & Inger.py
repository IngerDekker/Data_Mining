# -*- coding: utf-8 -*-
"""

@author: Inger & Tara
"""
#KNN Model

# Import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Importing the date set
wine_data = pd.read_csv('WineQt.csv')
#site for the dataset: 
#https://www.kaggle.com/datasets/yasserh/wine-quality-dataset

#Look at the information in the data
wine_data.info

#Info on the data
wine_data.shape

# Check for missing values
print(wine_data.isnull().sum())
#Outcome: there are no missing values 

# Split the data into features (X) and target (y) for training set
X = wine_data.drop("quality", axis=1)
y = wine_data["quality"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (scaling) - Normalize the data 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Make the KNN-model with K = 3 and training the model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

# We are going to make our first predictions 
y_pred = knn.predict(X_test)
print(y_pred)
#With print we look at the predictions that are made with the testset 

# Evaluate 
## Calculate accuracy: correct predictions 
## Correct prediction samples / total samples
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
# Accuracy = 53% that it is classified right 
# Accuracy is not enough the see how good the model works so you can make a classification report

# Generate a classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

#For class 3, the model did not correctly predict anything because no examples for this class were given during the training of the model
#For class 4, the model did not predict anything correctly with 6 examples of these classes (conclusion: class 4 cannot be predicted well due to relatively too few examples)
#For class 5, the scores are acceptable with relatively many examples
#For class 6, the scores are acceptable with relatively many examples
#For class 7, the scores are moderate (conclusion: the model has difficulty correctly classifying class 7)
#For class 8, the model did not correctly predict anything based on 2 examples

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualize confusion matrix using Seaborn heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

## Making graphs
# List to save the results
k_values = []
accuracy_values = []

# Try different values of n_neighbors
for k in range(1, 41):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    k_values.append(k)
    accuracy_values.append(accuracy)

# Create a graph of accuracy versus number of neighbors
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy_values, marker='o', linestyle='--')
plt.title('Accuracy vs. Number of Neighbors (K)')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# see with other KNN
# Make the KNN-model with K = 35
knn = KNeighborsClassifier(n_neighbors=35)
knn.fit(X_train,y_train)  
# The model is trained 
# We are going to make predictions 

# Make predictions 
y_pred = knn.predict(X_test)
print(y_pred)

# Evaluate 
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
# Accuracy = 63% that it is classified right

# Generate a classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

# With n_neighbor = 35, the accuracy score has improved to about 63%
# De values have also improved reasonably but no disproportionate differences

# Overall conclusion: increasing n-neugbors to 35 allowed the model to achieve better overall presets of accuracy
# The model: with this sample still difficulty classifying some class due to lack of training examples.

## Model retrain with a different random_state to look at a different training set
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X, y, test_size=0.2, random_state=123)  # Verander 123 in een willekeurig getal naar keuze
 
# Normalise the new trainingset
X_train_new = scaler.fit_transform(X_train_new)
X_test_new = scaler.transform(X_test_new)

# Build and train the KNN model with the new split
knn_new = KNeighborsClassifier(n_neighbors=35)
knn_new.fit(X_train_new, y_train_new)

# Make predictions on the new test data
y_pred_new = knn_new.predict(X_test_new)
print(y_pred_new)

# Evaluate the performance of the model on the new split
accuracy_new = accuracy_score(y_test_new, y_pred_new)
print(accuracy_new)

# Classification report for the new split
class_report_new = classification_report(y_test_new, y_pred_new)
print(class_report_new)

# Other n_neigbors 
# Build and train the KNN model with the new split
knn_new = KNeighborsClassifier(n_neighbors=3)
knn_new.fit(X_train_new, y_train_new)

# Make predictions on the new test data
y_pred_new = knn_new.predict(X_test_new)
print(y_pred_new)

# Classification report for the new split
class_report_new = classification_report(y_test_new, y_pred_new)
print(class_report_new)

# Conclusion: another testsample does not improve the results. So the accuracy of 63% is the highest one that we found.

# See wat N is the best value 
from sklearn.cluster import KMeans

k_values = range(1, 41)
scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    scores.append(kmeans.inertia_)

plt.plot(k_values, scores)
plt.xlabel("number of k clusters")
plt.ylabel("inertia score")
plt.grid(True)
plt.show()

# In the first graph you will see that the highest accuracy is 65%

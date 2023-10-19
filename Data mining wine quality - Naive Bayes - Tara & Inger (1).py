# -*- coding: utf-8 -*-
"""

@author: Inger & Tara
"""
# Import libraries 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Import the data 
wine_data = pd.read_csv("wineQT.csv")
#site for the dataset: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset

# Preparing data
print(wine_data.head())

# Split the data
X = wine_data.drop("quality", axis=1)
y = wine_data["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train model
nb_classifier = GaussianNB()

#Train model
nb_classifier.fit(X_train, y_train)

# Evaluate model
## Predictions 
y_pred = nb_classifier.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
## Accuracy = 60% which is reasenable but not great

# Evaluation report 
report = classification_report(y_test, y_pred)
print(report)

## Difficulty with classes 3, 4 and 8 precision and recall low 
## Class 5 and 6 better performance and also more values

##Hyperparameter tuning

# Define the hyperparameters and their possible values
param_grid = {
    "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}

# Import new libraries for model
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=nb_classifier, param_grid=param_grid, cv=5, scoring="accuracy")

# Perform grid search on the training data
grid_search.fit(X_train, y_train)

# Check out the best hyperparameters
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

# Evaluate the model with the best hyperparameters on the test set
best_nb_classifier = grid_search.best_estimator_
y_pred = best_nb_classifier.predict(X_test)

# View accuracy
accuracy = accuracy_score(y_test, y_pred)
print("New Test Set Features:", accuracy)
# This did not improve the model as the outcome is not very different from the previous outcomes
# Accuracy is now 63,3%

#Confusion Matrix: A graphical representation of the confusion matrix can help visualize model performance for each class
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(3, 9), yticklabels=range(3, 9))
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.title("Confusion Matrix")
plt.show()

# This shows how accuracy changes as you add more data to the model.
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(best_nb_classifier, X_train, y_train, cv=5)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training accuracy')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Test accuracy')
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.show()

# Conclusions:
# K-Nearest Neigbhors model (KNN)
# The KNN model achieved an accuracy of approximately 65%, indicating that it correctly predicted wine quality for about 65% of the test samples. 
# The F1-scores for different wine quality classes varied, with the highest F1-score of 0.69 for class 5 and the lowest F1-score of 0.00 for classes 4 and 8. 
# This suggests that the model performed well for some classes but struggled to make correct predictions for others due to a lack of training examples. 
# The average F1 score was 0.61 which indicates the average performance across all classes. 

# Naive Bayes model 
# The NB model achieved an accuracy of approximately 61%, indicating that it correctly predicted wine quality for about 61% of the test samples. 
# The F1-scores for different wine quality classes also varied, with the highest F1-score of 0.70 for class 5 and the lowest F1-score of 0.00 for class 4. 
# Similar to the KNN model, the NB has problems making predictions in certain classes 
  
# Overal conclusion
# •	The KNN model has a slightly higher accuracy (65%) compared to the NB model (61%).
# •	Both models have problems making predictions for classes 4 and 8 due to little to no training data
# •	Both models achieved a moderate overal accuracy but no outstanding performance. 

# We can conclude that the dataset is imbalanced. Classes 4, 7 and 8 have less data available meaning that these classes have fewer examples to train the model. 
# This makes it difficult for the model to learn meaningful patterns and make correct predictions. 
# Models like KNN and NB rely on learning patterns from training data so these models may not be the right models to use for this specific data set given the fact that the data in this set is not distributed evenly.

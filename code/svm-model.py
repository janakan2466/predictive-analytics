# SVM Model of the Bank Customer Churn dataset
# Janakan Sivaloganthan
# 20/11/2021

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv('../datasets/dataset.csv')

# Separate the target variable
target = df['Exited']
df.drop(['Surname', 'Exited', 'CustomerId', 'RowNumber'], axis='columns', inplace=True)

# Encode categorical variables
le_Geography = LabelEncoder()
le_Gender = LabelEncoder()
df['Geography_n'] = le_Geography.fit_transform(df['Geography'])
df['Gender_n'] = le_Gender.fit_transform(df['Gender'])
df_n = df.drop(['Geography', 'Gender'], axis='columns')  # Drop the string variables that are not encoded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_n, target, test_size=0.3, random_state=100)

# Train the Support Vector Machine (SVM) model
model = SVC()
# Default parameters: SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, gamma='auto', kernel='rbf' max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
model.fit(X_train, y_train)

# Calculate and print the accuracy of the model
accuracy = model.score(X_test, y_test)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# You can also make predictions using the trained model:
# prediction = model.predict([[4.8, 3.0, 1.5, 0.3]])

# Tune Parameters:
# You can experiment with different parameters (e.g., C, gamma, kernel) to improve the model's performance.
# Here's an example of tuning the regularization parameter C:

# model_C = SVC(C=1)
# model_C.fit(X_train, y_train)
# print("Accuracy with C=1: {:.2f}%".format(model_C.score(X_test, y_test) * 100))

# model_C = SVC(C=10)
# model_C.fit(X_train, y_train)
# print("Accuracy with C=10: {:.2f}%".format(model_C.score(X_test, y_test) * 100))

# You can similarly tune other parameters like gamma and kernel to find the best combination for your problem.

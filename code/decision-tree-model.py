# Machine Learning Assignment
# Decision Tree Model of the Bank Customer Churn dataset
# Janakan Sivaloganthan
# 20/11/2021

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

df = pd.read_csv('../datasets/dataset.csv')
print(df.head())

target = df['Exited']
df.drop(['Surname', 'Exited', 'CustomerId', 'RowNumber'], axis='columns', inplace=True)
print(df.info())

le_geography = LabelEncoder()
le_gender = LabelEncoder()

df['Geography_n'] = le_geography.fit_transform(df['Geography'])
df['Gender_n'] = le_gender.fit_transform(df['Gender'])

df_n = df.drop(['Geography', 'Gender'], axis='columns')
print(df_n.head())

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(df_n, target, test_size=0.3, random_state=100)

# Create and train the Decision Tree Classifier
model = tree.DecisionTreeClassifier()
model = model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

accuracy = model.score(df_n, target) * 100
print(f"\nAccuracy = {accuracy:.2f}%")

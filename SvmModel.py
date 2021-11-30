#Janakan Sivaloganthan 20/11/2021 Machine Learning Assignment
#SVM Model of the Bank Customer Churn dataset

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


data_file= pd.read_csv('dataset.csv')
#print(data_file)

print(data_file.dtypes)

#data_file["CreditScore"] = pd.to_numeric(data_file["CreditScore"], errors='raise', downcast="float")

feature_cols = ['CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
X = data_file[feature_cols] # Features
y = data_file.Exited # Target variable

# Split dataset into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,Y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Evaluate predictions


#print(df)
#Janakan Sivaloganthan 20/11/2021 Machine Learning Assignment
#SVM Model of the Bank Customer Churn dataset

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
#matplotlib inline

df= pd.read_csv('dataset.csv')

target= df['Exited']
df.drop(['Surname','Exited', 'CustomerId', 'RowNumber'], axis= 'columns', inplace= True)
#print(df.info())

le_Geography= LabelEncoder()
le_Gender= LabelEncoder()

df['Geography_n']= le_Geography.fit_transform(df['Geography'])
df['Gender_n']= le_Gender.fit_transform(df['Gender'])

df_n = df.drop(['Geography', 'Gender'], axis='columns') #drops the string variables that are not encoded
#print(df_n.head())

X_train, X_test, y_train, y_test = train_test_split(df_n, target, test_size=0.3, random_state=100) #df_n represents inputs while target represents output

#Train Using Support Vector Machine (SVM)


print(len(X_train)) #6999 rows are presented for training

print(len(X_test)) #3000 rows are presented for test


from sklearn.svm import SVC
model = SVC()
#Default parameters: SVC(C=10, cache_size= 200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, gamma='auto', kernel='rbf' max_iter= -1, probability= False, random_state= None, shrinking= True, tol= 0.001, verbose= False)


print(model.fit(X_train, y_train))

print("accuracy: " +str(model.score(X_test, y_test)*100) +"%")

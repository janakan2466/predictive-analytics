#Janakan Sivaloganthan 20/11/2021 Machine Learning Assignment
#Decision Tree Model of the Bank Customer Churn dataset

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Decision Tree Classifier
from sklearn.model_selection import train_test_split #train_test_split function (split the training set)
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

df= pd.read_csv('dataset.csv')
print(df.head())

target= df['Exited']
df.drop(['Surname','Exited', 'CustomerId', 'RowNumber'], axis= 'columns', inplace= True)
print(df.info())

le_Geography= LabelEncoder()
le_Gender= LabelEncoder()

df['Geography_n']= le_Geography.fit_transform(df['Geography'])
df['Gender_n']= le_Gender.fit_transform(df['Gender'])

df_n = df.drop(['Geography', 'Gender'], axis='columns') #drops the string variables that are not encoded
print(df_n.head())

model= tree.DecisionTreeClassifier() #Create the classifier tree
model.fit(df_n, target) #train the model

print("\n\nAccuracy= " +str(model.score(df_n, target)*100) +"%") #prints the accuracy of the model

#model.predict(['Hello', 619, 24, 3, 0.00, 1, 1, 1, 11254.58, 0, 1])

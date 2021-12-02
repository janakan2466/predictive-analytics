#Janakan Sivaloganthan 20/11/2021 Machine Learning Assignment
#Script to display the characteristics of the Bank Customer Churn dataset

#import libraries
from matplotlib import colors
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('dataset.csv')
#print(df.head()) #prints the first 5 rows of df "datafile" 

# A distribution of the total distribution of geographical location
plt.figure(1)
df.Geography.value_counts().plot(kind='bar', title= "Total Distribution per Geographic Location", color=['green', 'yellow', 'blue'])

# Representation of the summation of target variable values as a pie chart
plt.figure(2)
df.Exited.value_counts().plot(kind='pie', colors=['lightgreen', 'red'], title= "Holistic Analysis of Churned Customers", autopct='%1.1f%%') 

# Representation of the summation of the customer gender values
plt.figure(3)
df.Gender.value_counts().plot(kind='bar',title= "Distribution per Gender", color=['blue','pink'])

# A distribution of the customers and their specified age values
plt.figure(4)
df["Age"].plot(kind = 'hist', title= "Age Statistic")

# Representation of the taget variable vs the customer geographical location
plt.figure(5)
sns.countplot(x= 'Exited', hue= 'Geography', data= df).set_title('Target vs Geography')

plt.show()

print("Present features")
print("-------------------")
print(df.columns())

print("Present null values")
print("-------------------")
print(df.isna().sum())

print("\nSum of target variable")
print("-------------------")
print(df.Exited.value_counts())

print("\nPresent the dimensions of the dataset")
print("-------------------")
print(df.shape)

# print("\nPresent additional information of the dataset")
# print("-------------------")
# print(df.describe())

print("\nPresent additional information such as datatype of dataset")
print("-------------------")
print(df.info())

print("\nPlot a sample graph to showcase the age distribution")
print("-------------------\n")


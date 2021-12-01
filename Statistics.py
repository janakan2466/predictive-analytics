#Janakan Sivaloganthan 20/11/2021 Machine Learning Assignment
#Script to display the characteristics of the Bank Customer Churn dataset

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df= pd.read_csv('dataset.csv')
#print(df.head()) #prints the first 5 rows of df "datafile" 

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

df["Age"].plot(kind = 'hist')

plt.show()

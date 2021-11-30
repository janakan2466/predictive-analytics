#Janakan Sivaloganthan 20/11/2021 Machine Learning Assignment
#Script to display the characteristics of the Bank Customer Churn dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_file= pd.read_csv('dataset.csv')
#print(data_file)

df= data_file

print(df.isna().sum().sum())
print(df.Exited.value_counts())
print(df.shape)
#print(df.describe())
print(df.info())
df["Age"].plot(kind = 'hist')

plt.show()

#print(df)
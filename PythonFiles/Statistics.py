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
sns.countplot(x= 'Geography', hue= 'Geography', data= df).set_title('Distribution of Customers over Geography')

# Representation of the summation of target variable values as a pie chart
plt.figure(2)
df.Exited.value_counts().plot(kind='pie', colors=['lightgreen', 'red'], title= "Holistic Analysis of Churned Customers", autopct='%1.1f%%') 

# Representation of the summation of the customer gender values
plt.figure(3)
sns.countplot(x= 'Exited', hue= 'Gender', data= df).set_title('Exited vs Gender')

# A distribution of the customers and their specified age values
plt.figure(4)
df["Age"].plot(kind = 'hist', title= "Age Statistic")

# Representation of the taget variable vs the customer geographical location
plt.figure(5)
sns.countplot(x= 'Exited', hue= 'Geography', data= df).set_title('Exited vs Geography')

# Representation of the taget variable vs if the customer has a credit card
plt.figure(6)
sns.countplot(x= 'Exited', hue= 'HasCrCard', data= df).set_title('Exited vs HasCrCard')

plt.figure(7)
sns.countplot(x= 'Exited', hue= 'IsActiveMember', data= df).set_title('Exited vs Active Member')

plt.show()


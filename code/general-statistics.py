# Script to display the characteristics of the Bank Customer Churn dataset
# Janakan Sivaloganthan
# 20/11/2021

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('../datasets/dataset.csv')

# Visualization 1: Total Distribution per Geographic Location
plt.figure(1)
df['Geography'].value_counts().plot(kind='bar', title="Total Distribution per Geographic Location", color=['green', 'yellow', 'blue'])

# Visualization 2: Holistic Analysis of Churned Customers
plt.figure(2)
df['Exited'].value_counts().plot(kind='pie', colors=['lightgreen', 'red'], title="Holistic Analysis of Churned Customers", autopct='%1.1f%%')

# Visualization 3: Distribution per Gender
plt.figure(3)
df['Gender'].value_counts().plot(kind='bar', title="Distribution per Gender", color=['blue', 'pink'])

# Visualization 4: Age Statistic
plt.figure(4)
df['Age'].plot(kind='hist', title="Age Statistic")

# Visualization 5: Target vs Geography
plt.figure(5)
sns.countplot(x='Exited', hue='Geography', data=df).set_title('Target vs Geography')

plt.show()

# Display dataset characteristics
print("Present features")
print("-------------------")
print(df.columns)

print("Present null values")
print("-------------------")
print(df.isna().sum())

print("\nSum of target variable")
print("-------------------")
print(df['Exited'].value_counts())

print("\nPresent the dimensions of the dataset")
print("-------------------")
print(df.shape)

print("\nPresent additional information such as datatype of dataset")
print("-------------------")
print(df.info())

print("\nPlot a sample graph to showcase the age distribution")
print("-------------------\n")

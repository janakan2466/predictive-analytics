import joblib
import datetime
import warnings
warnings.filterwarnings(action='ignore')

import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, TensorDataset

import seaborn as sns
import matplotlib.pyplot as plt



df= pd.read_csv('dataset.csv')
df.info()
df.isnull().sum()


# def describe_missing_data(data, show_all=False):
#     """
#     Definition:
#         Takes in raw DataFrame, and returns DataFrame with information about missing values (by variable)
#     Parameters:
#         - data (Pandas DataFrame): Pandas DataFrame of dataset
#         - show_all (bool): True to show variables without missing values; False otherwise. Default: False
#     Returns:
#         Returns Pandas DataFrame with information about missing values (by variable).
#         Columns include: ['Variable', 'NumMissingValues', 'PercentMissingValues', 'DataType']
#     """
#     # Datatype info
#     df_dtypes = pd.DataFrame()
#     data.dtypes.to_frame().reset_index()
#     df_dtypes['Variable'] = data.dtypes.to_frame().reset_index()['index']
#     df_dtypes['DataType'] = data.dtypes.to_frame().reset_index()[0]
    
#     # Missing value info
#     rename_dict = {
#         'index': 'Variable',
#         0: 'NumMissingValues',
#         '0': 'NumMissingValues'
#     }
#     df_missing_values = data.isnull().sum().to_frame().reset_index()
#     df_missing_values.rename(mapper=rename_dict, axis=1, inplace=True)
#     df_missing_values.sort_values(by='NumMissingValues', ascending=False, inplace=True)
#     df_missing_values = df_missing_values[df_missing_values['NumMissingValues'] > 0].reset_index(drop=True)
#     percent_missing_values = (df_missing_values['NumMissingValues'] / len(data)).mul(100).apply(round, args=[3])
#     df_missing_values['PercentMissingValues'] = percent_missing_values
    
#     # Merge everything
#     df_description = pd.merge(left=df_missing_values, right=df_dtypes, on='Variable', how='outer')
#     df_description.fillna(value=0, inplace=True)
#     if not show_all:
#         df_description = df_description[df_description['NumMissingValues'] > 0]
#     df_description['NumMissingValues'] = df_description['NumMissingValues'].astype(int)
#     return df_description


# def _fillna_with_random_choice(string, unique_choices):
#     """
#     Definition:
#         Helper function for 'fillna_with_random_choice'.
#         Returns random choice from `unique_choices` (list) if string == 'NULL', else returns same string.
#     """
#     if string == 'NULL':
#         return random.choice(unique_choices)
#     return string


# def fillna_with_random_choice(data, subset=[]):
#     """
#     Definition:
#         Fills missing values of categorical features by selecting random value from said feature
#     Parameters:
#         - data (Pandas DataFrame): Pandas DataFrame of dataset
#         - subset (list): Subset of categorical variables for which you want to randomly fillna. Default: []
#     Returns:
#         Pandas DataFrame of given dataset, with categorical variables randomly filled (by picking random value from same variable)
#     """
#     categorical_variables = data.select_dtypes(include='object').columns.tolist()
#     if subset:
#         categorical_variables = list(set(categorical_variables).intersection(set(subset)))

#     # fillna by picking random value from set of unique values by column (for all categorical variables)
#     for cv in categorical_variables:
#         if data[cv].isnull().sum() > 0:
#             data[cv].fillna(value='NULL', inplace=True) # Fills NaNs with 'NULL' (string)
#             unique_choices = data[cv].unique().tolist() # Get list of unique choices for column
#             if 'NULL' in unique_choices:
#                 unique_choices.remove('NULL') # Remove 'NULL' from list of unique choices
#             data[cv] = data[cv].apply(func=_fillna_with_random_choice, unique_choices=unique_choices)
#     return data


# def get_percentage_missing_values(data):
#     """ Gets percentage (float) of all missing values in Pandas DataFrame """
#     return float(round(data.isnull().sum().sum() * 100 / len(data), 3))


def plot_categorical_valuecount_barh(data, unique_value_limit=15, subset=[]):
    """
    Definition:
        Plots barh charts of value counts of all categorical variables in dataset
    Parameters:
        - data (Pandas DataFrame): Pandas DataFrame of dataset
        - unique_value_limit (int): Threshold for maximum number of unique categorical variables you want to consider plotting.
          If set to None, no threshold is set (NO PLOTS WILL BE CREATED). Default: 15
        - subset (list): Subset of categorical variables you want to consider. Default: []
    """
    categorical_variables = data.select_dtypes(include='object').columns.tolist()
    if subset:
        categorical_variables = list(set(categorical_variables).intersection(set(subset)))
    
    for cv in categorical_variables:
        value_count_series = data[cv].value_counts()
        # Will only plot if number of unique values of the categorical variable is <= threshold
        if type(unique_value_limit) is int and len(value_count_series) <= unique_value_limit:
            plt.figure(figsize=(12, 5))
            value_count_series.plot(kind='barh')
            plt.title(cv, fontsize=20)
            plt.show()
        # Else, will print details of value counts
        else:
            print(
                f"\n{cv}: \t\t(has {len(value_count_series)} unique categories)",
                f"\n{value_count_series.head(10)}"
            )
        print("\n\n")
    return None


def plot_numerical_distributions(data, describe=True, subset=[]):
    """
    Definition:
        Plots histograms of all numerical variables in dataset
    Parameters:
        - data (Pandas DataFrame): Pandas DataFrame of dataset
        - describe (bool): True if you want to see additional descriptive stats. Default: True
        - subset (list): Subset of numerical variables you want to consider. Default: []
    """
    numerical_variables = data.select_dtypes(exclude=['object', 'bool']).columns.tolist()
    # if subset:
    #     numerical_variables = list(set(numerical_variables).intersection(set(subset)))
    
    # for nv in numerical_variables:
    #     plt.figure(figsize=(12, 5))
    #     data[nv].plot(kind='hist', bins=6, label=None)
    #     plt.title(s=f"Distribution - {nv}", fontsize=20, label='values')
    #     plt.xlabel(s=nv, fontsize=12)
    #     if describe:
    #         plt.axvline(x=data[nv].median(), color='red', label='Median')
    #         plt.axvline(x=data[nv].mean(), color='green', label='Mean')
    #         plt.legend(loc='best', fontsize=12)
    #     plt.show()
    # return None

# print(df.head())

# plot_categorical_valuecount_barh(data=df, unique_value_limit=15, subset=['Geography', 'Gender'])

# numerical_variables = df.select_dtypes(exclude='object')\
#                         .drop(labels=['RowNumber', 'CustomerId', 'HasCrCard', 'IsActiveMember', 'NumOfProducts', 'Exited'],
#                               axis=1)\
#                         .columns.tolist()
# plot_numerical_distributions(data=df, describe=True, subset=numerical_variables)

# Clean up
df.drop(labels=['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Encoding categorical variables
df = pd.get_dummies(data=df, drop_first=True)

df.head()

df.shape

# Create holdout set
holdout_size = 1000
random_indices = []

while True:
    random_index = random.randint(0, len(df)-1)
    if random_index not in random_indices:
        random_indices.append(random_index)
    if len(random_indices) == holdout_size:
        break

df_regular = df.drop(labels=random_indices, axis=0).reset_index(drop=True).copy()
df_holdout = df.iloc[random_indices].reset_index(drop=True).copy()

print(f"Shape of regular data: {df_regular.shape}")
print(f"Shape of holdout data: {df_holdout.shape}")

# Contains data used to train the model
df = df_regular.copy()

# Train-test split
X = df.drop(labels=['Exited'], axis=1)
y = df['Exited'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Feature scaling
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(len(y_train), 1)
y_test = y_test.view(len(y_test), 1)


class Network(nn.Module):
    
    def __init__(self, num_input_features):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(num_input_features, 15)
        self.linear2 = nn.Linear(15, 8)
        self.linear3 = nn.Linear(8, 1)
    
    def forward(self, xb):
        prediction = torch.sigmoid(input=self.linear1(xb))
        prediction = torch.sigmoid(input=self.linear2(prediction))
        prediction = torch.sigmoid(input=self.linear3(prediction))
        return prediction


lr = 0.01
model = Network(num_input_features=11)
print(f"Model Architecture:\n{model}")
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

nb_epochs = 1200
print_offset = 200

df_tracker = pd.DataFrame()
print("\nTraining the model...")
for epoch in range(1, nb_epochs+1):
    y_pred = model(X_train)
    loss = criterion(input=y_pred, target=y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % print_offset == 0:
        print(f"\nEpoch {epoch} \t Loss: {round(loss.item(), 4)}")
    
    # Print test-accuracy after certain number of epochs
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_class = y_pred.round()
        accuracy = y_pred_class.eq(y_test).sum() / float(len(y_test))
        if epoch % print_offset == 0:
            print(f"Accuracy (on test-set): {round(accuracy.item(), 4)}")
    
    df_temp = pd.DataFrame(data={
        'Epoch': epoch,
        'Loss': round(loss.item(), 4),
        'Accuracy': round(accuracy.item(), 4)
    }, index=[0])
    df_tracker = pd.concat(objs=[df_tracker, df_temp], ignore_index=True, sort=False)

print(f"\nFinal Accuracy (on test-set): {round(accuracy.item(), 4)}")
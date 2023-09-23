# Neural Network Model of the Bank Customer Churn dataset
# Janakan Sivaloganthan
# 20/11/2021

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, TensorDataset
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('../datasets/dataset.csv')

df.drop(labels=['Surname', 'CustomerId', 'RowNumber'], axis=1, inplace=True)

df = pd.get_dummies(data=df, drop_first=True)

print("Dimensions of the dataset:", df.shape)

test_size = 3000
random_index_calls = []

# Create a random array of indexes for the test input
while len(random_index_calls) < test_size:
    random_index = random.randint(0, len(df) - 1)
    if random_index not in random_index_calls:
        random_index_calls.append(random_index)

df_regular = df.drop(labels=random_index_calls, axis=0).reset_index(drop=True).copy()
df_test = df.iloc[random_index_calls].reset_index(drop=True).copy()

print("Dimensions of training data:", df_regular.shape)
print("Dimensions of test data:", df_test.shape)

df = df_regular.copy()

X = df.drop(labels=['Exited'], axis=1)
y = df['Exited'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Feature transformation
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
    def __init__(self, input_feature_amount):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(input_feature_amount, 15)
        self.linear2 = nn.Linear(15, 8)
        self.linear3 = nn.Linear(8, 1)

    def forward(self, in_var):
        prediction = torch.relu(input=self.linear1(in_var))
        prediction = torch.relu(input=self.linear2(prediction))
        prediction = torch.relu(input=self.linear3(prediction))
        return prediction

# Create the neural network model
model = Network(input_feature_amount=11)
print("Neural Network Diagram:", str(model))

# Define training parameters
learning_rate = 0.01
criterion_measure = nn.BCELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

# Training loop
num_epochs = 3200
offset_interval = 100
df_track = pd.DataFrame()

print("\nTraining")
print("----------")

for epoch in range(1, num_epochs + 1):
    y_pred = model(X_train)
    loss = criterion_measure(input=y_pred, target=y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % offset_interval == 0:
        print("Epochs =", epoch)
        print("Loss =", round((loss.item()) * 100, 2), "%")

    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_class = y_pred.round()
        accuracy = y_pred_class.eq(y_test).sum() / float(len(y_test))
        if epoch % offset_interval == 0:
            print("Accuracy:", round((accuracy.item() * 100), 2), "%\n")

    df_temp = pd.DataFrame(data={'Epochs': epoch, 'Loss': round(loss.item(), 5), 'Accuracy': round(accuracy.item(), 5)}, index=[0])
    df_track = pd.concat(objs=[df_track, df_temp], ignore_index=True, sort=False)

print("\nOverall Accuracy:", round((accuracy.item()) * 100, 2), "%\n")

# Plot the correlation of the number of epochs vs loss function
plt.figure(1)
plt.plot(df_track['Epochs'], df_track['Loss'], color='red', label='Loss')
plt.title("Epoch vs Loss", fontsize=20)
plt.xlabel("Number of Epoch")
plt.ylabel("Loss Value")

# Plot the correlation of the number of epochs vs accuracy
plt.figure(2)
plt.plot(df_track['Epochs'], df_track['Accuracy'], color='green', label='Loss')
plt.title("Epoch vs Accuracy", fontsize=20)
plt.xlabel("Number of Epoch")
plt.ylabel("Accuracy Value")
plt.show()

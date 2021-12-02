#Janakan Sivaloganthan 20/11/2021 Machine Learning Assignment
#Neural Network Model of the Bank Customer Churn dataset

#imported library
import torch # PyTorch essential library
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, TensorDataset
import random # Library used for random index generation
import numpy as np # Library included for vector manipulation
import pandas as pd # Library included for dataframe manipulation
import matplotlib.pyplot as plt # Library included for plotting analysis
from sklearn.preprocessing import StandardScaler # Used to scale for normally distributed data
from sklearn.model_selection import train_test_split # Used to split test and train set

# read the dataset
df= pd.read_csv('dataset.csv')

# Drop the unnecessary columns
df.drop(labels=['Surname', 'CustomerId', 'RowNumber'], axis=1, inplace=True)

# Encoding all the values
df = pd.get_dummies(data=df, drop_first=True)

df.head()

df.shape

# Test set 30% of 10000= 3000
testSize = 3000
randomIndexCalls = []

while True:
    random_index = random.randint(0, (len(df))-1)
    if random_index not in randomIndexCalls:
        randomIndexCalls.append(random_index)
    if len(randomIndexCalls) == testSize:
        break

df_regular = df.drop(labels=randomIndexCalls, axis=0).reset_index(drop=True).copy()
df_test = df.iloc[randomIndexCalls].reset_index(drop=True).copy()

print("Dimensions of training data: " +str(df_regular.shape))
print("Dimensions of test data: " +str(df_test.shape))

# Contains data used to train the model
df = df_regular.copy()

# Train-test split
X = df.drop(labels=['Exited'], axis=1)
y = df['Exited'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

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
    # Utilizes the 11 inputs as specified from the columns
    def __init__(self, inputFeatureAmount):
        super(Network, self).__init__()
        # Branch out to 15 hidden layers 
        self.linear1 = nn.Linear(inputFeatureAmount, 15)
        # Branch again to 8 hidden layers
        self.linear2 = nn.Linear(15, 8)
        # Layer to determine the Exit value
        self.linear3 = nn.Linear(8, 1)
    
    #Utilizes the Sigmoid activiation function
    def forward(self, xb):
        prediction = torch.sigmoid(input=self.linear1(xb))
        prediction = torch.sigmoid(input=self.linear2(prediction))
        prediction = torch.sigmoid(input=self.linear3(prediction))
        return prediction
    
    # def forward(self, xb):
    #     prediction = torch.tanh(input=self.linear1(xb))
    #     prediction = torch.tanh(input=self.linear2(prediction))
    #     prediction = torch.tanh(input=self.linear3(prediction))
    #     return prediction

    # def forward(self, xb):
    #     prediction = torch.relu(input=self.linear1(xb))
    #     prediction = torch.relu(input=self.linear2(prediction))
    #     prediction = torch.relu(input=self.linear3(prediction))
    #     return prediction

model = Network(inputFeatureAmount=11)
print("Neural Network Diagram: " +str(model))
learningRate = 0.01

# Utilized the Binary Cross Entropy function for binary classification
criterionMeasure = nn.BCELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=learningRate)
#optimizer = torch.optim.Adam(params=model.parameters(), lr=learningRate)

numEpochs = 1000
offsetInterval = 200

dfTrack = pd.DataFrame()

print("\nTraining")
print("----------")

for epoch in range(1, numEpochs+1):
    y_pred = model(X_train)
    loss = criterionMeasure(input=y_pred, target=y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # For 1000 epochs there are 5 iterations
    if epoch % offsetInterval == 0:
        print("Epochs= " +str(epoch))
        print("Loss= " +str(round((loss.item())*100,3)) +"%")
    
    # For each epoch in the loop print the accuracy
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_class = y_pred.round()
        accuracy = y_pred_class.eq(y_test).sum() / float(len(y_test))
        if epoch % offsetInterval == 0:
            print("Accuracy: " +str(round((accuracy.item()*100),3)) +"%\n")
    
    df_temp = pd.DataFrame(data={'Epochs': epoch, 'Loss': round(loss.item(), 5), 'Accuracy': round(accuracy.item(), 5)}, index=[0])
    dfTrack = pd.concat(objs=[dfTrack, df_temp], ignore_index=True, sort=False)

print("\nOverall Accuracy: " +str(round((accuracy.item())*100,3)) +"%\n")

plt.figure(1)



plt.figure(2)
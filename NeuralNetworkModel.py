#Janakan Sivaloganthan 20/11/2021 Machine Learning Assignment
#The NeuralNetworkModel of the Bank Customer Churn dataset

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder


df= pd.read_csv('dataset.csv')

class CustomerChurn(Dataset):
    def __init__(self, csvpath, mode = 'train'):
            self.mode = mode
            df = pd.read_csv(csvpath)
            le = LabelEncoder()        
    """       
    <------Some Data Preprocessing---------->
    Removing Null Values, Outliers and Encoding the categorical labels etc
    """
    # if self.mode == 'train':
    #     df = df.dropna()
    #     self.inp = df.iloc[:,1:].values
    #     self.oup = df.iloc[:,0].values.reshape(891,1)
    # else:
    #     self.inp = df.values
    def __len__(self):
        return len(self.inp)
    def __getitem__(self, idx):
        if self.mode == 'train':
            inpt  = torch.Tensor(self.inp[idx])
            oupt  = torch.Tensor(self.oup[idx])
            return { 'inp': inpt,
                        'oup': oupt,
            }
        else:
            inpt = torch.Tensor(self.inp[idx])
            return { 'inp': inpt
            }


## Initialize the DataSet
data = CustomerChurn('dataset.csv')
## Load the Dataset
BATCH_SIZE = 16
data_train = DataLoader(dataset = data, batch_size = BATCH_SIZE, shuffle =False)


#Neural Network architecture
def swish(x):
    return x * F.sigmoid(x)

class Network(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(8, 16)
        self.b1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 8)
        self.b2 = nn.BatchNorm1d(8)
        self.fc3 = nn.Linear(8,4)
        self.b3 = nn.BatchNorm1d(4)
        self.fc4 = nn.Linear(4,1)

    def forward(self,x):

        x = swish(self.fc1(x))
        x = self.b1(x)
        x = swish(self.fc2(x))
        x = self.b2(x)
        x = swish(self.fc3(x))
        x = self.b3(x)
        x = F.sigmoid(self.fc4(x))

        return x


if torch.cuda.is_available():
  device = torch.device("cuda:0")
  print("Cuda Device Available")
  print("Name of the Cuda Device: ", torch.cuda.get_device_name())
  print("GPU Computational Capablity: ", torch.cuda.get_device_capability())

#training
EPOCHS = 200
data_train = DataLoader(dataset = data, batch_size = BATCH_SIZE, shuffle =False)
criterion = nn.MSELoss()
for epoch in range(EPOCHS):
    epoch_loss = 0
    correct = 0
    for bidx, batch in tqdm(enumerate(data_train)):
        x_train, y_train = batch['inp'], batch['oup']
        x_train = x_train.view(-1,8)
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        loss, predictions = train(net,x_train,y_train, optm, criterion)
        for idx, i in enumerate(predictions):
            i  = torch.round(i)
            if i == y_train[idx]:
                correct += 1
        acc = (correct/len(data))
        epoch_loss+=loss
    print('Epoch {} Accuracy : {}'.format(epoch+1, acc*100))
    print('Epoch {} Loss : {}'.format((epoch+1),epoch_loss))

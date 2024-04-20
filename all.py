
import numpy as np
import torch

def readData(f):
  # Read Data from the system
    return np.genfromtxt(f, delimiter=',', dtype=str)[1:]
def saveModel(net, path):
  # Save the model
    torch.save(net.state_dict(), path)

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, labels, device='gpu'):
        'Initialization'
        self.data = data.to(device)
        self.labels = labels

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.data[index]
        y = self.labels[index]
        return X, y, index
        
import numpy as np
import torch


def preprocess(data, flip=True):
    date   = data[:, 0]
    open   = data[:, 1]
    high   = data[:, 2]
    low    = data[:, 3]
    close  = data[:, 4]
    adj = data[:, 5]
    volume = data[:, 6]
    prices = np.array([close for date, open, high, low, close, adj, volume in data]).astype(np.float64)
    if flip:
        prices = np.flip(prices)
    return prices


def train_test_split(data, percentage=0.8):
    train_size  = int(len(data) * percentage)
    train, test = data[:train_size], data[train_size:]
    return train, test


def transform_dataset(dataset, look_back=5):
    # N days as training sample
    dataX = [dataset[i:(i + look_back)]
            for i in range(len(dataset)-look_back-1)]
    # 1 day as groundtruth
    dataY = [dataset[i + look_back]
            for i in range(len(dataset)-look_back-1)]
    return torch.tensor(np.array(dataX), dtype=torch.float32), torch.tensor(np.array(dataY), dtype=torch.float32)


import torch.nn as nn
import torch


class LSTMPredictor(nn.Module):

    def __init__(self, look_back, num_layers=2, dropout=0.5, bidirectional=True):
        super(LSTMPredictor, self).__init__()

        # Nerual Layers LSTM - Long Short Term Memory
        self.rnn   = nn.LSTM(look_back, 32, num_layers, dropout=dropout, bidirectional=True)
        self.ly_a  = nn.Linear(32*(2 if bidirectional else 1), 16)
        # self.ly_a  = nn.Linear(look_back, 16)
        self.relu  = nn.ReLU()
        self.reg   = nn.Linear(16, 1)

    def predict(self, input):
        with torch.no_grad():
            return self.forward(input).item()

    def forward(self, input):
        r_out, (h_n, h_c) = self.rnn(input.unsqueeze(1), None)
        logits = self.reg(self.relu(self.ly_a(r_out.squeeze(1))))
        # logits = self.reg(self.relu(self.ly_a(input)))

        return logits
    
    
    import numpy as np
import torch

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')



def trainer(net, criterion, optimizer, trainloader, devloader, epoch_n=100, path="./checkpoint/save.pt"):
    # with open("result.csv",'a') as f:
    for epoch in range(epoch_n): # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        train_loss = 0.0
        valid_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, data_index = data

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs.cpu(), labels.unsqueeze(1).cpu())
            train_loss += loss.item()*inputs.shape[0]
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        ######################
        # validate the model #
        ######################
        net.eval()
        for i, data in enumerate(devloader, 0):
            # move tensors to GPU if CUDA is available
            inputs, labels, data_index = data
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = net(inputs)
            # calculate the batch loss
            loss = criterion(outputs.cpu(), labels.cpu())
            # update average validation loss
            valid_loss += loss.item()*inputs.shape[0]

        # calculate average losses
        train_loss = train_loss/len(trainloader.dataset)
        valid_loss = valid_loss/len(devloader.dataset)

        # print training/validation statistics
        # f.writelines(f"epoch:{epoch}, train loss{train_loss}, val loss{valid_loss}/n")
        print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(train_loss, valid_loss))

    print('Finished Training')

    ## Save model
    saveModel(net, path)
    
def tester(net, criterion, testloader):
    loss = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels, data_index = data
            outputs = net(inputs)
            loss += criterion(outputs.cpu(), labels.unsqueeze(1).cpu())
    return loss.item()


import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data.dataloader import default_collate

import os
import math

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

data = readData("0050.TW.csv")
print('Num of samples:', len(data))

prices = preprocess(data)
# Divide trainset and test set
train, test = train_test_split(prices, 0.8)
# Set the N(look_back)=5 because from the five day stock, we are predicting the next day
look_back = 5
trainX, trainY = transform_dataset(train, look_back)
testX, testY   = transform_dataset(test, look_back)
# Get dataset
trainset = Dataset(trainX, trainY, device)
testset  = Dataset(testX, testY, device)
# Get dataloader
batch_size = 200
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0) # num_workers should set 1 if put data on CUDA
testloader  = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)


net = LSTMPredictor(look_back)
net.to(device)


criterion = nn.MSELoss() # Feel free to use any other loss

optimizer = optim.Adam(net.parameters(), lr=0.0001) # you can tweak the lr and see if it affects anything


## Training
checkpoint = "checkpoint/save.pt"
if not os.path.isfile(checkpoint):
  os.makedirs("./checkpoint")
  trainer(net, criterion, optimizer, trainloader, testloader, epoch_n=300, path=checkpoint)
else:
  net.load_state_dict(torch.load(checkpoint))
  
  
  test = tester(net, criterion, testloader)
# Show the difference between predict and groundtruth (loss)
print('Test Result: ', test)

predict = net.predict(torch.tensor([[143.95,143.35,143.3,142.8,146.95]], dtype=torch.float32).to(device))

print('Predicted Result', predict)

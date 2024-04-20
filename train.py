import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data.dataloader import default_collate

import os
import math
from data import readData, Dataset
from preprocessing import preprocess, train_test_split, transform_dataset
from model import LSTMPredictor
from trainer import trainer , tester

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
batch_size = 40
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
  trainer(net, criterion, optimizer, trainloader, testloader, epoch_n=1000, path=checkpoint)
else:
  net.load_state_dict(torch.load(checkpoint))
  
test = tester(net, criterion, testloader)
# Show the difference between predict and groundtruth (loss)
print('Test Result: ', test)

predict = net.predict(torch.tensor([[154.40	,154.05,153.20,157.40	,157.20]], dtype=torch.float32).to(device))
print('Predicted Result', predict)

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import os
# from data import readData, Dataset
# from preprocessing import preprocess, train_test_split, transform_dataset
# from model import Seq2Seq ,Encoder,Decoder
# from trainer import trainer , tester

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")

# data = readData("0050.TW.csv")
# print('Num of samples:', len(data))

# prices = preprocess(data)
# # Divide trainset and test set
# train, test = train_test_split(prices, 0.8)
# # Set the N(look_back)=5 because from the five day stock, we are predicting the next day
# look_back = 5
# trainX, trainY = transform_dataset(train, look_back)
# testX, testY   = transform_dataset(test, look_back)
# # Get dataset
# trainset = Dataset(trainX, trainY, device)
# testset  = Dataset(testX, testY, device)
# # Get dataloader
# batch_size = 40
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0) 
# testloader  = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

# encoder = Encoder(input_size=look_back, hidden_size=32)
# decoder = Decoder(input_size=1, hidden_size=32, output_size=1)

# net = Seq2Seq(encoder, decoder, device)
# net.to(device)

# criterion = nn.MSELoss()  # or any other loss function you want to use
# optimizer = optim.Adam(net.parameters(), lr=0.0001)
# ## Training
# checkpoint = "checkpoint/save2.pt"
# if not os.path.isfile(checkpoint):
#     os.makedirs("./checkpoint")
#     trainer(net, criterion, optimizer, trainloader, testloader, epoch_n=1000, path=checkpoint)
# else:
#     net.load_state_dict(torch.load(checkpoint))
  
# test = tester(net, criterion, testloader)
# print('Test Result:', test)

# # Modify your input data to match the input format of Seq2Seq model
# predict_input = torch.tensor([[154.40, 154.05, 153.20, 157.40, 157.20]], dtype=torch.float32).to(device)
# # Perform prediction using the Seq2Seq model
# predict = net(predict_input, torch.zeros_like(predict_input))  
#   # Adjust this line according to your decoder's input requirements
# print('Predicted Result', predict)

import random
import torch.nn as nn
import torch
import numpy as np
class LSTM(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super(LSTM, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(input_size=self.n_features, hidden_size=self.hidden_dim, 
                            num_layers=2, bidirectional=False, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(self.hidden_dim, 5)
        
    def forward(self, x): 
        x, _ = self.lstm(x)
        x = self.linear(x[:,-1,:]) #取最後一個state做NN
        
        return x

model = LSTM(5, 128)
model


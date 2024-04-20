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
    

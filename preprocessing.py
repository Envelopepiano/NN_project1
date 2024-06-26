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

import numpy as np
import torch
import pandas as pd  

def read_data(file_path):
    """
    Read data from a CSV file and return a DataFrame.
    
    Parameters:
        file_path (str): The path to the CSV file.
        
    Returns:
        pandas.DataFrame: The DataFrame containing the data.
    """
    df = pd.read_csv(file_path)
    return df

def preprocess(file_path, train_ratio, n_past):
    """
    Read data from a CSV file, preprocess it, and create sequences for training and validation.
    
    Parameters:
        file_path (str): The path to the CSV file.
        train_ratio (float): The ratio of data to be used for training.
        n_past (int): The number of past time steps to consider for each sequence.
        
    Returns:
        torch.Tensor: X_train (input sequences for training)
        torch.Tensor: Y_train (output sequences for training)
        torch.Tensor: X_val (input sequences for validation)
        torch.Tensor: Y_val (output sequences for validation)
    """
    df = read_data(file_path)
    # Selecting columns except 'Date' and 'Adj Close'
    data = df[[c for c in df.columns if c not in ['Date', 'Adj Close']]].values
    
    train_ind = int(len(data) * train_ratio)
    train_data = data #train all data
    val_data = data[train_ind:]

    # Convert train data into PyTorch tensor
    X_train, Y_train = create_sequences(train_data, n_past)
    X_val, Y_val = create_sequences(val_data, n_past)

    return X_train, Y_train, X_val, Y_val

def create_sequences(data, n_past):
    """
    Create sequences from the input data.
    
    Parameters:
        data (numpy.ndarray): The input data.
        n_past (int): The number of past time steps to consider for each sequence.
        
    Returns:
        torch.Tensor: X (input sequences)
        torch.Tensor: Y (output sequences)
    """
    X, Y = [], []
    L = len(data)
    for i in range(L - (n_past + 5)):
        X.append(data[i:i + n_past])
        Y.append(data[i + n_past:i + n_past + 5][:, 3])
    Y = torch.Tensor(np.array(Y)).unsqueeze(1)
    return torch.Tensor(np.array(X)), torch.Tensor(np.array(Y))

# Set your parameters
file_path = '0050.TW.csv'
train_ratio = 0.8
n_past = 20

# Preprocess your data
X_train, Y_train, X_val, Y_val = preprocess(file_path, train_ratio, n_past)
#print("X_train:",X_train)
# Create data loaders
batch_size = 32

train_set = torch.utils.data.TensorDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
val_set = torch.utils.data.TensorDataset(X_val, Y_val)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

print(X_train.shape) # torch.Size([624, 20, 5])
print(Y_train.shape) # torch.Size([624, 5])

# for batch_idx, (inputs, labels) in enumerate(train_loader):
#     print("Batch", batch_idx + 1)
#     print("Inputs shape:", inputs.shape)
#     print("Labels shape:", labels.shape)
import numpy as np
import torch
import pandas as pd  

def read_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess(file_path, train_ratio, n_past):
    df = read_data(file_path)
    # 除了日期跟adj
    data = df[[c for c in df.columns if c not in ['Date', 'Adj Close']]].values
    
    train_ind = int(len(data) * train_ratio)
    train_data = data 
    val_data = data[train_ind:]
    #轉換torch
    X_train, Y_train = create_sequences(train_data, n_past)
    X_val, Y_val = create_sequences(val_data, n_past)

    return X_train, Y_train, X_val, Y_val

def create_sequences(data, n_past):
    X, Y = [], []
    L = len(data)
    for i in range(L - (n_past + 5)):
        X.append(data[i:i + n_past])
        Y.append(data[i + n_past:i + n_past + 5][:, 3])
    return torch.Tensor(np.array(X)), torch.Tensor(np.array(Y))

#參數
file_path = '0050.TW.csv'
train_ratio = 0.8
n_past = 20

# preprocess
X_train, Y_train, X_val, Y_val = preprocess(file_path, train_ratio, n_past)
#print("X_train:",X_train)
#測試
batch_size = 32
train_set = torch.utils.data.TensorDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)

val_set = torch.utils.data.TensorDataset(X_val, Y_val)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

print(X_train.shape) #測形狀
print(Y_train.shape) # 

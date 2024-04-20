import torch
# from seq_preprocessing import scaler
from seq_model import model
from seq_preprocessing import train_loader

# 迭代 train_loader 的第一个 batch，并打印其中的数据
for batch in train_loader:
    inputs, labels = batch
    print("Inputs shape:", inputs.shape)
    print("Labels shape:", labels.shape)
    break  # 仅查看第一个 batch

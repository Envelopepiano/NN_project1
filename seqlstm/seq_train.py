import torch
import torch.nn as nn
import torch.optim as optim
from seq_model import model
# from seq_preprocessing import scaler 用正規化
from seq_preprocessing import train_loader, val_loader 
from torch.utils.data.dataloader import default_collate
from torch_optimizer import Ranger
import matplotlib.pyplot as plt
# MSE
criterion = nn.MSELoss()
# 用RANGER
optimizer = Ranger(model.parameters(),lr=0.001)
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
# 設一個最低val loss
best_val_loss = float('inf')
train_losses = []
val_losses = []
epochs = 1000

for epoch in range(epochs):
    # 用訓練模式
    model.train()
    train_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    # 用驗證模式
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    
    # 存loss
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    

    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# 畫圖
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# 保存
torch.save(model.state_dict(), 'model_epoch_{}.pt'.format(epoch+1))
print('Finished Training')
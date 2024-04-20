import torch
import torch.nn as nn
import torch.optim as optim
from en_de_coder import Encoder, Decoder
from new_model import Seq2Seq
# from seq_preprocessing import scaler
from new_preprocessing import train_loader, val_loader 
from torch.utils.data.dataloader import default_collate
from torch_optimizer import Ranger
import matplotlib.pyplot as plt
criterion = nn.MSELoss()

# 创建 Encoder 和 Decoder
encoder = Encoder(input_size=5, embedding_size=36, hidden_size=128, n_layers=1, dropout=0.3)
decoder = Decoder(output_size=5, embedding_size=36, hidden_size=128, n_layers=1, dropout=0.3)

# 创建 Seq2Seq 模型
model = Seq2Seq(encoder, decoder, 'cpu')

# 定义优化器
optimizer = Ranger(model.parameters(), lr=0.01)

# 存储训练和验证损失
train_losses = []
val_losses = []

epochs = 1000

for epoch in range(epochs):
    # 训练模式
    model.train()
    train_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        # 将输入序列和目标序列传递给模型
        outputs = model(inputs, labels)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 计算平均训练损失
    train_loss /= len(train_loader)

    # 验证模式
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data

            # 将输入序列和目标序列传递给模型
            outputs = model(inputs, labels)
            # 计算损失
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    # 计算平均验证损失
    val_loss /= len(val_loader)

    # 存储损失
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # 打印训练损失和验证损失
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# 绘制损失曲线
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# 保存模型
torch.save(model.state_dict(), 'model_epoch_{}.pt'.format(epoch+1))
print('Finished Training')
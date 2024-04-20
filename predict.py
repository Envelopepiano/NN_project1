import torch
import torch.nn as nn
from train import net ,device,loadModel

loadModel(net, "./checkpoint/save.pt")

# 將模型轉移到指定的裝置上（例如GPU）
net.to(device)
predict = net.predict(torch.tensor([[155.10,153.80,150,150,150]], dtype=torch.float32).to(device))
print('Predicted Result', predict)
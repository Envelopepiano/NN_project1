
import torch
from seq_model import model
from seq_preprocessing import train_loader,val_loader
#載模型拉
model.load_state_dict(torch.load('C:/NN_project1/model_epoch_300.pt'))
# 拿最後一天的數據
last_day_data = val_loader.dataset[-1][0]
print("last_day_data:",last_day_data)
# 正規化
# last_day_data = scaler.transform(last_day_data)

last_day_data_tensor = torch.tensor(last_day_data, dtype=torch.float32).unsqueeze(0)
print("last_day_tensor:",last_day_data_tensor)
print("size:",last_day_data_tensor.shape)
#預測
with torch.no_grad():
    predicted_value = model(last_day_data_tensor)
print("Predicted value for the next five days:", predicted_value)

import torch
from en_de_coder import Encoder, Decoder 
from new_model import Seq2Seq
from new_preprocessing import preprocess

# 加载训练好的模型参数
encoder = Encoder(input_size=5, embedding_size=36, hidden_size=128, n_layers=1, dropout=0.3)
decoder = Decoder(output_size=5, embedding_size=36, hidden_size=128, n_layers=1, dropout=0.3)
model = Seq2Seq(encoder, decoder, 'cpu')
model.load_state_dict(torch.load('C:/NN_project1/model_epoch_400.pt'))

# 获取训练数据的最后一天数据
file_path = '0050.TW.csv'
train_ratio = 0.8
n_past = 20
X_train, Y_train, X_val, Y_val = preprocess(file_path, train_ratio, n_past)
last_day_data = X_val[-1]
last_day_data_tensor = torch.tensor(last_day_data, dtype=torch.float32).unsqueeze(0)

# 使用模型进行预测
with torch.no_grad():
    # 提供一个空的张量作为目标序列 y
    dummy_target = torch.zeros_like(last_day_data_tensor)  # 创建一个与输入形状相同的空目标序列
    predicted_value = model(last_day_data_tensor, dummy_target)  # 将空目标序列传递给模型

# 仅保留最后五天的预测值
predicted_value_last_five_days = predicted_value[:, :5, :]  # 仅保留最后五个时间步的预测值

# 输出预测值
print("Predicted value for the next five days:", predicted_value_last_five_days)

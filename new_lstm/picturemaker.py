import matplotlib.pyplot as plt

# 创建一个新的图表
plt.figure(figsize=(10, 6))

# 绘制 Encoder 部分
encoder_layers = 2
encoder_hidden_size = 128
for i in range(encoder_layers):
    plt.text(0.1, i * 0.2, f"LSTM Layer {i+1}\nInput: (?, ?, 5)\nHidden: (?, {encoder_hidden_size})", fontsize=10, va='center')
    plt.arrow(0.25, i * 0.2 + 0.1, 0.1, 0, head_width=0.05, head_length=0.03, fc='k', ec='k')

# 绘制 Decoder 部分
decoder_layers = 2
decoder_hidden_size = 128
for i in range(decoder_layers):
    plt.text(0.7, i * 0.2, f"LSTM Layer {i+1}\nInput: (?, ?, 5)\nHidden: (?, {decoder_hidden_size})", fontsize=10, va='center')
    plt.arrow(0.85, i * 0.2 + 0.1, 0.1, 0, head_width=0.05, head_length=0.03, fc='k', ec='k')

# 连接 Encoder 和 Decoder
plt.arrow(0.45, 0.3, 0.1, 0, head_width=0.05, head_length=0.03, fc='k', ec='k')
plt.arrow(0.45, 0.7, 0.1, 0, head_width=0.05, head_length=0.03, fc='k', ec='k')

# 添加标注
plt.text(0.2, -0.1, "Encoder", fontsize=12, va='center')
plt.text(0.7, -0.1, "Decoder", fontsize=12, va='center')

# 设置图表标题和坐标轴
plt.title("Seq2Seq Model Architecture")
plt.axis('off')

# 显示图表
plt.show()

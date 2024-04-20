from seq_preprocessing import val_set
import torch
from seq_model import model
import numpy as np
import matplotlib.pyplot as plt

close_price_true, close_price_pred = [], []
for i in range(len(val_set)):
    x,y = val_set[i]
    x = torch.unsqueeze(x,0)
    y = torch.unsqueeze(y,0)
    pred = model(x.to('cpu'))
    
    np_output = pred.detach().numpy()[0]
    np_target = y[0][0].item()
    # np_output = np.sqrt(scaler.var_[3])*np_output + scaler.mean_[3] #dim=3為close price的
    # np_target = np.sqrt(scaler.var_[3])*np_target + scaler.mean_[3]
    
    close_price_true.append(np_target)
    close_price_pred.append(np_output[0])
 
mape = np.mean(np.absolute(np.array(close_price_true) - np.array(close_price_pred))/np.array(close_price_true))
mae =  np.mean( np.absolute(np.array(close_price_true) - np.array(close_price_pred)))
plt.plot(close_price_true)
plt.plot(close_price_pred)
plt.title('MAE:{:.3f}, MAPE:{:.3f}'.format(mae, mape))
plt.legend(['true', 'predict'])
plt.show()
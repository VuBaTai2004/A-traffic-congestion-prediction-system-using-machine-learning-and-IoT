import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Giả sử bạn có input_size = 6, hidden_size = 64
model = LSTMModel(input_size=3, hidden_size=64, num_layers=2)

model.load_state_dict(torch.load('weather_lstm_model.pth'))  # <-- Đường dẫn file bạn lưu


# Đảm bảo model đang ở chế độ đánh giá
model.eval()


# Không cần gradient
with torch.no_grad():
    # Dự đoán trên tập train
    train_pred = model(X_train).cpu().numpy()
    y_train_true = Y_train.cpu().numpy()

    # Dự đoán trên tập validation
    val_pred = model(X_val).cpu().numpy()
    y_val_true = Y_val.cpu().numpy()

# In ra metrics
print("\n📊 Training Evaluation:")
print("MSE:", mean_squared_error(y_train_true, train_pred))
print("MAE:", mean_absolute_error(y_train_true, train_pred))

print("\n📊 Validation Evaluation:")
print("MSE:", mean_squared_error(y_val_true, val_pred))
print("MAE:", mean_absolute_error(y_val_true, val_pred))

# Vẽ biểu đồ
plt.figure(figsize=(14,5))

# Train plot
plt.subplot(1, 2, 1)
plt.plot(y_train_true, label='Ground Truth')
plt.plot(train_pred, label='Prediction')
plt.title('Training Set')
plt.legend()

# Validation plot
plt.subplot(1, 2, 2)
plt.plot(y_val_true, label='Ground Truth')
plt.plot(val_pred, label='Prediction')
plt.title('Validation Set')
plt.legend()

plt.show()

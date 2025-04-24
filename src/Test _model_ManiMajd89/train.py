import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
df = pd.read_csv('weatherHistory.csv')

# Chọn cột cần dùng
features = df[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)']]

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(features)

# Tạo dữ liệu dạng sequence
def create_sequences(data, look_back=30):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        Y.append(data[i + look_back, 0])  # Temperature (C)
    return np.array(X), np.array(Y)

look_back = 30
X, Y = create_sequences(data_scaled, look_back)

# Chia train-test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# Chuyển sang tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)

# Định nghĩa model LSTM
class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(WeatherLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Khởi tạo model
input_size = X_train.shape[2]  # số lượng feature
model = WeatherLSTM(input_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

from torch.utils.data import TensorDataset, DataLoader

# Tạo dataset
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

# Train model
epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_x, batch_y in train_loader:
        output = model(batch_x)
        loss = criterion(output, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")


# Dự đoán
model.eval()
with torch.no_grad():
    predicted = model(X_test).numpy()

# Giải scale dữ liệu
predicted_full = np.concatenate((predicted, np.zeros((predicted.shape[0], features.shape[1] - 1))), axis=1)
Y_test_full = np.concatenate((Y_test.numpy(), np.zeros((Y_test.shape[0], features.shape[1] - 1))), axis=1)

predicted_temp = scaler.inverse_transform(predicted_full)[:, 0]
actual_temp = scaler.inverse_transform(Y_test_full)[:, 0]

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))
plt.plot(actual_temp, label='Actual Temperature')
plt.plot(predicted_temp, label='Predicted Temperature')
plt.legend()
plt.title("Weather Forecast with LSTM")
plt.show()


torch.save(model.state_dict(), 'weather_lstm_model.pth')

# %%
#Импорт библиотек
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchmetrics import MetricCollection, MeanAbsoluteError, MeanSquaredError, R2Score

# %%
from torchinfo import summary

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
#Загрузка данных
data = yf.download('NVDA', start='2020-01-01', end='2024-01-01')
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
data['Volatility'] = data['High'] - data['Low']
data['Price_Change'] = data['Close'].diff()
data.fillna(data.bfill(),inplace=True)
data.fillna(data.ffill(), inplace=True)

# %%
data.head(30)

# %%
#Нормировка данных
features = ['Close', 'SMA_10', 'SMA_20', 'EMA_10', 'Volatility', 'Price_Change']
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[features])

# %%
#Параметр длины окна
seq_length = 60

# %%
#Создание временных окон
def create_multifeature_sequences(data, seq_length, stride):
    x, y = [], []
    for i in range(0, len(data) - seq_length, stride):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])
    return np.array(x), np.array(y)


x, y = create_multifeature_sequences(scaled_data, seq_length, stride=1)

# %%
#Разделение на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

# %%
#Преобразование в torch тензоры
x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# %%
#LSTM модель
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

input_size = len(features)
hidden_size = 1000
num_layers = 2
output_size = 1


# %%
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

# %%
#Определение функции потерь и отпимизатора
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
#Число эпох и размер батча
num_epochs = 20
batch_size = 64 

# %%
#Обучение модели
for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        outputs = model(x_batch)
        loss = criterion(outputs.squeeze(), y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# %%
summary(model)

# %%
#Тестирование модели
model.eval()
with torch.no_grad():
    y_pred = model(x_test).squeeze()
    y_pred = y_pred.cpu().numpy()
    y_test = y_test.cpu().numpy()

# %%
# Масштабирование обратно
y_pred_rescaled = scaler.inverse_transform(
    np.concatenate((y_pred.reshape(-1, 1), np.zeros((len(y_pred), len(features)-1))), axis=1)
)[:, 0]

y_test_rescaled = scaler.inverse_transform(
    np.concatenate((y_test.reshape(-1, 1), np.zeros((len(y_test), len(features)-1))), axis=1)
)[:, 0]


# %%
# Преобразуем предсказания и реальные значения в тензоры
y_pred_tensor = torch.tensor(y_pred_rescaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_rescaled, dtype=torch.float32).to(device)

# Создаем коллекцию метрик
metrics = MetricCollection([
    MeanAbsoluteError().to(device),
    MeanSquaredError().to(device),
    R2Score().to(device)
])

# Применяем метрики
metrics_result = metrics(y_pred_tensor, y_test_tensor)
print(f"Metrics: {metrics_result}")


# %%
#Постройка графика
plt.figure(figsize=(14, 5))
plt.plot(y_test_rescaled, label='Реальные цены', color='blue')
plt.plot(y_pred_rescaled, label='Предугаданные цены', color='red')
plt.title('Предсказание цены закрытия акций NVIDIA')
plt.xlabel('Время')
plt.ylabel('Цена')
plt.legend()
plt.show()



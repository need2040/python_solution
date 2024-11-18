# %% Импорт необходимых библиотек
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
import torch
import matplotlib.pyplot as plt

# %% Настройка устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Загрузка данных
data = yf.download('NVDA', start='2020-01-01', end='2024-01-01')

# Удаление многоуровневых индексов, если они есть
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(0)
#%%
# Сброс индекса и настройка данных
data.reset_index(inplace=True)
data['time_idx'] = range(len(data))  # Индекс времени
data = data[['time_idx', 'Close']]  # Только временной индекс и целевая переменная

# Нормализация данных
scaler = MinMaxScaler()
data['Close'] = scaler.fit_transform(data[['Close']])

# %% Подготовка данных для TimeSeriesDataSet
max_encoder_length = 60  # Длина входной последовательности
max_prediction_length = 10  # Длина прогноза

dataset = TimeSeriesDataSet(
    data,
    time_idx="time_idx",
    target="Close",
    group_ids=["time_idx"],  # Для одиночного временного ряда используем time_idx как группу
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    time_varying_known_reals=["time_idx"],  # Временные признаки, известные заранее
    time_varying_unknown_reals=["Close"],  # Значение целевой переменной как неизвестное
    target_normalizer=None  # Мы уже нормализовали данные
)

# %% Разделение данных на обучающие и тестовые наборы
train_size = 0.8
train_cutoff = int(len(data) * train_size)

train_dataset = dataset.slice(0, train_cutoff)
val_dataset = dataset.slice(train_cutoff, len(data))

# DataLoaders для обучения и тестирования
batch_size = 64
train_dataloader = train_dataset.to_dataloader(train=True, batch_size=batch_size)
val_dataloader = val_dataset.to_dataloader(train=False, batch_size=batch_size)

# %% Создание модели Temporal Fusion Transformer
model = TemporalFusionTransformer.from_dataset(
    train_dataset,
    learning_rate=0.01,
    hidden_size=128,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=64,
    output_size=1,  # Количество выходов (зависит от задачи)
    loss=torch.nn.MSELoss(),
    log_interval=10,
    reduce_on_plateau_patience=3,
)

# %% Обучение модели
early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")
trainer = Trainer(
    max_epochs=30,
    gpus=1 if torch.cuda.is_available() else 0,
    gradient_clip_val=0.1,
    callbacks=[early_stop_callback],
)

trainer.fit(model, train_dataloader, val_dataloader)

# %% Предсказания
raw_predictions, x = model.predict(val_dataloader, mode="raw", return_x=True)

# Построение графика предсказаний
actual = torch.cat([x["decoder_target"] for x in raw_predictions]).numpy()
predicted = torch.cat([x["prediction"][:, 0] for x in raw_predictions]).numpy()

plt.figure(figsize=(12, 6))
plt.plot(actual, label="Actual")
plt.plot(predicted, label="Predicted", alpha=0.7)
plt.legend()
plt.title("Actual vs Predicted (Temporal Fusion Transformer)")
plt.show()

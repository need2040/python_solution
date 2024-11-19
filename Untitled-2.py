# %%
# Импорт необходимых библиотек
import warnings
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline, GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl
import torch
import matplotlib.pyplot as plt
from pytorch_forecasting.metrics import RMSE
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Загрузка данных
data = yf.download('NVDA', start='2020-01-01', end='2024-01-01')


# %%
data['time_idx'] = range(len(data))  # Индекс времени
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
data['Volatility'] = data['High'] - data['Low']
data['Price_Change'] = data['Close'].diff()
data.bfill(inplace=True)
data.ffill(inplace=True)

# %%
data.reset_index(inplace=True)
data.columns = data.columns.droplevel(1)
data.drop(columns=['Date'], inplace=True)
data["series_id"] = 0  # один уникальный идентификатор для всего ряда

# %%
data.head()

# %%
# Подготовка данных для TimeSeriesDataSet
max_encoder_length = 60  # Длина входной последовательности
max_prediction_length = 10  # Длина прогноза (1 шаг вперёд)
training_cutoff = data["time_idx"].max() - max_prediction_length

# %%
# Создание TimeSeriesDataSet
training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="Close",
    group_ids=["series_id"],  # Указываем фиктивный идентификатор группы
    min_encoder_length=max_encoder_length //2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=[],  # Нет категориальных данных
    time_varying_known_reals=['SMA_10', 'SMA_20','Volatility'],  # Временные признаки, известные заранее
    time_varying_unknown_reals=["Close"],  # Используем только значение закрытия
    target_normalizer=GroupNormalizer(groups=["series_id"], transformation="softplus"),  
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)


# %%
validation = TimeSeriesDataSet.from_dataset(training,data, predict=True,stop_randomization=True)

# %%
#Размеры батча
batch_size = 64
train_dataloader = training.to_dataloader(train=True,batch_size=batch_size,num_workers = 0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

# %%
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

# %%
trainer = pl.Trainer(
    accelerator='gpu',
    gradient_clip_val=0.1,
    max_epochs=30,
    #fast_dev_run = True,
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
    log_every_n_steps=5
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    # not meaningful for finding the learning rate but otherwise very important
    learning_rate=0.03,
    hidden_size=16,  # most important hyperparameter apart from learning rate
    # number of attention heads. Set to up to 4 for large datasets
    attention_head_size=2,
    dropout=0.1,  # between 0.1 and 0.3 are good values
    hidden_continuous_size=8,  # set to <= hidden_size
    loss=RMSE(),
    optimizer="ranger",
    
)
print(f"Число параметров сети: {tft.size()/1e3:.1f}k")

# %%
# fit network
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# %%
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)



# %% Importing necessary libraries
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE
import matplotlib.pyplot as plt
import numpy as np

# %% Setting up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Downloading stock data
data = yf.download("NVDA", start="2020-01-01", end="2024-01-01")
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# %% Feature engineering
data["time_idx"] = range(len(data))  # Adding time index
data["SMA_10"] = data["Close"].rolling(window=10).mean()
data["SMA_20"] = data["Close"].rolling(window=20).mean()
data["EMA_10"] = data["Close"].ewm(span=10, adjust=False).mean()
data["Volatility"] = data["High"] - data["Low"]
data["Price_Change"] = data["Close"].diff()

# Fill missing values
data.fillna(method="bfill", inplace=True)
data.fillna(method="ffill", inplace=True)

# Add group column for TimeSeriesDataSet
data["group"] = 0

# Scaling features
scaler = MinMaxScaler()
scaled_features = ["Close", "SMA_10", "SMA_20", "EMA_10", "Volatility", "Price_Change"]
data[scaled_features] = scaler.fit_transform(data[scaled_features])

# %% Define TimeSeriesDataSet parameters
max_encoder_length = 60  # Input sequence length
max_prediction_length = 10  # Prediction sequence length

# %% Create TimeSeriesDataSet
dataset = TimeSeriesDataSet(
    data,
    time_idx="time_idx",
    target="Close",
    group_ids=["group"],
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    time_varying_known_reals=["time_idx", "SMA_10", "SMA_20", "EMA_10", "Volatility", "Price_Change"],
    time_varying_unknown_reals=["Close"],
    target_normalizer=None,  # Data is already normalized
)

# %% Split data into train and validation sets
train_size = 0.8
train_cutoff = int(len(data) * train_size)
train_data = data.iloc[:train_cutoff]
val_data = data.iloc[train_cutoff:]

train_dataset = TimeSeriesDataSet.from_dataset(dataset, train_data)
val_dataset = TimeSeriesDataSet.from_dataset(dataset, val_data)

# %% Create DataLoaders
batch_size = 64
train_dataloader = train_dataset.to_dataloader(train=True, batch_size=batch_size)
val_dataloader = val_dataset.to_dataloader(train=False, batch_size=batch_size)

# %% Initialize Temporal Fusion Transformer model
model = TemporalFusionTransformer.from_dataset(
    train_dataset,
    learning_rate=0.01,
    hidden_size=128,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=64,
    output_size=1,
    loss=RMSE(),
    log_interval=10,
)
model.to(device)

# %% Training and evaluation loops
num_epochs = 30
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = RMSE()

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()

        # Check the batch type (tuple)
        x, y = batch
        x = {key: value.to(device) for key, value in x.items()}
        y = y.to(device)

        output = model.forward(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_dataloader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            x, y = batch
            x = {key: value.to(device) for key, value in x.items()}
            y = y.to(device)

            output = model.forward(x)
            loss = loss_fn(output, y)
            val_loss += loss.item()

    val_loss /= len(val_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# %% Evaluate and visualize predictions
model.eval()
with torch.no_grad():
    predictions = []
    actuals = []
    for batch in val_dataloader:
        x, y = batch
        x = {key: value.to(device) for key, value in x.items()}
        y = y.to(device)

        output = model.predict(x)
        predictions.append(output.cpu().numpy())
        actuals.append(y.cpu().numpy())

predictions = np.concatenate(predictions, axis=0)
actuals = np.concatenate(actuals, axis=0)

plt.figure(figsize=(12, 6))
plt.plot(actuals, label="Actual")
plt.plot(predictions, label="Predicted", alpha=0.7)
plt.legend()
plt.show()

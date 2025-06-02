import numpy as np
import pandas as pd
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import datetime

API_KEY = "Binance api key"
API_SECRET = "secret key"
client = Client(API_KEY, API_SECRET)


symbol = 'BTCUSDT'
interval = Client.KLINE_INTERVAL_1HOUR
start_str = '1 Jan, 2020'
end_str = '21 Apr, 2025'

print("Downloading data from Binance")
candlesticks = client.get_historical_klines(symbol, interval, start_str, end_str)

close_prices = [float(candle[4]) for candle in candlesticks]
data = np.array(close_prices).reshape(-1, 1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

sequence_length = 60
X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

early_stop = EarlyStopping(monitor='loss', patience=5)
print("Training model")
model.fit(X, y, epochs=5, batch_size=32, callbacks=[early_stop])

model.save('lstm_model.h5')
print("lstm_model.h5 created successfully.")

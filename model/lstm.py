import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_lstm(history, n_features, horizon):
    model = Sequential([
        LSTM(128, input_shape=(history, n_features)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(horizon)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


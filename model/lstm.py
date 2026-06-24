import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout


def build_lstm(history, n_features, horizon):
    model = Sequential([
        Input(shape=(history, n_features)),
        LSTM(128),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(horizon)
    ])
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    return model


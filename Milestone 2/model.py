from tensorflow import keras
from keras import layers

def build_lstm_model(input_shape):
    
    model = keras.Sequential([
        layers.LSTM(64, input_shape=input_shape, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)  # Output: predicted RUL
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss='mse',
        metrics=['mae']
    )

    return model

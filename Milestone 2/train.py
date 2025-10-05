import numpy as np
import os
import matplotlib.pyplot as plt
from model import build_lstm_model
from tensorflow import keras

FD_LIST = ["FD001", "FD002", "FD003", "FD004"]

DATA_DIR = "sequences_data"
GRAPH_DIR = "graphs"
MODEL_DIR = "outputs"

# Create necessary folders
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

for FD_ID in FD_LIST:
    print(f"\n Processing dataset: {FD_ID}")

    # Load training and testing data
    X_train = np.load(os.path.join(DATA_DIR, f"X_train_{FD_ID}.npy"))
    y_train = np.load(os.path.join(DATA_DIR, f"y_train_{FD_ID}.npy"))
    X_test = np.load(os.path.join(DATA_DIR, f"X_test_{FD_ID}.npy"))
    y_test = np.load(os.path.join(DATA_DIR, f"y_test_{FD_ID}.npy"))

    print(f" Loaded dataset {FD_ID}")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")

    # Build the LSTM model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.summary()

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=64,
        verbose=1
    )

    # Save the model
    model.save(os.path.join(MODEL_DIR, f"lstm_model_{FD_ID}.h5"))
    print(f" Model saved to {MODEL_DIR}/lstm_model_{FD_ID}.h5")

    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title(f"Training vs Validation Loss ({FD_ID})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, f"loss_curve_{FD_ID}.png"))
    plt.close()
    print(f" Loss curve saved in {GRAPH_DIR}/loss_curve_{FD_ID}.png")

import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow import keras

FD_LIST = ["FD001", "FD002", "FD003", "FD004"]

DATA_DIR = "sequences_data"
GRAPH_DIR = "graphs"
MODEL_DIR = "outputs"

for FD_ID in FD_LIST:
    print(f"\n🔹 Evaluating dataset: {FD_ID}")

  
    X_test = np.load(os.path.join(DATA_DIR, f"X_test_{FD_ID}.npy"))
    y_test = np.load(os.path.join(DATA_DIR, f"y_test_{FD_ID}.npy"))

    model = keras.models.load_model(
        os.path.join(MODEL_DIR, f"lstm_model_{FD_ID}.h5"),
        compile=False
    )

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError()]
    )

    y_pred = model.predict(X_test)

    plt.figure(figsize=(8,5))
    plt.plot(y_test, label='Actual RUL', color='green')
    plt.plot(y_pred, label='Predicted RUL', color='red', linestyle='--')
    plt.title(f"Predicted vs Actual RUL ({FD_ID})")
    plt.xlabel("Test Samples")
    plt.ylabel("RUL")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, f"predicted_vs_actual_{FD_ID}.png"))
    plt.close()

    print(f" Evaluation complete for {FD_ID}")
    print(f" Graph saved in {GRAPH_DIR}/predicted_vs_actual_{FD_ID}.png")

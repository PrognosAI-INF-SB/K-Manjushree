from tensorflow import keras
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

MODEL_PATH = "models_m2/optimized_fd1.h5"
TEST_DATA_PATH = "processed/fd1_test_ws30.npz"
RESULTS_DIR = "results"

import os
os.makedirs(RESULTS_DIR, exist_ok=True)
PLOT_PATH = os.path.join(RESULTS_DIR, "pred_vs_actual.png")

# Load test data
data = np.load(TEST_DATA_PATH)
X_test, y_test = data['X'], data['y']

X_test = np.nan_to_num(X_test, nan=0.0)
y_test = np.nan_to_num(y_test, nan=0.0)

# Load model with custom_objects fix for mse
model = load_model(MODEL_PATH, custom_objects={'mse': keras.metrics.MeanSquaredError()})

# Predict
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)

def accuracy_within_tolerance(y_true, y_pred, tol=0.05):
    y_true_safe = np.maximum(y_true, 1e-6)  # avoid division by zero
    return np.mean(np.abs((y_pred - y_true_safe) / y_true_safe) <= tol) * 100

acc_5pct = accuracy_within_tolerance(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Accuracy within 5% tolerance: {acc_5pct:.2f}%")

# Plot predicted vs actual
plt.figure(figsize=(12,6))
plt.plot(y_test[:200], label='Actual RUL')
plt.plot(y_pred[:200], label='Predicted RUL')
plt.title("Predicted vs Actual RUL (first 200 samples)")
plt.xlabel("Sample Index")
plt.ylabel("RUL")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.show()

print(f"Plot saved at {PLOT_PATH}")

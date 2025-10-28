
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# === Configuration ===
fd_number = 1
window_size = 30
data_dir = "processed"
model_dir = "models_m2"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, f"optimized_fd{fd_number}_milestone4.h5")
scaler_path = os.path.join(model_dir, f"scaler_fd{fd_number}_milestone4.save")

# === Load data ===
train_file = f"{data_dir}/fd{fd_number}_train_ws{window_size}.npz"
val_file = f"{data_dir}/fd{fd_number}_val_ws{window_size}.npz"

train_data = np.load(train_file)
X_train, y_train = train_data["X"], train_data["y"]

if os.path.exists(val_file):
    val_data = np.load(val_file)
    X_val, y_val = val_data["X"], val_data["y"]
else:
    split_idx = int(0.8 * len(X_train))
    X_val, y_val = X_train[split_idx:], y_train[split_idx:]
    X_train, y_train = X_train[:split_idx], y_train[:split_idx]

# === Scale data ===
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

# Save scaler
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved at {scaler_path}")

# === Build LSTM model ===
model = Sequential([
    LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),
    
    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.summary()

# === Callbacks ===
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
]

# === Train model ===
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

print(f"✅ Model saved at {model_path}")

# === Evaluate model ===
y_val_pred = model.predict(X_val_scaled).flatten()
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
mae = mean_absolute_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)
accuracy = 100 - (rmse / np.max(y_val) * 100)

print(f"\n🔹 Model Evaluation — FD00{fd_number}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R² Score : {r2:.4f}")
print(f"Approx. Accuracy : {accuracy:.2f}%")

# Save metrics
results_folder = "results"
os.makedirs(results_folder, exist_ok=True)
results = pd.DataFrame({
    "FD_Set": [f"FD00{fd_number}"],
    "RMSE": [rmse],
    "MAE": [mae],
    "R2_Score": [r2],
    "Accuracy(%)": [accuracy]
})
results.to_csv(os.path.join(results_folder, f"model_performance_fd{fd_number}_milestone4.csv"), index=False)
print(f"📊 Metrics saved to {results_folder}")

# === Risk Thresholding & Alerts ===
warning_threshold = 50   # RUL < 50 cycles → Warning
critical_threshold = 20  # RUL < 20 cycles → Critical

alerts = []
for i, rul in enumerate(y_val_pred):
    if rul <= critical_threshold:
        alerts.append((i, rul, "CRITICAL"))
    elif rul <= warning_threshold:
        alerts.append((i, rul, "WARNING"))

alert_df = pd.DataFrame(alerts, columns=["Sample_Index", "Predicted_RUL", "Alert_Level"])
alert_df.to_csv(os.path.join(results_folder, f"alerts_fd{fd_number}_milestone4.csv"), index=False)
print(f"📢 Alerts saved to {results_folder}")
print(alert_df.head(10))


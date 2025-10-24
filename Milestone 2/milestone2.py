import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import Sequential, Input
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# -------------------------------
# Parameters
# -------------------------------
FD_NUMBER = 1
WINDOW_SIZE = 30
N_SPLITS = 5
EPOCHS = 50
BATCH_SIZE = 64

BASE_PATH = r"C:\Users\MANJUSHREE\Desktop\AI_PrognosAI\data\archive (2)\CMaps"
OUTPUT_DIR = "models_m2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
GRAPH_DIR = "graphs_m2"
os.makedirs(GRAPH_DIR, exist_ok=True)

# -------------------------------
# Load Data
# -------------------------------
train_file = os.path.join(BASE_PATH, f"train_FD00{FD_NUMBER}.txt")
test_file = os.path.join(BASE_PATH, f"test_FD00{FD_NUMBER}.txt")
rul_file = os.path.join(BASE_PATH, f"RUL_FD00{FD_NUMBER}.txt")

cols = ['engine_id', 'cycle'] + [f'operational_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
train_df = pd.read_csv(train_file, sep=r'\s+', header=None, names=cols, engine='python')
test_df = pd.read_csv(test_file, sep=r'\s+', header=None, names=cols, engine='python')
rul_df = pd.read_csv(rul_file, sep=r'\s+', header=None, names=['RUL'])

# Drop NaNs
train_df = train_df.dropna().reset_index(drop=True)
test_df = test_df.dropna().reset_index(drop=True)

# Compute RUL for train
rul_train = train_df.groupby('engine_id')['cycle'].max().reset_index()
rul_train.columns = ['engine_id', 'max_cycle']
train_df = train_df.merge(rul_train, on='engine_id')
train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']
train_df.drop('max_cycle', axis=1, inplace=True)

# Adjust test RUL
rul_df['engine_id'] = rul_df.index + 1
max_cycle_test = test_df.groupby('engine_id')['cycle'].max().reset_index()
max_cycle_test.columns = ['engine_id', 'max_cycle']
test_df = test_df.merge(max_cycle_test, on='engine_id')
test_df = test_df.merge(rul_df, on='engine_id')
test_df['RUL'] = test_df['RUL'] + test_df['max_cycle'] - test_df['cycle']
test_df.drop('max_cycle', axis=1, inplace=True)

# Drop irrelevant sensors
cols_to_drop = ['sensor_1', 'sensor_5', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
train_df.drop(cols_to_drop, axis=1, inplace=True)
test_df.drop(cols_to_drop, axis=1, inplace=True)

# Scale features
feature_cols = [c for c in train_df.columns if c not in ['engine_id', 'cycle', 'RUL']]
scaler = MinMaxScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])

# Trend features
for col in feature_cols:
    train_df[f'{col}_mean'] = train_df.groupby('engine_id')[col].transform('mean')
    train_df[f'{col}_std'] = train_df.groupby('engine_id')[col].transform('std')
    test_df[f'{col}_mean'] = test_df.groupby('engine_id')[col].transform('mean')
    test_df[f'{col}_std'] = test_df.groupby('engine_id')[col].transform('std')

def add_trend_features(df, cols):
    for col in cols:
        df[f'{col}_diff'] = df.groupby('engine_id')[col].diff().fillna(0)
        df[f'{col}_rolling_mean'] = df.groupby('engine_id')[col].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    return df

train_df = add_trend_features(train_df, feature_cols)
test_df = add_trend_features(test_df, feature_cols)

# Generate sequences
def gen_sequences(df, window):
    seqs, labels = [], []
    for eid in df['engine_id'].unique():
        subset = df[df['engine_id'] == eid]
        features = subset[[c for c in subset.columns if c not in ['engine_id', 'cycle', 'RUL']]].values
        rul = subset['RUL'].values
        for i in range(len(subset) - window):
            seqs.append(features[i:i + window])
            labels.append(rul[i + window])
    return np.array(seqs), np.array(labels)

X_train, y_train = gen_sequences(train_df, WINDOW_SIZE)
X_test, y_test = gen_sequences(test_df, WINDOW_SIZE)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# -------------------------------
# Build LSTM Model
# -------------------------------
def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        BatchNormalization(),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        BatchNormalization(),
        LSTM(32),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# -------------------------------
# 5-Fold Cross Validation
# -------------------------------
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
fold = 1
val_scores = []

for train_idx, val_idx in kf.split(X_train):
    print(f"\n--- Fold {fold}/{N_SPLITS} ---")
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate fold
    y_pred = model.predict(X_val).flatten()
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    val_scores.append((mae, r2))
    print(f"Fold {fold} MAE: {mae:.3f}, R²: {r2:.3f}")

    fold += 1

# Average CV metrics
avg_mae = np.mean([s[0] for s in val_scores])
avg_r2 = np.mean([s[1] for s in val_scores])
print(f"\nAverage CV MAE: {avg_mae:.3f}, Average CV R²: {avg_r2:.3f}")

# -------------------------------
# Train Final Model on Full Data
# -------------------------------
final_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

history = final_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# Evaluate final model
y_pred = final_model.predict(X_test).flatten()
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nFinal Model MAE: {mae:.3f}, R²: {r2:.3f}")

# Save final model
final_model_path = os.path.join(OUTPUT_DIR, f"optimized_fd{FD_NUMBER}.h5")
final_model.save(final_model_path)
print(f"Final optimized model saved at: {final_model_path}")

# Plot loss
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, f'loss_curve_fd{FD_NUMBER}.png'))
plt.close()
print(f"Loss curve saved at {GRAPH_DIR}/loss_curve_fd{FD_NUMBER}.png")

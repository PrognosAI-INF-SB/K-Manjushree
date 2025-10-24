import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

base_path = r"C:\Users\MANJUSHREE\Desktop\AI_PrognosAI\data\archive (2)\CMaps"
fd_number = 1
window_size = 30

train_file = os.path.join(base_path, f"train_FD00{fd_number}.txt")
test_file = os.path.join(base_path, f"test_FD00{fd_number}.txt")
rul_file = os.path.join(base_path, f"RUL_FD00{fd_number}.txt")

cols = ['engine_id', 'cycle'] + [f'operational_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]

train_df = pd.read_csv(train_file, sep=r'\s+', header=None, names=cols, engine='python')
test_df = pd.read_csv(test_file, sep=r'\s+', header=None, names=cols, engine='python')
rul_df = pd.read_csv(rul_file, sep=r'\s+', header=None, names=['RUL'])
train_df = train_df.dropna().reset_index(drop=True)
test_df = test_df.dropna().reset_index(drop=True)

rul_train = train_df.groupby('engine_id')['cycle'].max().reset_index()
rul_train.columns = ['engine_id', 'max_cycle']
train_df = train_df.merge(rul_train, on='engine_id')
train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']
train_df.drop('max_cycle', axis=1, inplace=True)

rul_df['engine_id'] = rul_df.index + 1
max_cycle_test = test_df.groupby('engine_id')['cycle'].max().reset_index()
max_cycle_test.columns = ['engine_id', 'max_cycle']
test_df = test_df.merge(max_cycle_test, on='engine_id')
test_df = test_df.merge(rul_df, on='engine_id')
test_df['RUL'] = test_df['RUL'] + test_df['max_cycle'] - test_df['cycle']
test_df.drop('max_cycle', axis=1, inplace=True)

cols_to_drop = ['sensor_1', 'sensor_5', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
train_df.drop(cols_to_drop, axis=1, inplace=True)
test_df.drop(cols_to_drop, axis=1, inplace=True)

feature_cols = [c for c in train_df.columns if c not in ['engine_id', 'cycle', 'RUL']]
scaler = MinMaxScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])

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

X_train, y_train = gen_sequences(train_df, window_size)
X_test, y_test = gen_sequences(test_df, window_size)

os.makedirs('processed', exist_ok=True)
np.savez(f'processed/fd{fd_number}_train_ws{window_size}.npz', X=X_train, y=y_train)
np.savez(f'processed/fd{fd_number}_test_ws{window_size}.npz', X=X_test, y=y_test)
train_df.to_csv(f'processed/fd{fd_number}_train_preprocessed.csv', index=False)
test_df.to_csv(f'processed/fd{fd_number}_test_preprocessed.csv', index=False)

print("Preprocessing and feature engineering complete.")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

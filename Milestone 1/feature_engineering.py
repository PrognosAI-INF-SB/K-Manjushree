import pandas as pd
import numpy as np
import os

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
RAW_PATH = "./processed_data"  
SAVE_PATH = "./sequences_data" 
WINDOW_SIZE = 30  

os.makedirs(SAVE_PATH, exist_ok=True)

def create_sequences(df, window_size=30, feature_cols=None):
    X, y = [], []
    engine_ids = df['unit_number'].unique()
    
    for eid in engine_ids:
        engine_data = df[df['unit_number'] == eid].reset_index(drop=True)
        features = engine_data[feature_cols].values
        rul = engine_data['RUL'].values
        
        for i in range(len(engine_data) - window_size + 1):
            X.append(features[i:i+window_size])
            y.append(rul[i+window_size-1])
    
    return np.array(X), np.array(y)

for ds in DATASETS:
    print(f"\n--- Processing {ds} ---")
    
    
    train = pd.read_csv(f"{RAW_PATH}/train_{ds}_preprocessed.csv")
    test = pd.read_csv(f"{RAW_PATH}/test_{ds}_preprocessed.csv")
    rul_test = pd.read_csv(f"{RAW_PATH}/RUL_{ds}.csv", header=None)  
    
    
    feature_cols = [c for c in train.columns if c not in ['unit_number', 'time_in_cycles', 'RUL']]
    
   
    X_train, y_train = create_sequences(train, window_size=WINDOW_SIZE, feature_cols=feature_cols)
    
    
    test_rul_full = []
    engine_ids = test['unit_number'].unique()
    for idx, eid in enumerate(engine_ids):
        engine_data = test[test['unit_number'] == eid].reset_index(drop=True)
        true_rul = rul_test.iloc[idx, 0]  
        seq_length = len(engine_data)
      
        test_rul = np.arange(seq_length-1, -1, -1) + true_rul
        test_rul_full.extend(test_rul)
    
    test['RUL'] = test_rul_full
    X_test, y_test = create_sequences(test, window_size=WINDOW_SIZE, feature_cols=feature_cols)
    
   
    np.save(f"{SAVE_PATH}/X_train_{ds}.npy", X_train)
    np.save(f"{SAVE_PATH}/y_train_{ds}.npy", y_train)
    np.save(f"{SAVE_PATH}/X_test_{ds}.npy", X_test)
    np.save(f"{SAVE_PATH}/y_test_{ds}.npy", y_test)
    
    print(f" {ds} sequences saved: X_train({X_train.shape}), y_train({y_train.shape}), X_test({X_test.shape}), y_test({y_test.shape})")


import numpy as np
import os
import matplotlib.pyplot as plt


data_dir = './sequences_data'

files = {
    'X_train_FD001': 'X_train_FD001.npy',
    'y_train_FD001': 'y_train_FD001.npy',
    'X_test_FD001': 'X_test_FD001.npy',
    'y_test_FD001': 'y_test_FD001.npy',
}


def load_npy(file_path):
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        raise FileNotFoundError(f"{file_path} not found!")


X_train = load_npy(os.path.join(data_dir, files['X_train_FD001']))
y_train = load_npy(os.path.join(data_dir, files['y_train_FD001']))
X_test = load_npy(os.path.join(data_dir, files['X_test_FD001']))
y_test = load_npy(os.path.join(data_dir, files['y_test_FD001']))

def check_integrity(X, y, name):
    print(f"--- Checking {name} ---")
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    
    
    print(f"Missing values in X: {np.isnan(X).sum()}")
    print(f"Missing values in y: {np.isnan(y).sum()}")
    
    
    print(f"X min: {X.min():.4f}, max: {X.max():.4f}, mean: {X.mean():.4f}")
    print(f"y min: {y.min()}, max: {y.max()}, mean: {y.mean():.2f}")
    print("\n")


check_integrity(X_train, y_train, "X_train & y_train")
check_integrity(X_test, y_test, "X_test & y_test")

import matplotlib.pyplot as plt

#graph for sensor 1
plt.figure(figsize=(10,4))
plt.plot(X_train[0][:,0], marker='o')
plt.title('Sensor 1 readings over first 30-cycle sequence')
plt.xlabel('Cycle')
plt.ylabel('Normalized Value')
plt.show()


plt.figure(figsize=(6,4))
plt.hist(y_train, bins=50, color='skyblue')
plt.title('Distribution of RUL in Training Set')
plt.xlabel('RUL')
plt.ylabel('Frequency')
plt.show()


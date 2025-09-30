import numpy as np


X_train = np.load("sequences_data/X_train_FD001.npy")
y_train = np.load("sequences_data/y_train_FD001.npy")

# Check shapes
print(X_train.shape)  
print(y_train.shape)  

print("First sequence (X_train[0]):\n", X_train[0])
print("RUL for first sequence:", y_train[0])

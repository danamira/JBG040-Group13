import numpy as np
import os
from pathlib import Path

path = Path.cwd()
os.chdir(path.parent)

print(Path.cwd())


# Import the data
X_train = np.load('data/preprocessed/remove_outliers/X_train.npy')
X_test = np.load('data/preprocessed/remove_outliers/X_test.npy')
y_train = np.load('data/preprocessed/remove_outliers/Y_train.npy')
y_test = np.load('data/preprocessed/remove_outliers/Y_test.npy')

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

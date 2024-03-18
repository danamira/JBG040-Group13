
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

# Change to parent directory
path = Path.cwd()
os.chdir(path.parent)
import cv2

# In[ ]:

# Import the data
X_train = np.load('data/X_train.npy')
X_test = np.load('data/X_test.npy')
y_train = np.load('data/Y_train.npy')
y_test = np.load('data/Y_test.npy')

print(X_train.shape)

outliers_train = [1345, 1410, 2794, 3644, 4877, 4943, 5534, 5742, 6312, 9234, 9653, 10394, 10938, 11379, 13394, 15356, 15448]
outliers_test = [2942, 3095, 3235, 4226, 4555, 4699, 4731, 5891, 6907]

def show_outliers(outlier_idx, data):
    for i, image in enumerate(data):
        image = image[0]
        if i in outlier_idx:
            plt.imshow(image)
            plt.show()

show_outliers(outliers_train, X_train)
show_outliers(outliers_test, X_test)




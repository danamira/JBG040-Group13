from matplotlib import pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.neighbors import LocalOutlierFactor

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


# Reshape the images if needed
# images = images.reshape(images.shape[0], -1)

def find_outliers(data):
    images = list()
    for image in data:
        images.append(image[0])
    images = np.array(images)

    # Reshape the images array to have two dimensions (flatten each image)
    num_images = images.shape[0]
    image_size = images.shape[1] * images.shape[2]
    images_flat = images.reshape(num_images, image_size)

    # Apply Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=10, contamination=0.001)  # Adjust parameters as needed
    outlier_scores = lof.fit_predict(images_flat)

    # Identify outliers
    outliers_indices = np.where(outlier_scores == -1)[0]
    outliers = list()
    for idx in outliers_indices:
        outliers.append(idx)
    return outliers


outliers_train = find_outliers(X_train)
outliers_test = find_outliers(X_test)

print(outliers_train)
print(outliers_test)

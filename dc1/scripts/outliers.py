from matplotlib import pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path
from os import path
from sklearn.neighbors import LocalOutlierFactor
import cv2

# Change to parent directory
# os.chdir(path.parent)

# print(Path.cwd())

dirOfThisFile=(os.path.dirname(os.path.realpath(__file__)))
os.chdir(dirOfThisFile)
cwd = os.getcwd()
# print(cwd)


# Import the data
X_train = np.load('../data/X_train.npy')
X_test = np.load('../data/X_test.npy')
y_train = np.load('../data/Y_train.npy')
y_test = np.load('../data/Y_test.npy')


def show_image(image):
    plt.imshow(image)
    plt.show()


def show_images_index(ind_list, data):
    for i, image in enumerate(data):
        image = image[0]
        if i in ind_list:
            show_image(image)


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


def remove_outliers_save(data, outliers_list, filename):
    np.save(f'../data/preprocessed/remove_outliers/{filename}.npy', np.delete(data, outliers_list, axis=0))


# ----------------------- #
# Run the functions above #
# ----------------------- #



if path.exists(path.join(cwd ,"../data/preprocessed/remove_outliers/")):
        print("Preprocessed data directory exists, files may be overwritten!")

else:
        # print(cwd)
        try:
            if not path.exists(path.join(cwd , "../data/preprocessed/")):
                os.mkdir(path.join(cwd, "../data/preprocessed/"))
            os.mkdir(path.join(cwd, "../data/preprocessed/remove_outliers/"))
        except:
            quit("Could not create the correct directory for pre-processed data. Please make directory `dc1/data/preprocessed/remove_outliers' relative to the project root and run this script again.")


train_outliers = find_outliers(X_train)
test_outliers = find_outliers(X_test)

remove_outliers_save(y_train, train_outliers, 'Y_train')
remove_outliers_save(y_test, test_outliers, 'Y_test')


remove_outliers_save(X_train, train_outliers, 'X_train')
remove_outliers_save(X_test, test_outliers, 'X_test')



